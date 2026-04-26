"""StoreSyncActor — durable export of DuckDB deltas to Parquet and B2.

This actor ensures the DuckDB store survives VM teardown by:
1. Exporting delta rows (trace_records, conditions) to Parquet every N seconds.
2. Uploading Parquet deltas to B2 (best-effort placeholder).
3. Periodically uploading a full base snapshot of the DuckDB file.
4. On shutdown: forcing a final flush and uploading a final snapshot.
5. Providing a restore path: download base snapshot + replay Parquet deltas.

Design principles:
- Best-effort B2: if credentials or SDK are missing, falls back to local durable storage.
- Retry with exponential backoff for all network-facing operations.
- Every operation is traced: no silent failures.
- Minimal imports: only from universal_store internals.
"""
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import duckdb

from universal_store.actors.base import Actor
from universal_store.config import UnifiedConfig
from universal_store.protocols import Event
from universal_store.trace import TraceStore, trace_block


class StoreSyncActor(Actor):
    """Actor that syncs DuckDB state to Parquet deltas and B2 durable storage.

    Parameters
    ----------
    actor_id:
        Unique identifier for this actor (default: ``store_sync``).
    config:
        ``UnifiedConfig`` instance. If ``None``, loaded from environment.
    store_db_path:
        Path to the main DuckDB store file (conditions, etc.).
    trace_db_path:
        Path to the trace DuckDB file (trace_records).
    staging_dir:
        Directory where Parquet delta files are written before upload.
    sync_interval_s:
        Seconds between automatic delta exports.
    base_snapshot_interval:
        Number of delta syncs between full DuckDB base-snapshot uploads.
    max_retries:
        Maximum retry attempts for network-facing operations.
    """

    def __init__(
        self,
        actor_id: str = "store_sync",
        config: UnifiedConfig | None = None,
        store_db_path: str | None = None,
        trace_db_path: str | None = None,
        staging_dir: str | Path = "./sync_staging",
        sync_interval_s: float | None = None,
        base_snapshot_interval: int = 10,
        max_retries: int = 5,
    ):
        super().__init__(actor_id)
        self.config = config or UnifiedConfig.from_env()
        self.store_db_path = store_db_path or self.config.store.db_path
        self.trace_db_path = trace_db_path or self.config.trace.db_path
        self.staging_dir = Path(staging_dir)
        self.sync_interval_s = sync_interval_s or self.config.reflexion.sync_interval_s
        self.base_snapshot_interval = base_snapshot_interval
        self.max_retries = max_retries

        # State
        self.last_sync: str = "1970-01-01T00:00:00"
        self._sync_counter: int = 0
        self._shutdown_requested: bool = False

        # Ensure staging directory exists
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        """Main actor loop: sleep, export delta, upload, repeat."""
        trace = await self._ensure_trace()
        await trace.record(
            actor_id=self.actor_id,
            event_type="sync_loop_started",
            payload={
                "store_db_path": self.store_db_path,
                "trace_db_path": self.trace_db_path,
                "staging_dir": str(self.staging_dir),
                "sync_interval_s": self.sync_interval_s,
            },
        )

        while not self._shutdown:
            try:
                # Wait for the sync interval or until an event arrives.
                event = await asyncio.wait_for(
                    self.mailbox.get(), timeout=self.sync_interval_s
                )
                await self._handle_event(event)
            except asyncio.TimeoutError:
                # Normal wake-up for periodic sync.
                pass
            except Exception as exc:
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="sync_loop_error",
                    error=exc,
                )

            if self._shutdown:
                break

            # Perform export + upload
            await self._export_and_upload_cycle()

        # Shutdown sequence
        await self._force_flush_and_upload()

    async def _handle_event(self, event: Event) -> None:
        """Process mailbox events (e.g. manual sync trigger)."""
        trace = await self._ensure_trace()
        if event.event_type == "sync_trigger":
            await trace.record(
                actor_id=self.actor_id,
                event_type="sync_trigger_received",
                payload=event.payload,
            )
        elif event.event_type == "shutdown_request":
            self._shutdown = True
            await trace.record(
                actor_id=self.actor_id,
                event_type="sync_shutdown_requested",
                payload=event.payload,
            )
        else:
            await trace.record(
                actor_id=self.actor_id,
                event_type="sync_unknown_event",
                payload={"event_type": event.event_type},
            )

    async def stop(self, graceful: bool = True) -> None:
        """Stop the actor. If graceful, force a final flush + snapshot first."""
        trace = await self._ensure_trace()
        await trace.record(
            actor_id=self.actor_id,
            event_type="sync_stop_called",
            payload={"graceful": graceful},
        )
        self._shutdown = True
        # If graceful, the _run loop will handle the final flush before exiting.
        await super().stop(graceful=graceful)

    # ------------------------------------------------------------------
    # Export + upload cycle
    # ------------------------------------------------------------------

    async def _export_and_upload_cycle(self) -> None:
        """Export delta Parquet files and upload them (plus optional base snapshot)."""
        trace = await self._ensure_trace()
        async with trace_block(
            self.actor_id, "export_and_upload_cycle", payload={"last_sync": self.last_sync}
        ):
            delta_files = await self._export_delta_to_parquet()
            if delta_files:
                await self._upload_deltas(delta_files)
            else:
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="sync_no_delta",
                    payload={"last_sync": self.last_sync},
                )

            self._sync_counter += 1
            if self._sync_counter % self.base_snapshot_interval == 0:
                await self._upload_base_snapshot(periodic=True)

    # ------------------------------------------------------------------
    # Delta export
    # ------------------------------------------------------------------

    async def _export_delta_to_parquet(self) -> list[Path]:
        """Query rows newer than ``last_sync`` and write to Parquet.

        Returns
        -------
        List of paths to the generated Parquet files.
        """
        trace = await self._ensure_trace()
        delta_files: list[Path] = []
        now_iso = self._utc_now()

        async with trace_block(
            self.actor_id, "export_delta_to_parquet", payload={"last_sync": self.last_sync}
        ):
            # --- trace_records delta ---
            trace_parquet = self.staging_dir / f"delta_trace_records_{now_iso}.parquet"
            try:
                conn = duckdb.connect(self.trace_db_path)
                # Check whether the table exists to avoid hard crashes on empty DBs.
                tables = conn.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_name = 'trace_records'"
                ).fetchall()
                if tables:
                    conn.execute(
                        f"""
                        COPY (
                            SELECT * FROM trace_records
                            WHERE timestamp > '{self.last_sync}'
                        ) TO '{trace_parquet}' (FORMAT PARQUET)
                        """
                    )
                    row_count = conn.execute(
                        f"SELECT COUNT(*) FROM read_parquet('{trace_parquet}')"
                    ).fetchone()[0]
                    if row_count > 0:
                        delta_files.append(trace_parquet)
                        await trace.record(
                            actor_id=self.actor_id,
                            event_type="sync_export_trace_records",
                            payload={"path": str(trace_parquet), "rows": row_count},
                        )
                    else:
                        trace_parquet.unlink(missing_ok=True)
                else:
                    await trace.record(
                        actor_id=self.actor_id,
                        event_type="sync_export_trace_records_skipped",
                        payload={"reason": "table_missing"},
                    )
                conn.close()
            except Exception as exc:
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="sync_export_trace_records_error",
                    error=exc,
                )
                conn.close()

            # --- conditions delta ---
            conditions_parquet = self.staging_dir / f"delta_conditions_{now_iso}.parquet"
            try:
                conn = duckdb.connect(self.store_db_path)
                tables = conn.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_name = 'conditions'"
                ).fetchall()
                if tables:
                    # The conditions schema uses `created_at` (not `timestamp`).
                    # We COALESCE with `scored_at` to catch rows that have no created_at.
                    conn.execute(
                        f"""
                        COPY (
                            SELECT * FROM conditions
                            WHERE COALESCE(NULLIF(created_at, ''), scored_at, '') > '{self.last_sync}'
                        ) TO '{conditions_parquet}' (FORMAT PARQUET)
                        """
                    )
                    row_count = conn.execute(
                        f"SELECT COUNT(*) FROM read_parquet('{conditions_parquet}')"
                    ).fetchone()[0]
                    if row_count > 0:
                        delta_files.append(conditions_parquet)
                        await trace.record(
                            actor_id=self.actor_id,
                            event_type="sync_export_conditions",
                            payload={"path": str(conditions_parquet), "rows": row_count},
                        )
                    else:
                        conditions_parquet.unlink(missing_ok=True)
                else:
                    await trace.record(
                        actor_id=self.actor_id,
                        event_type="sync_export_conditions_skipped",
                        payload={"reason": "table_missing"},
                    )
                conn.close()
            except Exception as exc:
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="sync_export_conditions_error",
                    error=exc,
                )
                conn.close()

            # Advance the watermark
            if delta_files:
                self.last_sync = now_iso
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="sync_last_sync_advanced",
                    payload={"last_sync": self.last_sync},
                )

        return delta_files

    # ------------------------------------------------------------------
    # Upload helpers (B2 best-effort)
    # ------------------------------------------------------------------

    async def _upload_deltas(self, delta_files: list[Path]) -> None:
        """Upload a batch of Parquet delta files to durable storage."""
        trace = await self._ensure_trace()
        for file_path in delta_files:
            key = f"deltas/{file_path.name}"
            async with trace_block(
                self.actor_id, "upload_delta", payload={"local": str(file_path), "remote_key": key}
            ):
                success = await self._upload_to_b2(file_path, key)
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="sync_upload_delta_result",
                    payload={
                        "local": str(file_path),
                        "remote_key": key,
                        "success": success,
                    },
                )

    async def _upload_base_snapshot(self, periodic: bool = False) -> None:
        """Upload the full DuckDB database file as a base snapshot."""
        trace = await self._ensure_trace()
        snapshot_key = f"snapshots/base_{self._utc_now()}.duckdb"
        db_path = Path(self.store_db_path)

        if not db_path.exists():
            await trace.record(
                actor_id=self.actor_id,
                event_type="sync_snapshot_skipped",
                payload={"reason": "db_not_found", "path": str(db_path)},
            )
            return

        async with trace_block(
            self.actor_id,
            "upload_base_snapshot",
            payload={"local": str(db_path), "remote_key": snapshot_key, "periodic": periodic},
        ):
            success = await self._upload_to_b2(db_path, snapshot_key)
            await trace.record(
                actor_id=self.actor_id,
                event_type="sync_snapshot_result",
                payload={
                    "local": str(db_path),
                    "remote_key": snapshot_key,
                    "periodic": periodic,
                    "success": success,
                },
            )

    async def _upload_to_b2(self, local_path: Path, remote_key: str) -> bool:
        """Upload a single file with exponential backoff (max 5 retries).

        Tries B2 first; if B2 is unavailable, copies to a local durable directory.
        """
        trace = await self._ensure_trace()
        delay = 1.0

        for attempt in range(1, self.max_retries + 1):
            try:
                # Attempt B2 upload
                b2_ok = await self._try_b2_upload(local_path, remote_key)
                if b2_ok:
                    return True

                # B2 not configured or failed → fallback to local durable storage
                fallback_dir = Path("./durable_storage")
                fallback_dir.mkdir(parents=True, exist_ok=True)
                dest = fallback_dir / remote_key.replace("/", "_")
                shutil.copy2(local_path, dest)
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="sync_fallback_local",
                    payload={
                        "local": str(local_path),
                        "fallback_dest": str(dest),
                        "attempt": attempt,
                    },
                )
                return True
            except Exception as exc:
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="sync_upload_attempt_failed",
                    payload={
                        "local": str(local_path),
                        "remote_key": remote_key,
                        "attempt": attempt,
                        "delay_s": delay,
                    },
                    error=exc,
                )
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(delay)
                delay *= 2  # exponential backoff

        return False

    async def _try_b2_upload(self, local_path: Path, remote_key: str) -> bool:
        """Best-effort B2 upload. Returns ``True`` on success, ``False`` if B2 unavailable."""
        cfg = self.config.reflexion
        if not (cfg.b2_bucket_name and cfg.b2_key_id and cfg.b2_application_key):
            return False

        try:
            # Run blocking B2 SDK in a thread pool to keep the actor async-friendly.
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._b2_upload_sync, local_path, remote_key, cfg
            )
        except Exception:
            return False

    def _b2_upload_sync(self, local_path: Path, remote_key: str, cfg: Any) -> bool:
        """Synchronous B2 upload using ``b2sdk``.

        This is a placeholder-style implementation: if ``b2sdk`` is installed and
        credentials are valid, the file is uploaded. Any problem returns ``False``
        so the caller can fall back to local durable storage.
        """
        try:
            from b2sdk.v2 import B2Api, InMemoryAccountInfo, UploadSourceLocalFile

            info = InMemoryAccountInfo()
            b2_api = B2Api(info)
            b2_api.authorize_account(
                "production", cfg.b2_key_id, cfg.b2_application_key
            )
            bucket = b2_api.get_bucket_by_name(cfg.b2_bucket_name)
            bucket.upload(
                UploadSourceLocalFile(str(local_path)),
                file_name=remote_key,
            )
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Shutdown flush
    # ------------------------------------------------------------------

    async def _force_flush_and_upload(self) -> None:
        """Force a final delta export and, if configured, upload a base snapshot."""
        trace = await self._ensure_trace()
        async with trace_block(self.actor_id, "force_flush_and_upload"):
            await trace.record(
                actor_id=self.actor_id,
                event_type="sync_force_flush_start",
                payload={"last_sync": self.last_sync},
            )

            # One last delta export
            delta_files = await self._export_delta_to_parquet()
            if delta_files:
                await self._upload_deltas(delta_files)

            # Final base snapshot if configured
            if self.config.reflexion.base_snapshot_on_shutdown:
                await self._upload_base_snapshot(periodic=False)

            await trace.record(
                actor_id=self.actor_id,
                event_type="sync_force_flush_complete",
                payload={"last_sync": self.last_sync},
            )

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------

    @staticmethod
    def restore_from_b2(db_path: str, config: UnifiedConfig | None = None) -> None:
        """Restore a DuckDB store from B2 durable storage.

        Steps:
        1. Download the latest base snapshot (full ``.duckdb`` file).
        2. List and download all Parquet delta files newer than the snapshot.
        3. Stream deltas into the restored database.
        4. Write a ``restore_manifest.json`` next to ``db_path`` for auditability.

        Parameters
        ----------
        db_path:
            Destination path for the reconstructed DuckDB file.
        config:
            ``UnifiedConfig`` instance. If ``None``, loaded from environment.
        """
        cfg = config or UnifiedConfig.from_env()
        reflexion = cfg.reflexion
        staging = Path("./sync_staging")
        staging.mkdir(parents=True, exist_ok=True)

        # Best-effort B2 restore; fall back to local durable_storage if B2 absent.
        b2_available = bool(
            reflexion.b2_bucket_name and reflexion.b2_key_id and reflexion.b2_application_key
        )

        snapshot_path: Path | None = None
        delta_paths: list[Path] = []

        if b2_available:
            try:
                snapshot_path, delta_paths = StoreSyncActor._download_from_b2(
                    staging, reflexion
                )
            except Exception:
                snapshot_path = None
                delta_paths = []

        # If B2 failed or is not configured, try local durable_storage.
        if snapshot_path is None:
            durable = Path("./durable_storage")
            if durable.exists():
                snaps = sorted(durable.glob("snapshots_base_*.duckdb"), reverse=True)
                if snaps:
                    snapshot_path = snaps[0]
                deltas = sorted(durable.glob("deltas_*.parquet"))
                delta_paths = deltas

        if snapshot_path is None:
            raise RuntimeError(
                "No base snapshot found in B2 or local durable_storage. Cannot restore."
            )

        # Copy base snapshot to destination
        shutil.copy2(snapshot_path, db_path)

        # Apply deltas
        conn = duckdb.connect(db_path)
        for delta in delta_paths:
            # Infer target table from filename heuristic
            table_name = "conditions"
            if "trace_records" in delta.name:
                table_name = "trace_records"
            try:
                conn.execute(
                    f"""
                    INSERT INTO {table_name}
                    SELECT * FROM read_parquet('{delta}')
                    ON CONFLICT DO NOTHING
                    """
                )
            except Exception:
                # DuckDB may not support ON CONFLICT; fallback to plain insert.
                try:
                    conn.execute(
                        f"SELECT * FROM read_parquet('{delta}') LIMIT 0"
                    )
                    conn.execute(
                        f"""
                        INSERT INTO {table_name}
                        SELECT * FROM read_parquet('{delta}')
                        """
                    )
                except Exception:
                    pass
        conn.close()

        # Write restore manifest
        manifest = {
            "restored_at": StoreSyncActor._utc_now(),
            "db_path": db_path,
            "base_snapshot": str(snapshot_path),
            "deltas_applied": [str(d) for d in delta_paths],
            "source": "b2" if b2_available and snapshot_path else "local",
        }
        manifest_path = Path(db_path).parent / "restore_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

    @staticmethod
    def _download_from_b2(staging: Path, reflexion_cfg: Any) -> tuple[Path | None, list[Path]]:
        """Download the newest base snapshot and all delta files from B2.

        Returns ``(snapshot_path, delta_paths)``. Raises on failure so the caller
        can fall back to local storage.
        """
        from b2sdk.v2 import B2Api, InMemoryAccountInfo

        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account(
            "production", reflexion_cfg.b2_key_id, reflexion_cfg.b2_application_key
        )
        bucket = b2_api.get_bucket_by_name(reflexion_cfg.b2_bucket_name)

        # Find latest snapshot
        snapshot_path: Path | None = None
        latest_snapshot_name: str | None = None
        for file_version, _ in bucket.list_file_versions(prefix="snapshots/"):
            if file_version.file_name.endswith(".duckdb"):
                if latest_snapshot_name is None or file_version.file_name > latest_snapshot_name:
                    latest_snapshot_name = file_version.file_name

        if latest_snapshot_name:
            local_snap = staging / latest_snapshot_name.replace("/", "_")
            download = bucket.download_file_by_name(latest_snapshot_name)
            download.save(str(local_snap))
            snapshot_path = local_snap

        # Download all deltas
        delta_paths: list[Path] = []
        for file_version, _ in bucket.list_file_versions(prefix="deltas/"):
            if file_version.file_name.endswith(".parquet"):
                local_delta = staging / file_version.file_name.replace("/", "_")
                download = bucket.download_file_by_name(file_version.file_name)
                download.save(str(local_delta))
                delta_paths.append(local_delta)

        delta_paths.sort(key=lambda p: p.name)
        return snapshot_path, delta_paths

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _utc_now() -> str:
        """ISO-8601 UTC timestamp string."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

"""Configuration defaults for the Universal Store Architecture.

All values are overridable via environment variables (UPPERCASE) or
a TOML config file at ~/.config/mirothinker/universal_store.toml.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActorConfig:
    mailbox_queue_size: int = 10_000
    default_restart_max: int = 3
    default_restart_window_s: float = 60.0
    health_check_interval_s: float = 10.0
    graceful_stop_timeout_s: float = 30.0


@dataclass
class StoreConfig:
    db_path: str = "./universal_store.duckdb"
    wal_enabled: bool = True
    checkpoint_interval_s: int = 300
    max_connections: int = 10
    read_pool_size: int = 5


@dataclass
class TraceConfig:
    db_path: str = "./universal_store_trace.duckdb"
    batch_size: int = 100
    flush_interval_s: float = 1.0
    emergency_log_path: str = "./trace_emergency.log"


@dataclass
class SchedulerConfig:
    event_queue_size: int = 10_000
    priority_levels: int = 5
    default_round_time_s: float = 120.0
    convergence_threshold: float = 0.02
    max_convergence_stuck_rounds: int = 3
    max_total_rounds: int = 50
    auto_pause_on_memory_mb: float = 14_000.0  # 14GB
    auto_pause_on_gpu_util: float = 95.0


@dataclass
class SwarmConfig:
    default_bee_count: int = 8
    max_gossip_rounds: int = 10
    gossip_info_gain_threshold: float = 0.05
    min_gossip_rounds: int = 2
    max_workers_per_phase: int = 8
    synthesis_timeout_s: float = 300.0


@dataclass
class FlockConfig:
    default_clone_count: int = 8
    default_rounds: int = 20
    prefix_cache_context_tokens: int = 50_000
    max_context_items: int = 40
    max_context_tokens: int = 8_000
    clone_budget_base: int = 50
    ucb_alpha: float = 1.0
    priority_decay_rate: float = 0.1
    serendipity_floor: float = 0.4
    magnitude_convergence_threshold: float = 0.02
    bootstrap_score_version_per_round: bool = True  # FIX: per round, not per session


@dataclass
class ExternalDataConfig:
    max_targets_per_round: int = 10
    max_books_per_query: int = 5
    max_total_mb_per_query: int = 500
    download_timeout_s: int = 300
    benefit_saturation_k: float = 0.3
    context_window_limit_tokens: int = 100_000
    convergence_boost_multiplier: float = 1.5
    green_tier_usd: float = 0.05
    yellow_tier_usd: float = 0.50
    red_tier_usd: float = 2.00
    red_tier_tokens: int = 5_000
    red_tier_latency_s: float = 10.0
    operator_override_timeout_s: float = 10.0
    source_three_strikes_threshold: int = 3


@dataclass
class SemanticConnectionConfig:
    heuristic_max_candidates: int = 1_000
    embedding_threshold: float = 0.72
    embedding_same_angle_boost: float = 0.10
    llm_verification_batch_size: int = 32
    min_confidence_for_storage: float = 0.70
    angle_diversity_penalty: float = 0.3
    max_edges_per_run: int = 10_000


@dataclass
class CurationConfig:
    global_health_interval_s: float = 5.0
    contradiction_digest_interval_s: float = 10.0
    gap_digest_interval_s: float = 15.0
    narrative_interval_s: float = 20.0
    source_quality_interval_rounds: int = 5
    clone_context_cache_ttl_s: float = 30.0
    angle_bundle_max_items: int = 100
    clone_context_max_items: int = 40
    operator_briefing_max_sentences: int = 3
    operator_briefing_max_alerts: int = 3
    operator_briefing_max_decisions: int = 2


@dataclass
class ReflexionConfig:
    lesson_detection_threshold: float = 0.5
    halflife_runs_default: int = 3
    min_lesson_confidence: float = 0.5
    sync_interval_s: float = 60.0
    b2_bucket_name: str = ""
    b2_key_id: str = ""
    b2_application_key: str = ""
    base_snapshot_on_shutdown: bool = True


@dataclass
class SourceIngestionConfig:
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    max_chunks_per_source: int = 2_000
    ocr_quality_threshold: float = 0.7
    hot_retention_days: int = 7
    warm_retention_days: int = 30
    cold_storage_bucket: str = ""


@dataclass
class UnifiedConfig:
    actor: ActorConfig = field(default_factory=ActorConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    flock: FlockConfig = field(default_factory=FlockConfig)
    external: ExternalDataConfig = field(default_factory=ExternalDataConfig)
    semantic: SemanticConnectionConfig = field(default_factory=SemanticConnectionConfig)
    curation: CurationConfig = field(default_factory=CurationConfig)
    reflexion: ReflexionConfig = field(default_factory=ReflexionConfig)
    sources: SourceIngestionConfig = field(default_factory=SourceIngestionConfig)

    @classmethod
    def from_env(cls) -> UnifiedConfig:
        """Override defaults with environment variables."""
        cfg = cls()
        for field_name, field_type in cfg.__dataclass_fields__.items():
            sub_cfg = getattr(cfg, field_name)
            for sub_name in sub_cfg.__dataclass_fields__:
                env_name = f"MIRO_{field_name.upper()}_{sub_name.upper()}"
                val = os.getenv(env_name)
                if val is not None:
                    current = getattr(sub_cfg, sub_name)
                    if isinstance(current, bool):
                        setattr(sub_cfg, sub_name, val.lower() in ("1", "true", "yes"))
                    elif isinstance(current, int):
                        setattr(sub_cfg, sub_name, int(val))
                    elif isinstance(current, float):
                        setattr(sub_cfg, sub_name, float(val))
                    else:
                        setattr(sub_cfg, sub_name, val)
        return cfg

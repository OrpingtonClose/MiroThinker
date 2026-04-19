#!/usr/bin/env python3
# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""YouTube Channel Harvester — bulk-download transcripts and comments for swarm processing.

Downloads transcripts and comments from YouTube channels using multiple backends
(Apify, Bright Data, yt-dlp, youtube-transcript-api) with automatic fallback.
Stores everything in Miro's cache infrastructure so it's recognized as a
first-class resource for later swarm synthesis.

Backends (tried in order):
1. Apify ``visita/youtube-scraper`` — transcripts + comments + metadata in one call
2. Bright Data YouTube Scraper API — separate datasets for channels/videos/comments
3. yt-dlp — subtitle download + comment extraction (needs cookies or proxy for cloud IPs)
4. youtube-transcript-api — Python library (needs proxy for cloud IPs)

Usage as library::

    from tools.youtube_harvester import Harvester, HarvestConfig

    harvester = Harvester(HarvestConfig(apify_token="...", bd_api_key="..."))
    result = await harvester.harvest_channel("@MorePlatesMoreDates", max_videos=100)
    print(result.summary())

Usage as CLI::

    python tools/youtube_harvester.py @MorePlatesMoreDates --max-videos 100
    python tools/youtube_harvester.py --status
    python tools/youtube_harvester.py --export --channel @MorePlatesMoreDates --out corpus.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("youtube-harvester")

# ── Configuration ────────────────────────────────────────────────────

REGISTRY_PATH = os.getenv(
    "HARVESTER_REGISTRY",
    os.path.expanduser("~/.miro/harvest_registry.duckdb"),
)
TRANSCRIPT_DIR = os.getenv(
    "HARVESTER_TRANSCRIPT_DIR",
    os.path.expanduser("~/.miro/transcripts"),
)
DEFAULT_DELAY = float(os.getenv("HARVESTER_DELAY", "2.0"))
DEFAULT_LANGUAGE = os.getenv("HARVESTER_LANGUAGE", "en")

# Apify
APIFY_ACTOR_YOUTUBE_TRANSCRIPT = os.getenv(
    "APIFY_ACTOR_YOUTUBE_TRANSCRIPT", "visita/youtube-scraper"
)

# Bright Data dataset IDs
BD_DATASET_CHANNELS = os.getenv("BD_DATASET_YOUTUBE_CHANNELS", "gd_lk538t2k2p1k3oos71")
BD_DATASET_VIDEOS = os.getenv("BD_DATASET_YOUTUBE_VIDEOS", "gd_lk56epmy2i5g7lzu0k")
BD_DATASET_COMMENTS = os.getenv("BD_DATASET_YOUTUBE_COMMENTS", "gd_lk9q0ew71spt1mxywf")

BD_POLL_INTERVAL = 5
BD_POLL_TIMEOUT = 180


# ── Data types ───────────────────────────────────────────────────────


@dataclass
class HarvestConfig:
    """Configuration for the harvester."""

    apify_token: str = ""
    bd_api_key: str = ""
    yt_dlp_cookies: str = ""  # path to cookies.txt
    yt_dlp_proxy: str = ""  # proxy URL for yt-dlp
    transcript_proxy: str = ""  # proxy URL for youtube-transcript-api
    language: str = DEFAULT_LANGUAGE
    delay: float = DEFAULT_DELAY
    max_concurrent: int = 5  # max concurrent Apify/BD requests
    fetch_comments: bool = True
    max_comments_per_video: int = 100

    @classmethod
    def from_env(cls) -> HarvestConfig:
        """Build config from environment variables."""
        return cls(
            apify_token=os.getenv("APIFY_API_TOKEN", ""),
            bd_api_key=os.getenv("BRIGHT_DATA_API_KEY", ""),
            yt_dlp_cookies=os.getenv("YT_DLP_COOKIES", ""),
            yt_dlp_proxy=os.getenv("YT_DLP_PROXY", ""),
            transcript_proxy=os.getenv("YOUTUBE_TRANSCRIPT_PROXY", ""),
            language=os.getenv("HARVESTER_LANGUAGE", DEFAULT_LANGUAGE),
            delay=float(os.getenv("HARVESTER_DELAY", str(DEFAULT_DELAY))),
            fetch_comments=os.getenv("HARVESTER_FETCH_COMMENTS", "true").lower()
            in ("true", "1", "yes"),
        )


@dataclass
class VideoRecord:
    """A single video in the content registry."""

    video_id: str
    channel_id: str
    channel_name: str
    title: str
    duration_s: int = 0
    view_count: int = 0
    upload_date: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    transcript: str = ""
    transcript_chars: int = 0
    comments: list[dict[str, Any]] = field(default_factory=list)
    comment_count: int = 0
    status: str = "pending"  # pending | downloaded | failed | swarmed
    backend_used: str = ""  # apify | bright_data | yt_dlp | transcript_api
    harvested_at: str = ""
    error: str = ""


@dataclass
class HarvestResult:
    """Result of a channel harvest operation."""

    channel_name: str = ""
    channel_id: str = ""
    total_listed: int = 0
    downloaded: int = 0
    failed: int = 0
    skipped: int = 0
    total_transcript_chars: int = 0
    total_comments: int = 0
    videos: list[VideoRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Channel: {self.channel_name} ({self.channel_id})\n"
            f"  Videos listed: {self.total_listed}\n"
            f"  Downloaded: {self.downloaded}\n"
            f"  Failed: {self.failed}\n"
            f"  Skipped (already cached): {self.skipped}\n"
            f"  Total transcript chars: {self.total_transcript_chars:,}\n"
            f"  Total comments: {self.total_comments:,}\n"
        )


# ── Topic extraction ─────────────────────────────────────────────────

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "steroids": ["steroid", "anabolic", "aas", "cycle", "pct", "sarm", "sarms"],
    "testosterone": ["testosterone", "trt", "test e", "test c", "enanthate", "cypionate"],
    "trenbolone": ["trenbolone", "tren"],
    "nandrolone": ["nandrolone", "deca", "npp"],
    "growth_hormone": ["hgh", "growth hormone", " gh ", "somatropin"],
    "peptides": ["peptide", "bpc-157", "bpc157", "tb-500", "tb500", "ipamorelin"],
    "insulin": ["insulin", "humalog", "novolog", "lantus", "metformin"],
    "nutrition": ["nutrition", "diet", "calories", "macros", "protein", "meal prep"],
    "training": ["training", "workout", "exercise", "hypertrophy", "strength"],
    "bloodwork": ["bloodwork", "blood work", "labs", "blood test", "hematocrit"],
    "harm_reduction": ["harm reduction", "side effect", "health", "safety"],
    "fat_loss": ["fat loss", "cutting", "weight loss", "shredding", "lean"],
    "supplements": ["supplement", "creatine", "pre-workout", "vitamin"],
    "sleep": ["sleep", "circadian", "melatonin", "insomnia"],
    "cardiovascular": ["cardio", "heart", "blood pressure", "cholesterol", "ldl", "hdl"],
    "liver": ["liver", "hepato", " alt ", " ast ", "tudca", "nac"],
    "kidney": ["kidney", "renal", "creatinine", "egfr"],
    "mental_health": ["depression", "anxiety", "mental health", "mood", "brain fog"],
    "hair_loss": ["hair loss", "finasteride", "dutasteride", "minoxidil", "mpb"],
    "fertility": ["fertility", "sperm", "hcg", "clomid", "enclomiphene"],
}


def extract_topics(title: str, description: str, tags: list[str]) -> list[str]:
    """Extract topic tags from video metadata using keyword matching.

    This is a fast heuristic — the librarian agent will do proper
    semantic topic classification later.

    Args:
        title: Video title.
        description: Video description.
        tags: Video tags.

    Returns:
        List of matched topic strings.
    """
    text = f" {title} {description} {' '.join(tags)} ".lower()
    return [topic for topic, kws in TOPIC_KEYWORDS.items() if any(kw in text for kw in kws)]


# ── SRT cleaning ─────────────────────────────────────────────────────


def clean_srt(raw: str) -> str:
    """Clean SRT/VTT subtitle text into readable paragraphs.

    Removes timing codes, sequence numbers, duplicate lines,
    and HTML/VTT tags.

    Args:
        raw: Raw SRT or VTT subtitle text.

    Returns:
        Clean paragraph text.
    """
    lines = raw.split("\n")
    clean_lines: list[str] = []
    prev_line = ""

    for line in lines:
        line = line.strip()
        if re.match(r"\d{2}:\d{2}:\d{2}", line):
            continue
        if re.match(r"^\d+$", line):
            continue
        if line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            continue
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"align:start position:\d+%", "", line)
        if not line or line == prev_line:
            continue
        prev_line = line
        clean_lines.append(line)

    text = " ".join(clean_lines)
    text = re.sub(r"\. ([A-Z])", r".\n\n\1", text)
    return text.strip()


# ── Channel listing (yt-dlp flat-playlist — works from cloud IPs) ────


def list_channel_videos(
    channel_input: str,
    max_videos: int = 0,
) -> tuple[str, str, list[dict[str, Any]]]:
    """List videos from a YouTube channel via yt-dlp flat-playlist.

    This endpoint still works from cloud IPs without cookies because
    it only fetches playlist metadata, not video content.

    Args:
        channel_input: Channel URL, handle (@name), or ID (UCxxxx).
        max_videos: Maximum videos to list (0 = all).

    Returns:
        Tuple of (channel_id, channel_name, list of video dicts).
    """
    if channel_input.startswith("UC"):
        url = f"https://www.youtube.com/channel/{channel_input}/videos"
    elif channel_input.startswith("@"):
        url = f"https://www.youtube.com/{channel_input}/videos"
    elif "youtube.com" in channel_input:
        url = channel_input
        if not url.endswith("/videos"):
            url = url.rstrip("/") + "/videos"
    else:
        url = f"https://www.youtube.com/@{channel_input}/videos"

    cmd = ["yt-dlp", "--flat-playlist", "--dump-json", url]
    if max_videos > 0:
        cmd.extend(["--playlist-end", str(max_videos)])

    logger.info("url=<%s>, max=<%d> | listing channel videos", url, max_videos)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.error("url=<%s>, error=<%s> | yt-dlp listing failed", url, exc)
        return "", "", []

    if result.returncode != 0:
        logger.error("url=<%s>, stderr=<%s> | yt-dlp listing failed", url, result.stderr[:200])
        return "", "", []

    videos: list[dict[str, Any]] = []
    channel_id = ""
    channel_name = ""

    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            info = json.loads(line)
            videos.append(info)
            if not channel_id:
                channel_id = info.get("channel_id") or ""
            if not channel_name:
                channel_name = info.get("channel") or info.get("uploader") or ""
        except json.JSONDecodeError:
            continue

    # Flat-playlist often doesn't include channel metadata — derive from input
    if not channel_name:
        channel_name = channel_input.lstrip("@")
    if not channel_id:
        channel_id = channel_input

    logger.info(
        "channel=<%s>, count=<%d> | listing complete",
        channel_name, len(videos),
    )
    return channel_id, channel_name, videos


# ══════════════════════════════════════════════════════════════════════
# BACKEND 1: Apify (visita/youtube-scraper)
# Gets transcript + comments + metadata in a single actor run.
# $8 per 1000 videos. Best value for bulk harvesting.
# ══════════════════════════════════════════════════════════════════════


async def _apify_fetch_batch(
    video_ids: list[str],
    config: HarvestConfig,
    include_comments: bool = True,
) -> list[dict[str, Any]]:
    """Fetch transcripts + comments for a batch of videos via Apify.

    Args:
        video_ids: List of YouTube video IDs.
        config: Harvest configuration.
        include_comments: Whether to fetch comments.

    Returns:
        List of result dicts with transcript, comments, and metadata.
    """
    if not config.apify_token:
        return []

    import httpx

    actor_id = APIFY_ACTOR_YOUTUBE_TRANSCRIPT.replace("/", "~")
    # visita/youtube-scraper uses runMode + scrapeConfig for specific video IDs
    run_input: dict[str, Any] = {
        "runMode": "scrape",
        "scrapeConfig": {
            "videoIDs": video_ids,
        },
        "lang": config.language,
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=15.0)) as client:
            # Start actor run
            resp = await client.post(
                f"https://api.apify.com/v2/acts/{actor_id}/runs",
                params={"token": config.apify_token, "timeout": 300},
                json=run_input,
            )
            if resp.status_code not in (200, 201):
                logger.warning(
                    "status=<%d>, body=<%s> | Apify actor start failed",
                    resp.status_code, resp.text[:200],
                )
                return []

            run_data = resp.json().get("data", {})
            run_id = run_data.get("id", "")
            if not run_id:
                logger.warning("run_data=<%s> | no run ID in Apify response", run_data)
                return []

            # Poll for completion
            dataset_id = run_data.get("defaultDatasetId", "")
            status = run_data.get("status", "")
            start_time = time.monotonic()

            while status in ("READY", "RUNNING") and time.monotonic() - start_time < 300:
                await asyncio.sleep(5)
                status_resp = await client.get(
                    f"https://api.apify.com/v2/actor-runs/{run_id}",
                    params={"token": config.apify_token},
                )
                if status_resp.status_code == 200:
                    run_info = status_resp.json().get("data", {})
                    status = run_info.get("status", "")
                    dataset_id = run_info.get("defaultDatasetId", dataset_id)

            if status != "SUCCEEDED":
                logger.warning("run_id=<%s>, status=<%s> | Apify run did not succeed", run_id, status)
                return []

            # Fetch results from dataset
            if not dataset_id:
                return []

            items_resp = await client.get(
                f"https://api.apify.com/v2/datasets/{dataset_id}/items",
                params={"token": config.apify_token, "format": "json"},
            )
            if items_resp.status_code == 200:
                items = items_resp.json()
                logger.info(
                    "run_id=<%s>, items=<%d> | Apify batch complete",
                    run_id, len(items),
                )
                return items if isinstance(items, list) else []

    except Exception as exc:
        logger.warning("error=<%s> | Apify fetch failed", exc)

    return []


def _parse_apify_result(item: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]]]:
    """Parse an Apify visita/youtube-scraper result item.

    The actor output schema:
      - videoId: str
      - title: str
      - channel: str
      - views: str (e.g. "1.2M views")
      - likes: str (e.g. "10K likes")
      - transcriptMerged: str (full transcript as single text block)
      - comments: str (JSON string of [{author, text, likes}, ...])
      - error: str | None

    Args:
        item: Raw Apify result dict.

    Returns:
        Tuple of (transcript_text, video_title, comments_list).
    """
    # Primary field is transcriptMerged (single text block)
    transcript = item.get("transcriptMerged") or ""

    # Fallback: captions field is a JSON string of [{start, text}, ...] segments
    if not transcript:
        captions_raw = item.get("captions") or item.get("transcript") or ""
        if isinstance(captions_raw, str) and captions_raw.strip():
            try:
                parsed_captions = json.loads(captions_raw)
                if isinstance(parsed_captions, list):
                    transcript = " ".join(
                        seg.get("text", "") for seg in parsed_captions
                        if isinstance(seg, dict)
                    )
            except (json.JSONDecodeError, TypeError):
                # Not JSON — use the raw string as transcript text
                transcript = captions_raw
        elif isinstance(captions_raw, list):
            transcript = " ".join(
                seg.get("text", "") for seg in captions_raw
                if isinstance(seg, dict)
            )

    title = item.get("title", "")

    comments: list[dict[str, Any]] = []
    raw_comments = item.get("comments") or ""

    # comments field is a JSON string in the visita/youtube-scraper output
    if isinstance(raw_comments, str) and raw_comments.strip():
        try:
            parsed = json.loads(raw_comments)
            if isinstance(parsed, list):
                raw_comments = parsed
            else:
                raw_comments = []
        except (json.JSONDecodeError, TypeError):
            raw_comments = []
    elif not isinstance(raw_comments, list):
        raw_comments = []

    for c in raw_comments:
        if isinstance(c, dict):
            comments.append({
                "author": c.get("author") or c.get("user") or "Anonymous",
                "text": c.get("text") or c.get("comment") or "",
                "likes": c.get("likes") or c.get("like_count") or 0,
                "replies": c.get("replies") or c.get("reply_count") or 0,
                "date": c.get("date") or c.get("published_at") or "",
            })

    return transcript, title, comments


# ══════════════════════════════════════════════════════════════════════
# BACKEND 2: Bright Data YouTube Scraper API
# Separate datasets for videos and comments.
# ══════════════════════════════════════════════════════════════════════


async def _bd_scrape(
    dataset_id: str,
    inputs: list[dict[str, Any]],
    api_key: str,
) -> list[dict[str, Any]]:
    """Trigger a Bright Data scraper and poll for results.

    Args:
        dataset_id: Bright Data dataset identifier.
        inputs: List of input dicts (each with a "url" key).
        api_key: Bright Data API key.

    Returns:
        List of result dicts.
    """
    if not api_key:
        return []

    import httpx

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Trigger
            resp = await client.post(
                "https://api.brightdata.com/datasets/v3/trigger",
                params={"dataset_id": dataset_id},
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=inputs,
            )
            if resp.status_code != 200:
                logger.warning(
                    "dataset=<%s>, status=<%d> | BD trigger failed",
                    dataset_id, resp.status_code,
                )
                return []

            response_data = resp.json()
            if isinstance(response_data, list):
                # Synchronous response — data returned directly
                return response_data
            snapshot_id = response_data.get("snapshot_id", "")
            if not snapshot_id:
                return []

            # Poll
            start_time = time.monotonic()
            while time.monotonic() - start_time < BD_POLL_TIMEOUT:
                await asyncio.sleep(BD_POLL_INTERVAL)
                poll_resp = await client.get(
                    f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}",
                    params={"format": "json"},
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if poll_resp.status_code == 200:
                    data = poll_resp.json()
                    if isinstance(data, list):
                        return data
                    status = data.get("status", "")
                    if status in ("ready", "complete"):
                        return data.get("data", [])
                elif poll_resp.status_code != 202:
                    break

            logger.warning("dataset=<%s>, snapshot=<%s> | BD poll timeout", dataset_id, snapshot_id)
    except Exception as exc:
        logger.warning("dataset=<%s>, error=<%s> | BD scrape failed", dataset_id, exc)

    return []


async def _bd_fetch_comments(
    video_ids: list[str],
    api_key: str,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch comments for multiple videos via Bright Data.

    Args:
        video_ids: List of YouTube video IDs.
        api_key: Bright Data API key.

    Returns:
        Dict mapping video_id to list of comment dicts.
    """
    inputs = [
        {"url": f"https://www.youtube.com/watch?v={vid}"}
        for vid in video_ids
    ]
    results = await _bd_scrape(BD_DATASET_COMMENTS, inputs, api_key)

    comments_by_video: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        vid_url = r.get("url", "")
        vid_id = _extract_video_id_from_url(vid_url)
        if not vid_id:
            continue
        comment = {
            "author": r.get("comment_user") or "Anonymous",
            "text": r.get("comment") or "",
            "likes": r.get("likes") or 0,
            "replies": r.get("replies") or 0,
            "date": r.get("comment_date") or "",
        }
        if comment["text"]:
            comments_by_video.setdefault(vid_id, []).append(comment)

    return comments_by_video


# ══════════════════════════════════════════════════════════════════════
# BACKEND 3: yt-dlp (with cookies/proxy)
# ══════════════════════════════════════════════════════════════════════


def _ytdlp_transcript(video_id: str, config: HarvestConfig) -> str:
    """Download transcript via yt-dlp.

    Args:
        video_id: YouTube video ID.
        config: Harvest config with optional cookies/proxy.

    Returns:
        Clean transcript text, or empty string on failure.
    """
    import tempfile

    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", f"{config.language}.*",
        "--sub-format", "vtt",
        "--convert-subs", "srt",
    ]
    if config.yt_dlp_cookies:
        cmd.extend(["--cookies", config.yt_dlp_cookies])
    if config.yt_dlp_proxy:
        cmd.extend(["--proxy", config.yt_dlp_proxy])

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd.extend(["-o", os.path.join(tmpdir, "subs"), url])
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ""

        srt_files = list(Path(tmpdir).glob("*.srt"))
        vtt_files = list(Path(tmpdir).glob("*.vtt"))
        sub_files = srt_files or vtt_files
        if not sub_files:
            return ""

        raw = sub_files[0].read_text(encoding="utf-8", errors="replace")
        return clean_srt(raw)


def _ytdlp_comments(video_id: str, config: HarvestConfig) -> list[dict[str, Any]]:
    """Extract comments via yt-dlp.

    Args:
        video_id: YouTube video ID.
        config: Harvest config with optional cookies/proxy.

    Returns:
        List of comment dicts.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-download",
        "--write-comments",
        "--extractor-args",
        f"youtube:comment_sort=top;max_comments={config.max_comments_per_video}",
    ]
    if config.yt_dlp_cookies:
        cmd.extend(["--cookies", config.yt_dlp_cookies])
    if config.yt_dlp_proxy:
        cmd.extend(["--proxy", config.yt_dlp_proxy])
    cmd.append(url)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            return []
        info = json.loads(result.stdout)
        raw_comments = info.get("comments", [])
        return [
            {
                "author": c.get("author", "Anonymous"),
                "text": c.get("text", ""),
                "likes": c.get("like_count", 0),
                "replies": 0,
                "date": str(c.get("timestamp", "")),
            }
            for c in raw_comments[:config.max_comments_per_video]
            if c.get("text")
        ]
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return []


# ══════════════════════════════════════════════════════════════════════
# BACKEND 4: youtube-transcript-api (with proxy)
# ══════════════════════════════════════════════════════════════════════


def _transcript_api_fetch(video_id: str, config: HarvestConfig) -> str:
    """Fetch transcript via youtube-transcript-api Python library.

    Args:
        video_id: YouTube video ID.
        config: Harvest config with optional proxy.

    Returns:
        Transcript text, or empty string on failure.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        kwargs: dict[str, Any] = {}
        if config.transcript_proxy:
            from youtube_transcript_api.proxies import GenericProxyConfig
            kwargs["proxy_config"] = GenericProxyConfig(
                https_url=config.transcript_proxy,
            )

        api = YouTubeTranscriptApi(**kwargs)
        transcript = api.fetch(video_id, languages=[config.language, "en"])
        return " ".join(s.text for s in transcript.snippets)
    except Exception as exc:
        logger.debug("video_id=<%s>, error=<%s> | transcript-api failed", video_id, exc)
        return ""


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════


def _extract_video_id_from_url(url: str) -> str:
    """Extract video ID from a YouTube URL."""
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
        r"(?:shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    if re.match(r"^[a-zA-Z0-9_-]{11}$", url):
        return url
    return ""


def _format_comments_md(comments: list[dict[str, Any]]) -> str:
    """Format comments as markdown for corpus inclusion.

    Args:
        comments: List of comment dicts.

    Returns:
        Markdown-formatted comment section.
    """
    if not comments:
        return ""

    display_comments = comments[:100]
    lines = [f"\n### Comments ({len(display_comments)} extracted)\n"]
    for i, c in enumerate(display_comments, 1):
        author = c.get("author", "Anonymous")
        text = c.get("text", "").replace("\n", " ").strip()
        likes = c.get("likes", 0)
        if text:
            like_str = f" [{likes} likes]" if likes else ""
            lines.append(f"- **{author}**{like_str}: {text[:500]}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# MAIN HARVESTER
# ══════════════════════════════════════════════════════════════════════


class Harvester:
    """Multi-backend YouTube channel harvester.

    Tries backends in order (Apify → Bright Data → yt-dlp → transcript-api)
    until one succeeds for each video.

    Args:
        config: Harvest configuration.
        cache_fn: Optional function to store content in Miro's cache.
            Signature: ``cache_fn(url, content, source_type, title, summary, tags, metadata)``
    """

    def __init__(
        self,
        config: HarvestConfig | None = None,
        cache_fn: Any | None = None,
    ) -> None:
        self.config = config or HarvestConfig.from_env()
        self.cache_fn = cache_fn

    async def harvest_channel(
        self,
        channel_input: str,
        max_videos: int = 0,
    ) -> HarvestResult:
        """Harvest all transcripts and comments from a YouTube channel.

        Args:
            channel_input: Channel URL, handle (@name), or ID (UCxxxx).
            max_videos: Maximum videos to process (0 = all).

        Returns:
            HarvestResult with all video records and summary stats.
        """
        result = HarvestResult()

        # Step 1: List channel videos
        channel_id, channel_name, videos = list_channel_videos(
            channel_input, max_videos,
        )
        if not videos:
            result.errors.append(f"No videos found for {channel_input}")
            return result

        result.channel_id = channel_id
        result.channel_name = channel_name
        result.total_listed = len(videos)

        # Build video records
        records: list[VideoRecord] = []
        for v in videos:
            vid = v.get("id", "")
            if not vid:
                continue
            tags = v.get("tags") or []
            if isinstance(tags, str):
                tags = [tags]
            title = v.get("title", "Unknown") or "Unknown"
            desc = (v.get("description") or "")[:2000]
            records.append(VideoRecord(
                video_id=vid,
                channel_id=channel_id,
                channel_name=channel_name,
                title=title,
                duration_s=v.get("duration") or 0,
                view_count=v.get("view_count") or 0,
                upload_date=v.get("upload_date", "") or "",
                description=desc,
                tags=tags[:20],
                topics=extract_topics(title, desc, tags),
            ))

        # Step 2: Check cache for already-downloaded videos
        cached_ids = set()
        if self.cache_fn:
            try:
                # Import cache_get to check existing entries
                sys.path.insert(0, os.path.join(
                    os.path.dirname(__file__), "..", "apps", "strands-agent",
                ))
                from cache import cache_get
                for rec in records:
                    cached = cache_get(url=f"youtube-transcript://{rec.video_id}/{self.config.language}")
                    if cached and cached.get("content"):
                        cached_ids.add(rec.video_id)
                        rec.status = "downloaded"
                        rec.backend_used = "cache"
                        result.skipped += 1
            except ImportError:
                pass

        pending = [r for r in records if r.video_id not in cached_ids]

        if not pending:
            logger.info("channel=<%s> | all %d videos already cached", channel_name, len(records))
            result.videos = records
            return result

        logger.info(
            "channel=<%s>, pending=<%d>, cached=<%d> | starting harvest",
            channel_name, len(pending), len(cached_ids),
        )

        # Step 3: Try backends in order
        remaining = list(pending)

        # Backend 1: Apify (batch — most efficient)
        if self.config.apify_token and remaining:
            remaining = await self._try_apify(remaining)

        # Backend 2: Bright Data (individual video comments)
        if self.config.bd_api_key and remaining:
            remaining = await self._try_bright_data(remaining)

        # Backend 3: yt-dlp (sequential, needs cookies/proxy)
        if remaining and (self.config.yt_dlp_cookies or self.config.yt_dlp_proxy):
            remaining = await self._try_ytdlp(remaining)

        # Backend 4: youtube-transcript-api (sequential, needs proxy)
        if remaining and self.config.transcript_proxy:
            remaining = await self._try_transcript_api(remaining)

        # Mark remaining as failed
        for rec in remaining:
            if rec.status == "pending":
                rec.status = "failed"
                rec.error = "all backends failed"
                result.failed += 1

        # Step 4: Store in cache
        for rec in records:
            if rec.status == "downloaded" and rec.backend_used != "cache":
                result.downloaded += 1
                result.total_transcript_chars += rec.transcript_chars
                result.total_comments += rec.comment_count
                self._store_in_cache(rec)

        result.videos = records
        return result

    async def _try_apify(self, records: list[VideoRecord]) -> list[VideoRecord]:
        """Try Apify backend for all pending records.

        Args:
            records: Pending video records.

        Returns:
            List of records that still need processing (Apify failed for them).
        """
        video_ids = [r.video_id for r in records]
        remaining: list[VideoRecord] = []

        # Process in batches of 50
        batch_size = 50
        results_by_id: dict[str, dict[str, Any]] = {}

        for i in range(0, len(video_ids), batch_size):
            batch = video_ids[i:i + batch_size]
            logger.info(
                "batch=<%d-%d/%d> | Apify batch fetch",
                i + 1, min(i + batch_size, len(video_ids)), len(video_ids),
            )
            items = await _apify_fetch_batch(batch, self.config)
            for item in items:
                vid = item.get("videoId") or item.get("id") or ""
                if vid:
                    results_by_id[vid] = item

            if self.config.delay > 0 and i + batch_size < len(video_ids):
                await asyncio.sleep(self.config.delay)

        for rec in records:
            if rec.video_id in results_by_id:
                item = results_by_id[rec.video_id]
                transcript, title, comments = _parse_apify_result(item)
                if transcript:
                    rec.transcript = transcript
                    rec.transcript_chars = len(transcript)
                    if self.config.fetch_comments:
                        rec.comments = comments
                        rec.comment_count = len(comments)
                    rec.status = "downloaded"
                    rec.backend_used = "apify"
                    rec.harvested_at = datetime.now(timezone.utc).isoformat()
                    if title:
                        rec.title = title
                else:
                    remaining.append(rec)
            else:
                remaining.append(rec)

        logger.info(
            "apify_success=<%d>, apify_failed=<%d> | Apify batch done",
            len(records) - len(remaining), len(remaining),
        )
        return remaining

    async def _try_bright_data(self, records: list[VideoRecord]) -> list[VideoRecord]:
        """Try Bright Data backend — mainly for comments.

        Note: Bright Data doesn't have a direct transcript endpoint.
        We use it for comments on videos where we already have transcripts,
        or for video metadata.

        Args:
            records: Pending video records.

        Returns:
            List of records still pending.
        """
        video_ids = [r.video_id for r in records]

        if self.config.fetch_comments:
            logger.info("videos=<%d> | fetching comments via Bright Data", len(video_ids))
            comments_map = await _bd_fetch_comments(video_ids, self.config.bd_api_key)
            for rec in records:
                if rec.video_id in comments_map:
                    rec.comments = comments_map[rec.video_id]
                    rec.comment_count = len(rec.comments)

        # BD doesn't provide transcripts directly — return all as remaining
        return records

    async def _try_ytdlp(self, records: list[VideoRecord]) -> list[VideoRecord]:
        """Try yt-dlp backend (sequential).

        Args:
            records: Pending video records.

        Returns:
            List of records still pending.
        """
        remaining: list[VideoRecord] = []

        for i, rec in enumerate(records):
            logger.info(
                "progress=<%d/%d>, video_id=<%s> | yt-dlp attempt",
                i + 1, len(records), rec.video_id,
            )
            transcript = _ytdlp_transcript(rec.video_id, self.config)
            if transcript:
                rec.transcript = transcript
                rec.transcript_chars = len(transcript)
                rec.status = "downloaded"
                rec.backend_used = "yt_dlp"
                rec.harvested_at = datetime.now(timezone.utc).isoformat()

                if self.config.fetch_comments and not rec.comments:
                    rec.comments = _ytdlp_comments(rec.video_id, self.config)
                    rec.comment_count = len(rec.comments)
            else:
                remaining.append(rec)

            if self.config.delay > 0 and i < len(records) - 1:
                await asyncio.sleep(self.config.delay)

        return remaining

    async def _try_transcript_api(self, records: list[VideoRecord]) -> list[VideoRecord]:
        """Try youtube-transcript-api backend (sequential).

        Args:
            records: Pending video records.

        Returns:
            List of records still pending.
        """
        remaining: list[VideoRecord] = []

        for i, rec in enumerate(records):
            transcript = _transcript_api_fetch(rec.video_id, self.config)
            if transcript:
                rec.transcript = transcript
                rec.transcript_chars = len(transcript)
                rec.status = "downloaded"
                rec.backend_used = "transcript_api"
                rec.harvested_at = datetime.now(timezone.utc).isoformat()
            else:
                remaining.append(rec)

            if self.config.delay > 0 and i < len(records) - 1:
                await asyncio.sleep(self.config.delay)

        return remaining

    def _store_in_cache(self, rec: VideoRecord) -> None:
        """Store a harvested video's transcript and comments in Miro's cache.

        Args:
            rec: Completed video record.
        """
        if not self.cache_fn:
            return

        # Build corpus-ready content
        content_parts = [
            f"# {rec.title}\n",
            f"**Channel:** {rec.channel_name}\n",
            f"**Video ID:** {rec.video_id}\n",
            f"**Duration:** {rec.duration_s // 60}m\n",
            f"**Views:** {rec.view_count:,}\n",
            f"**Upload date:** {rec.upload_date}\n",
            f"**Topics:** {', '.join(rec.topics)}\n",
            f"**Backend:** {rec.backend_used}\n\n",
            "---\n\n",
            "## Transcript\n\n",
            rec.transcript,
        ]

        if rec.comments and self.config.fetch_comments:
            content_parts.append(_format_comments_md(rec.comments))

        content = "\n".join(content_parts)

        try:
            self.cache_fn(
                url=f"youtube-transcript://{rec.video_id}/{self.config.language}",
                content=content,
                content_type="text/plain",
                source_type="youtube_harvest",
                title=f"[{rec.channel_name}] {rec.title}",
                summary=rec.transcript[:500] if rec.transcript else "",
                tags=["youtube", "transcript", "harvest", rec.channel_name]
                + rec.topics,
                ttl=365 * 24 * 3600,  # 1 year — harvested content is permanent corpus
                metadata={
                    "video_id": rec.video_id,
                    "channel_id": rec.channel_id,
                    "channel_name": rec.channel_name,
                    "duration_s": rec.duration_s,
                    "view_count": rec.view_count,
                    "upload_date": rec.upload_date,
                    "comment_count": rec.comment_count,
                    "backend": rec.backend_used,
                    "topics": rec.topics,
                },
            )
            logger.info(
                "video_id=<%s>, chars=<%d>, comments=<%d> | cached in Miro",
                rec.video_id, rec.transcript_chars, rec.comment_count,
            )
        except Exception as exc:
            logger.warning(
                "video_id=<%s>, error=<%s> | cache store failed",
                rec.video_id, exc,
            )


def export_corpus(
    records: list[VideoRecord],
    output_path: str = "",
    max_chars: int = 0,
    include_comments: bool = True,
) -> str:
    """Export harvested videos as a single corpus file for swarm processing.

    Args:
        records: Video records to export.
        output_path: Output file path (empty = return as string).
        max_chars: Maximum corpus size in chars (0 = unlimited).
        include_comments: Whether to include comments in the corpus.

    Returns:
        Corpus text (if no output_path) or path to exported file.
    """
    downloaded = [r for r in records if r.status == "downloaded" and r.transcript]
    if not downloaded:
        return ""

    parts: list[str] = []
    total_chars = 0

    for v in downloaded:
        section = (
            f"## {v.title}\n\n"
            f"**Channel:** {v.channel_name} | "
            f"**Views:** {v.view_count:,} | "
            f"**Duration:** {v.duration_s // 60}m | "
            f"**Date:** {v.upload_date} | "
            f"**Topics:** {', '.join(v.topics)}\n\n"
            f"---\n\n"
            f"{v.transcript}\n"
        )
        if include_comments and v.comments:
            section += _format_comments_md(v.comments) + "\n"
        section += "\n---\n\n"

        if max_chars > 0 and total_chars + len(section) > max_chars:
            break

        parts.append(section)
        total_chars += len(section)

    header = (
        f"# YouTube Corpus Export\n\n"
        f"**Videos:** {len(parts)}\n"
        f"**Total chars:** {total_chars:,}\n"
        f"**Exported:** {datetime.now(timezone.utc).isoformat()}\n\n"
        f"---\n\n"
    )
    corpus = header + "".join(parts)

    if output_path:
        Path(output_path).write_text(corpus, encoding="utf-8")
        return output_path
    return corpus


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="YouTube Channel Harvester — bulk transcript + comment downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s @MorePlatesMoreDates\n"
            "  %(prog)s @MorePlatesMoreDates @JeffNippard --max-videos 100\n"
            "  %(prog)s --export --channel @MorePlatesMoreDates --out corpus.md\n"
        ),
    )
    parser.add_argument(
        "channels", nargs="*",
        help="YouTube channel URLs, handles (@name), or IDs (UCxxxx)",
    )
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--no-comments", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--channel", default=None)
    parser.add_argument("--out", default="")
    parser.add_argument("--max-chars", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    config = HarvestConfig.from_env()
    config.delay = args.delay
    config.language = args.language
    config.fetch_comments = not args.no_comments

    # Try to wire in Miro cache
    cache_fn = None
    try:
        sys.path.insert(0, os.path.join(
            os.path.dirname(__file__), "..", "apps", "strands-agent",
        ))
        from cache import cache_put
        cache_fn = cache_put
    except ImportError:
        logger.info("Miro cache not available — transcripts will not be cached")

    harvester = Harvester(config=config, cache_fn=cache_fn)

    if not args.channels and not args.export:
        parser.print_help()
        sys.exit(1)

    if args.export:
        # TODO: Load from registry/cache for export
        print("Export from cache not yet implemented — use harvest first, then export from result")
        sys.exit(1)

    # Harvest each channel, accumulating all videos for export
    all_videos: list[VideoRecord] = []
    for channel in args.channels:
        print(f"\n{'=' * 60}")
        print(f"Harvesting: {channel}")
        print(f"{'=' * 60}\n")

        result = asyncio.run(harvester.harvest_channel(channel, max_videos=args.max_videos))
        print(result.summary())
        all_videos.extend(result.videos)

        if result.errors:
            print("Errors:")
            for err in result.errors:
                print(f"  - {err}")

    # Export combined corpus after all channels are harvested
    if args.out and all_videos:
        path = export_corpus(
            all_videos, args.out, args.max_chars,
            include_comments=config.fetch_comments,
        )
        print(f"Corpus exported to: {path}")


if __name__ == "__main__":
    main()

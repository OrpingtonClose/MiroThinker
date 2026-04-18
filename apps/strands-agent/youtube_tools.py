# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
YouTube intelligence tools for the Strands research agent.

Provides:
1. Video transcript download with multi-backend cascade:
   yt-dlp → TranscriptAPI → Bright Data (anti-block scraping)
2. YouTube search — find videos by topic query
3. Channel search — find videos within a specific channel
4. Bulk channel analysis (download all videos, transcribe, extract insights)
5. Comment extraction (YouTube Data API v3 or yt-dlp fallback)
6. Transcript search — keyword search across cached transcripts

All downloaded content is stored in the persistent cache (cache.py) for
reuse across sessions.

Resolves:
- https://github.com/OrpingtonClose/MiroThinker/issues/91  (YouTube download)
- https://github.com/OrpingtonClose/MiroThinker/issues/84  (YouTube comments)
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

from strands import tool

logger = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────


def _yt_dlp_available() -> bool:
    """Check if yt-dlp is installed."""
    try:
        subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True,
            timeout=10,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _whisper_available() -> bool:
    """Check if whisper (openai-whisper) is installed."""
    try:
        import whisper  # noqa: F401

        return True
    except ImportError:
        return False


def _extract_video_id(url_or_id: str) -> str:
    """Extract YouTube video ID from URL or return as-is if already an ID."""
    # Already an ID (11 chars, no slashes)
    if re.match(r"^[a-zA-Z0-9_-]{11}$", url_or_id):
        return url_or_id

    # Standard URLs
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
        r"(?:shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    return url_or_id  # Return as-is, let yt-dlp handle it


def _extract_channel_id(url_or_id: str) -> str:
    """Extract YouTube channel identifier from URL or return as-is."""
    # Already a channel ID or handle
    if url_or_id.startswith("UC") or url_or_id.startswith("@"):
        return url_or_id

    # Channel URL patterns
    patterns = [
        r"youtube\.com/channel/(UC[a-zA-Z0-9_-]+)",
        r"youtube\.com/(@[a-zA-Z0-9_.-]+)",
        r"youtube\.com/c/([a-zA-Z0-9_.-]+)",
        r"youtube\.com/user/([a-zA-Z0-9_.-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    return url_or_id


# ═══════════════════════════════════════════════════════════════════════
# BACKEND HELPERS — transcript acquisition cascade
# ═══════════════════════════════════════════════════════════════════════
#
# YouTube aggressively blocks cloud IPs. The cascade tries multiple
# approaches in order of speed and reliability:
#
#   1. yt-dlp          — fastest, works locally / non-cloud IPs
#   2. youtube-transcript-api (direct) — pure Python, same IP constraints
#   3. youtube-transcript-api (proxied) — routes through residential proxy
#      (Bright Data or Oxylabs) to bypass IP blocks
#   4. TranscriptAPI REST — third-party service that handles unblocking
#
# For cloud deployments, configure at least one of:
#   - BRIGHT_DATA_PROXY_URL (e.g. http://user:pass@brd.superproxy.io:22225)
#   - OXYLABS_PROXY_URL (e.g. http://user:pass@pr.oxylabs.io:7777)
#   - TRANSCRIPTAPI_KEY (direct API key from transcriptapi.com/dashboard)
#
# The TranscriptAPI MCP server (already wired in tools.py) also provides
# cloud-friendly transcript access as an MCP tool when the agent is running.
# ═══════════════════════════════════════════════════════════════════════


def _build_proxy_url() -> str | None:
    """Build a residential proxy URL from available credentials.

    Checks in order: explicit proxy URL → Bright Data → Oxylabs.
    Returns None if no proxy is configured.
    """
    # Explicit proxy URL (highest priority — user knows best)
    explicit = os.environ.get("YOUTUBE_PROXY_URL", "")
    if explicit:
        return explicit

    # Bright Data proxy
    bd_proxy = os.environ.get("BRIGHT_DATA_PROXY_URL", "")
    if bd_proxy:
        return bd_proxy

    # Oxylabs proxy
    oxy_user = os.environ.get("OXYLABS_USERNAME", "")
    oxy_pass = os.environ.get("OXYLABS_PASSWORD", "")
    if oxy_user and oxy_pass:
        from urllib.parse import quote

        return f"http://{quote(oxy_user, safe='')}:{quote(oxy_pass, safe='')}@pr.oxylabs.io:7777"

    return None


def _ytdlp_get_transcript(video_id: str, language: str) -> str | None:
    """Backend 1: yt-dlp subtitle download. Fast when not IP-blocked."""
    if not _yt_dlp_available():
        return None

    url = f"https://www.youtube.com/watch?v={video_id}"

    # Try with proxy if available, otherwise direct
    proxy_url = _build_proxy_url()
    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs",
        f"{language}.*",
        "--sub-format",
        "vtt",
        "--convert-subs",
        "srt",
    ]
    if proxy_url:
        cmd.extend(["--proxy", proxy_url])
    cmd.extend(["-o", "", url])

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd[-2] = str(Path(tmpdir) / "subs")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            logger.warning("yt-dlp subtitle download timed out for %s", video_id)
            return None

        # Check for bot detection / sign-in errors
        if result.returncode != 0:
            stderr = (result.stderr or "") + (result.stdout or "")
            if any(
                kw in stderr.lower()
                for kw in [
                    "sign in",
                    "bot",
                    "captcha",
                    "blocked",
                    "http error 429",
                    "too many requests",
                    "login_required",
                ]
            ):
                logger.info(
                    "yt-dlp blocked by YouTube for %s, trying fallbacks", video_id
                )
                return None

        srt_files = list(Path(tmpdir).glob("*.srt"))
        vtt_files = list(Path(tmpdir).glob("*.vtt"))
        sub_files = srt_files or vtt_files

        if sub_files:
            raw_text = sub_files[0].read_text(encoding="utf-8", errors="replace")
            return _clean_srt(raw_text)

    return None


def _yttapi_get_transcript(video_id: str, language: str) -> str | None:
    """Backend 2: youtube-transcript-api library (direct, no proxy)."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        logger.debug("youtube-transcript-api not installed, skipping")
        return None

    try:
        ytt = YouTubeTranscriptApi()
        transcript = ytt.fetch(video_id, languages=[language, "en"])
        text = " ".join(seg.text for seg in transcript.snippets)
        return text.strip() if text.strip() else None
    except Exception as exc:
        exc_name = type(exc).__name__
        if "Blocked" in exc_name or "blocked" in str(exc).lower():
            logger.info(
                "youtube-transcript-api blocked for %s (cloud IP), trying proxied",
                video_id,
            )
        else:
            logger.warning(
                "youtube-transcript-api failed for %s: %s: %s",
                video_id,
                exc_name,
                str(exc)[:200],
            )
        return None


def _yttapi_proxied_get_transcript(video_id: str, language: str) -> str | None:
    """Backend 3: youtube-transcript-api via residential proxy (Bright Data/Oxylabs)."""
    proxy_url = _build_proxy_url()
    if not proxy_url:
        logger.debug("No proxy configured, skipping proxied youtube-transcript-api")
        return None

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api.proxies import GenericProxyConfig
    except ImportError:
        logger.debug("youtube-transcript-api not installed, skipping")
        return None

    try:
        proxy_config = GenericProxyConfig(https_url=proxy_url)
        ytt = YouTubeTranscriptApi(proxy_config=proxy_config)
        transcript = ytt.fetch(video_id, languages=[language, "en"])
        text = " ".join(seg.text for seg in transcript.snippets)
        return text.strip() if text.strip() else None
    except Exception as exc:
        logger.warning(
            "Proxied youtube-transcript-api failed for %s: %s: %s",
            video_id,
            type(exc).__name__,
            str(exc)[:200],
        )
        return None


def _transcriptapi_get_transcript(video_id: str, language: str) -> str | None:
    """Backend 4: TranscriptAPI.com REST API — cloud-friendly third-party service."""
    import httpx

    api_key = os.environ.get("TRANSCRIPTAPI_KEY", "")
    if not api_key:
        logger.debug("TRANSCRIPTAPI_KEY not set, skipping TranscriptAPI backend")
        return None

    try:
        resp = httpx.get(
            "https://transcriptapi.com/api/v2/youtube/transcript",
            params={
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "lang": language,
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )

        if resp.status_code == 200:
            data = resp.json()
            # TranscriptAPI returns either a transcript array or plain text
            if isinstance(data, dict):
                transcript_parts = data.get("transcript", [])
                if isinstance(transcript_parts, list) and transcript_parts:
                    # Array of {text, start, duration} objects
                    return " ".join(
                        seg.get("text", "") for seg in transcript_parts
                    ).strip()
                if isinstance(transcript_parts, str) and transcript_parts.strip():
                    return transcript_parts.strip()
                text = data.get("text", "")
                if text:
                    return text.strip()
            elif isinstance(data, str) and data.strip():
                return data.strip()

        logger.info("TranscriptAPI returned %d for %s", resp.status_code, video_id)
        return None

    except Exception as exc:
        logger.warning("TranscriptAPI failed for %s: %s", video_id, exc)
        return None


def _get_transcript_cascade(video_id: str, language: str) -> tuple[str | None, str]:
    """Try all backends in order, return (transcript, backend_name) or (None, "").

    Cascade order:
      1. yt-dlp (fast, local-friendly, uses proxy if configured)
      2. youtube-transcript-api direct (fast, pure Python)
      3. youtube-transcript-api via proxy (residential IP bypass)
      4. TranscriptAPI REST (third-party cloud service)
    """
    backends = [
        ("yt-dlp", lambda: _ytdlp_get_transcript(video_id, language)),
        ("youtube-transcript-api", lambda: _yttapi_get_transcript(video_id, language)),
        (
            "youtube-transcript-api+proxy",
            lambda: _yttapi_proxied_get_transcript(video_id, language),
        ),
        ("TranscriptAPI", lambda: _transcriptapi_get_transcript(video_id, language)),
    ]

    tried = []
    for name, fetch_fn in backends:
        logger.info("Trying %s for video %s...", name, video_id)
        try:
            result = fetch_fn()
            if result and len(result.strip()) > 50:  # Minimum viable transcript
                logger.info(
                    "Got transcript from %s for %s (%d chars)",
                    name,
                    video_id,
                    len(result),
                )
                return result, name
            tried.append(f"{name}:empty")
        except Exception as exc:
            logger.warning("%s failed for %s: %s", name, video_id, exc)
            tried.append(f"{name}:error")
            continue

    logger.warning(
        "All backends exhausted for %s. Tried: %s", video_id, ", ".join(tried)
    )
    return None, ""


# ═══════════════════════════════════════════════════════════════════════
# VIDEO DOWNLOAD & TRANSCRIPTION (with cascade)
# ═══════════════════════════════════════════════════════════════════════


@tool
def youtube_download_transcript(
    video_url: str,
    language: str = "en",
) -> str:
    """Download a YouTube video's transcript/captions.

    Uses a multi-backend cascade for maximum reliability:
    1. yt-dlp (fastest, but blocked on some cloud IPs)
    2. TranscriptAPI.com (cloud-friendly, paid API)
    3. Bright Data (anti-block scraping, bypasses geo/IP restrictions)
    4. Whisper audio transcription (slowest, last resort)

    All transcripts are cached for instant retrieval on subsequent calls.

    This is the primary tool for getting text from YouTube videos. Use this
    for transcript hunting — finding specific mentions, quotes, or
    information in video content.

    Args:
        video_url: YouTube video URL or video ID.
        language: Preferred subtitle language code (default "en").

    Returns:
        The transcript text with video metadata, or error message if unavailable.
    """
    video_id = _extract_video_id(video_url)
    url = f"https://www.youtube.com/watch?v={video_id}"

    # Check cache first
    try:
        from cache import cache_get

        cached = cache_get(url=f"youtube-transcript://{video_id}/{language}")
        if cached and cached.get("content"):
            try:
                return f"[FROM CACHE]\n{cached['content'].decode('utf-8')}"
            except (UnicodeDecodeError, AttributeError):
                pass
    except ImportError:
        pass

    # Try cascade
    transcript, backend = _get_transcript_cascade(video_id, language)

    if transcript:
        title = _get_video_title(url)
        _cache_transcript(video_id, language, transcript, title)

        return (
            f"**{title}**\n"
            f"Video: {url}\n"
            f"Language: {language}\n"
            f"Source: {backend}\n"
            f"---\n\n"
            f"{transcript}"
        )

    # Last resort: whisper audio transcription
    if _whisper_available() and _yt_dlp_available():
        with tempfile.TemporaryDirectory() as tmpdir:
            return _whisper_transcribe(url, video_id, language, tmpdir)

    backends_tried = "yt-dlp, youtube-transcript-api, youtube-transcript-api+proxy, TranscriptAPI"
    return (
        f"No transcript found for {url} in language '{language}'. "
        f"Tried: {backends_tried}. "
        f"Whisper not available for audio fallback."
    )


# ═══════════════════════════════════════════════════════════════════════
# YOUTUBE SEARCH — find videos by topic
# ═══════════════════════════════════════════════════════════════════════


@tool
def youtube_search(
    query: str,
    max_results: int = 20,
) -> str:
    """Search YouTube for videos matching a query.

    Uses TranscriptAPI's search endpoint (cloud-friendly) with yt-dlp
    as fallback. Returns video IDs, titles, channels, and durations.

    Use this to DISCOVER relevant videos before downloading their
    transcripts. For research: search for topic keywords, then
    bulk-transcribe the most promising results.

    Args:
        query: Search query (e.g. "Milos Sarcev insulin protocol podcast").
        max_results: Maximum results to return (default 20, max 50).

    Returns:
        Formatted list of matching videos with metadata.
    """
    max_results = min(max_results, 50)

    # Try TranscriptAPI search first (cloud-friendly)
    result = _transcriptapi_search(query, max_results)
    if result:
        return result

    # Fallback: yt-dlp search
    result = _ytdlp_search(query, max_results)
    if result:
        return result

    return f"No YouTube search results for: {query}"


def _transcriptapi_search(query: str, max_results: int) -> str | None:
    """Search YouTube via TranscriptAPI's /youtube/search endpoint."""
    import httpx

    api_key = os.environ.get("TRANSCRIPTAPI_KEY", "")
    if not api_key:
        return None

    try:
        resp = httpx.get(
            "https://transcriptapi.com/api/v2/youtube/search",
            params={"q": query, "limit": max_results},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )

        if resp.status_code != 200:
            logger.info("TranscriptAPI search returned %d", resp.status_code)
            return None

        data = resp.json()
        videos = (
            data
            if isinstance(data, list)
            else data.get("results", data.get("videos", []))
        )

        if not videos:
            return None

        lines = [
            f"YouTube search results for '{query}' ({len(videos)} found, via TranscriptAPI):\n"
        ]
        for i, v in enumerate(videos[:max_results], 1):
            vid = v.get("videoId", v.get("video_id", v.get("id", "???")))
            title = v.get("title", "Unknown")[:80]
            channel = v.get("channelTitle", v.get("channel", v.get("author", "")))
            duration = v.get("lengthText", v.get("duration", v.get("length", "")))
            views = v.get("viewCountText", v.get("views", v.get("view_count", "")))
            pub = (
                v.get("publishedTimeText", v.get("published", v.get("upload_date", "")))
                or ""
            )[:20]
            meta_parts = []
            if channel:
                meta_parts.append(channel)
            if duration:
                meta_parts.append(f"{duration}")
            if views:
                meta_parts.append(f"{views} views")
            if pub:
                meta_parts.append(pub)
            meta = " | ".join(meta_parts)
            lines.append(f"  {i}. [{vid}] {title}")
            if meta:
                lines.append(f"     {meta}")

        return "\n".join(lines)

    except Exception as exc:
        logger.warning("TranscriptAPI search failed: %s", exc)
        return None


def _ytdlp_search(query: str, max_results: int) -> str | None:
    """Search YouTube via yt-dlp's ytsearch."""
    if not _yt_dlp_available():
        return None

    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--flat-playlist",
                "--dump-json",
                f"ytsearch{max_results}:{query}",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            return None

        videos = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                videos.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        if not videos:
            return None

        lines = [
            f"YouTube search results for '{query}' ({len(videos)} found, via yt-dlp):\n"
        ]
        for i, v in enumerate(videos, 1):
            vid = v.get("id", "???")
            title = v.get("title", "Unknown")[:80]
            channel = v.get("channel", v.get("uploader", ""))
            duration = v.get("duration")
            dur_str = f"{duration // 60}m{duration % 60}s" if duration else "?"
            views = v.get("view_count")
            view_str = f"{views:,}" if views else "?"
            lines.append(f"  {i}. [{vid}] {title} ({dur_str}, {view_str} views)")
            if channel:
                lines.append(f"     Channel: {channel}")

        return "\n".join(lines)

    except subprocess.TimeoutExpired:
        return None


# ═══════════════════════════════════════════════════════════════════════
# CHANNEL SEARCH — find videos within a specific channel
# ═══════════════════════════════════════════════════════════════════════


@tool
def youtube_channel_search(
    channel: str,
    query: str,
    max_results: int = 20,
) -> str:
    """Search within a specific YouTube channel for videos matching a query.

    Useful for finding specific topics in prolific creators' catalogs.
    E.g., search @MorePlatesMoreDates for "trenbolone insulin" to find
    relevant episodes without browsing hundreds of uploads.

    Args:
        channel: Channel handle (@handle), channel URL, or channel ID (UCxxxxx).
        query: Search query within the channel.
        max_results: Maximum results to return (default 20).

    Returns:
        Formatted list of matching videos from the channel.
    """
    import httpx

    max_results = min(max_results, 50)
    channel_id = _extract_channel_id(channel)

    # Try TranscriptAPI channel search
    api_key = os.environ.get("TRANSCRIPTAPI_KEY", "")
    if api_key:
        try:
            resp = httpx.get(
                "https://transcriptapi.com/api/v2/youtube/channel/search",
                params={
                    "channel": channel_id,
                    "q": query,
                    "limit": max_results,
                },
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30,
            )

            if resp.status_code == 200:
                data = resp.json()
                videos = (
                    data
                    if isinstance(data, list)
                    else data.get("results", data.get("videos", []))
                )

                if videos:
                    lines = [
                        f"Channel search '{query}' in {channel_id} ({len(videos)} found):\n"
                    ]
                    for i, v in enumerate(videos[:max_results], 1):
                        vid = v.get("videoId", v.get("video_id", v.get("id", "???")))
                        title = v.get("title", "Unknown")[:80]
                        duration = v.get(
                            "lengthText", v.get("duration", v.get("length", ""))
                        )
                        pub = (
                            v.get(
                                "publishedTimeText",
                                v.get("published", v.get("upload_date", "")),
                            )
                            or ""
                        )[:20]
                        meta_parts = []
                        if duration:
                            meta_parts.append(str(duration))
                        if pub:
                            meta_parts.append(pub)
                        meta = " | ".join(meta_parts)
                        lines.append(f"  {i}. [{vid}] {title}")
                        if meta:
                            lines.append(f"     {meta}")
                    return "\n".join(lines)

        except Exception as exc:
            logger.warning("TranscriptAPI channel search failed: %s", exc)

    # Fallback: yt-dlp channel listing + grep
    if _yt_dlp_available():
        try:
            if channel_id.startswith("UC"):
                ch_url = f"https://www.youtube.com/channel/{channel_id}/videos"
            elif channel_id.startswith("@"):
                ch_url = f"https://www.youtube.com/{channel_id}/videos"
            else:
                ch_url = f"https://www.youtube.com/{channel_id}/videos"

            result = subprocess.run(
                [
                    "yt-dlp",
                    "--flat-playlist",
                    "--dump-json",
                    "--playlist-end",
                    "200",
                    ch_url,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0:
                query_lower = query.lower()
                query_terms = query_lower.split()
                matches = []

                for line in result.stdout.strip().split("\n"):
                    if not line.strip():
                        continue
                    try:
                        info = json.loads(line)
                        title = (info.get("title", "") or "").lower()
                        if any(term in title for term in query_terms):
                            matches.append(info)
                    except json.JSONDecodeError:
                        continue

                if matches:
                    lines = [
                        f"Channel search '{query}' in {channel_id} ({len(matches)} matches via yt-dlp):\n"
                    ]
                    for i, v in enumerate(matches[:max_results], 1):
                        vid = v.get("id", "???")
                        title = v.get("title", "Unknown")[:80]
                        duration = v.get("duration")
                        dur_str = f"{duration // 60}m" if duration else "?"
                        lines.append(f"  {i}. [{vid}] {title} ({dur_str})")
                    return "\n".join(lines)

        except subprocess.TimeoutExpired:
            pass

    return f"No results for '{query}' in channel {channel_id}"


# ═══════════════════════════════════════════════════════════════════════
# TRANSCRIPT SEARCH — keyword search across cached transcripts
# ═══════════════════════════════════════════════════════════════════════


@tool
def youtube_search_transcripts(
    query: str,
    max_results: int = 10,
) -> str:
    """Search across all cached YouTube transcripts for keyword mentions.

    This searches the LOCAL transcript cache — transcripts that have already
    been downloaded by youtube_download_transcript or youtube_bulk_transcribe.
    Use this to find which videos mention specific topics, compounds, names,
    or phrases across your research corpus.

    For each match, returns the video title, surrounding context (snippet),
    and approximate timestamp position.

    Args:
        query: Search terms (case-insensitive substring match).
        max_results: Maximum number of matching videos to return (default 10).

    Returns:
        Formatted search results with context snippets from matching transcripts.
    """
    try:
        from cache import CACHE_DIR, get_db
    except ImportError:
        return (
            "[TOOL_ERROR] Cache module not available. Download some transcripts first."
        )

    conn = get_db()
    query_lower = query.lower()

    # Search the cache for youtube transcript entries
    rows = conn.execute(
        """
        SELECT url, title, blob_path, blob_size
        FROM cache_entries
        WHERE url LIKE 'youtube-transcript://%'
        AND blob_size > 0
        ORDER BY created_at DESC
        """,
    ).fetchall()

    if not rows:
        return (
            "No cached transcripts found. Use youtube_download_transcript "
            "or youtube_bulk_transcribe to download transcripts first."
        )

    matches = []
    for row in rows:
        url = row[0]
        title = row[1] or "Unknown"
        blob_path = row[2]
        blob_size = row[3]

        if not blob_path or not (CACHE_DIR / blob_path).exists():
            continue

        try:
            text = (CACHE_DIR / blob_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        text_lower = text.lower()
        if query_lower not in text_lower:
            continue

        # Extract video ID from cache URL
        vid_match = re.match(r"youtube-transcript://([^/]+)/", url)
        video_id = vid_match.group(1) if vid_match else "?"

        # Find context snippets (up to 3 per video)
        snippets = []
        search_pos = 0
        while len(snippets) < 3:
            idx = text_lower.find(query_lower, search_pos)
            if idx == -1:
                break
            # Extract surrounding context (200 chars each side)
            start = max(0, idx - 200)
            end = min(len(text), idx + len(query) + 200)
            snippet = text[start:end].replace("\n", " ").strip()
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."
            snippets.append(snippet)
            search_pos = idx + len(query) + 100  # Skip ahead

        # Count total occurrences
        count = text_lower.count(query_lower)

        matches.append(
            {
                "video_id": video_id,
                "title": title,
                "occurrences": count,
                "snippets": snippets,
                "blob_size": blob_size,
            }
        )

        if len(matches) >= max_results:
            break

    if not matches:
        return (
            f"No cached transcripts contain '{query}'. "
            f"Searched {len(rows)} cached transcripts."
        )

    lines = [
        f"Transcript search for '{query}' — {len(matches)} matching videos "
        f"(searched {len(rows)} cached transcripts):\n"
    ]

    for i, m in enumerate(matches, 1):
        vid = m["video_id"]
        url = f"https://www.youtube.com/watch?v={vid}"
        lines.append(f"\n{i}. **{m['title']}** ({m['occurrences']} mentions)")
        lines.append(f"   Video: {url}")
        for j, snippet in enumerate(m["snippets"], 1):
            lines.append(f'   [{j}] "{snippet}"')

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# VIDEO METADATA
# ═══════════════════════════════════════════════════════════════════════


@tool
def youtube_video_info(video_url: str) -> str:
    """Get metadata about a YouTube video without downloading it.

    Returns title, description, duration, view count, upload date,
    channel info, tags, and available subtitle languages.

    Args:
        video_url: YouTube video URL or video ID.

    Returns:
        Formatted video metadata.
    """
    if not _yt_dlp_available():
        return "[TOOL_ERROR] yt-dlp not installed."

    video_id = _extract_video_id(video_url)
    url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--dump-json",
                "--no-playlist",
                "--no-download",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            return f"[TOOL_ERROR] yt-dlp failed: {result.stderr[:500]}"

        info = json.loads(result.stdout)

        # Extract subtitle languages
        subs = info.get("subtitles", {})
        auto_subs = info.get("automatic_captions", {})
        manual_langs = list(subs.keys())[:20]
        auto_langs = list(auto_subs.keys())[:20]

        duration_m = info.get("duration", 0) // 60
        duration_s = info.get("duration", 0) % 60

        return (
            f"**{info.get('title', 'Unknown')}**\n"
            f"Channel: {info.get('channel', info.get('uploader', 'Unknown'))}\n"
            f"URL: {url}\n"
            f"Duration: {duration_m}m {duration_s}s\n"
            f"Views: {info.get('view_count', 'N/A'):,}\n"
            f"Likes: {info.get('like_count', 'N/A')}\n"
            f"Upload date: {info.get('upload_date', 'N/A')}\n"
            f"Description: {(info.get('description', '') or '')[:1000]}\n"
            f"Tags: {', '.join(info.get('tags', [])[:15])}\n"
            f"Manual subtitles: {', '.join(manual_langs) or 'None'}\n"
            f"Auto-captions: {', '.join(auto_langs) or 'None'}\n"
        )

    except subprocess.TimeoutExpired:
        return "[TOOL_ERROR] yt-dlp metadata fetch timed out."
    except json.JSONDecodeError:
        return "[TOOL_ERROR] Failed to parse yt-dlp JSON output."


# ═══════════════════════════════════════════════════════════════════════
# CHANNEL LISTING
# ═══════════════════════════════════════════════════════════════════════


@tool
def youtube_channel_list(
    channel_url: str,
    max_videos: int = 50,
) -> str:
    """List all videos from a YouTube channel.

    Returns video IDs, titles, durations, and view counts for the channel.
    Use this to survey a channel's content before downloading specific transcripts.

    Args:
        channel_url: YouTube channel URL, channel ID (UCxxxxx), or handle (@handle).
        max_videos: Maximum number of videos to list (default 50, max 500).

    Returns:
        Formatted list of videos with metadata.
    """
    max_videos = min(max_videos, 500)
    channel_id = _extract_channel_id(channel_url)

    # Try TranscriptAPI first (cloud-friendly)
    result = _transcriptapi_channel_list(channel_id, max_videos)
    if result:
        return result

    # Fallback: yt-dlp
    if not _yt_dlp_available():
        return "[TOOL_ERROR] Neither TranscriptAPI nor yt-dlp available."

    # Build the channel URL
    if channel_id.startswith("UC"):
        url = f"https://www.youtube.com/channel/{channel_id}/videos"
    elif channel_id.startswith("@"):
        url = f"https://www.youtube.com/{channel_id}/videos"
    else:
        url = f"https://www.youtube.com/{channel_id}/videos"

    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--flat-playlist",
                "--dump-json",
                "--playlist-end",
                str(max_videos),
                url,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            return f"[TOOL_ERROR] yt-dlp channel listing failed: {result.stderr[:500]}"

        videos = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                info = json.loads(line)
                videos.append(info)
            except json.JSONDecodeError:
                continue

        if not videos:
            return f"No videos found for channel: {channel_url}"

        formatted = [f"Channel videos ({len(videos)} found, via yt-dlp):\n"]
        for i, v in enumerate(videos, 1):
            vid = v.get("id", "???")
            title = v.get("title", "Unknown")[:80]
            duration = v.get("duration")
            dur_str = f"{duration // 60}m" if duration else "?"
            views = v.get("view_count")
            view_str = f"{views:,}" if views else "?"
            formatted.append(f"  {i}. [{vid}] {title} ({dur_str}, {view_str} views)")

        return "\n".join(formatted)

    except subprocess.TimeoutExpired:
        return "[TOOL_ERROR] Channel listing timed out (120s)."


def _transcriptapi_channel_list(channel_id: str, max_videos: int) -> str | None:
    """List channel videos via TranscriptAPI."""
    import httpx

    api_key = os.environ.get("TRANSCRIPTAPI_KEY", "")
    if not api_key:
        return None

    try:
        # Try paginated endpoint first
        resp = httpx.get(
            "https://transcriptapi.com/api/v2/youtube/channel/videos",
            params={"channel": channel_id, "limit": min(max_videos, 50)},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )

        if resp.status_code != 200:
            # Try the free latest endpoint (up to 15 videos via RSS)
            resp = httpx.get(
                "https://transcriptapi.com/api/v2/youtube/channel/latest",
                params={"channel": channel_id},
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30,
            )
            if resp.status_code != 200:
                return None

        data = resp.json()
        videos = (
            data
            if isinstance(data, list)
            else data.get("videos", data.get("results", []))
        )

        if not videos:
            return None

        lines = [
            f"Channel videos for {channel_id} ({len(videos)} found, via TranscriptAPI):\n"
        ]
        for i, v in enumerate(videos[:max_videos], 1):
            vid = v.get("videoId", v.get("video_id", v.get("id", "???")))
            title = v.get("title", "Unknown")[:80]
            duration = v.get("lengthText", v.get("duration", v.get("length", "")))
            pub = (
                v.get("publishedTimeText", v.get("published", v.get("upload_date", "")))
                or ""
            )[:20]
            meta_parts = []
            if duration:
                meta_parts.append(str(duration))
            if pub:
                meta_parts.append(pub)
            meta = " | ".join(meta_parts)
            lines.append(f"  {i}. [{vid}] {title}")
            if meta:
                lines.append(f"     {meta}")

        return "\n".join(lines)

    except Exception as exc:
        logger.warning("TranscriptAPI channel list failed: %s", exc)
        return None


# ═══════════════════════════════════════════════════════════════════════
# BULK TRANSCRIPTION (with cascade)
# ═══════════════════════════════════════════════════════════════════════


@tool
def youtube_bulk_transcribe(
    video_ids: str,
    language: str = "en",
) -> str:
    """Download transcripts for multiple YouTube videos at once.

    Uses the same multi-backend cascade as youtube_download_transcript
    (yt-dlp → TranscriptAPI → Bright Data) for each video.

    Use this after youtube_search or youtube_channel_list to bulk-transcribe
    selected videos. Results are cached for future sessions.

    Args:
        video_ids: JSON array of video IDs or URLs (max 20).
            Example: '["dQw4w9WgXcQ", "jNQXAC9IVRw", "kJQP7kiw5Fk"]'
        language: Preferred subtitle language code (default "en").

    Returns:
        Summary of transcription results for each video.
    """
    try:
        ids = json.loads(video_ids)
    except (json.JSONDecodeError, TypeError):
        return "[TOOL_ERROR] Invalid JSON array of video IDs."

    if not isinstance(ids, list):
        return "[TOOL_ERROR] video_ids must be a JSON array."

    ids = ids[:20]  # Cap at 20
    results = []
    success_count = 0
    cache_count = 0

    for vid_url in ids:
        vid_id = _extract_video_id(str(vid_url))

        # Check cache first
        try:
            from cache import cache_get

            cached = cache_get(url=f"youtube-transcript://{vid_id}/{language}")
            if cached and cached.get("content"):
                results.append(
                    f"  [{vid_id}] CACHED — {cached.get('title', 'Unknown')} "
                    f"({cached['blob_size']:,} bytes)"
                )
                cache_count += 1
                continue
        except ImportError:
            pass

        # Try cascade
        transcript, backend = _get_transcript_cascade(vid_id, language)

        if transcript:
            url = f"https://www.youtube.com/watch?v={vid_id}"
            title = _get_video_title(url)
            _cache_transcript(vid_id, language, transcript, title)
            results.append(
                f"  [{vid_id}] OK via {backend} — {title} ({len(transcript):,} chars)"
            )
            success_count += 1
        else:
            title = _get_video_title(f"https://www.youtube.com/watch?v={vid_id}")
            results.append(f"  [{vid_id}] FAILED — {title} (all backends exhausted)")

    total = len(ids)
    summary = (
        f"Bulk transcription: {total} videos — "
        f"{success_count} new, {cache_count} cached, "
        f"{total - success_count - cache_count} failed"
    )
    return f"{summary}\n" + "\n".join(results)


# ═══════════════════════════════════════════════════════════════════════
# YOUTUBE COMMENT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════


@tool
def youtube_get_comments(
    video_url: str,
    max_comments: int = 100,
    sort_by: str = "relevance",
) -> str:
    """Extract comments from a YouTube video.

    Comments are a premium intelligence source — real people sharing opinions,
    recommendations, corrections, and anecdotes.

    Uses YouTube Data API v3 (if YOUTUBE_API_KEY set) for faster extraction,
    falls back to yt-dlp comment extraction (slower but no key needed).

    Args:
        video_url: YouTube video URL or video ID.
        max_comments: Maximum number of comments to extract (default 100, max 500).
        sort_by: "relevance" (top comments first) or "time" (newest first).

    Returns:
        Formatted comments with author, text, likes, and reply count.
    """
    video_id = _extract_video_id(video_url)
    max_comments = min(max_comments, 500)

    # Try YouTube Data API first (faster, more reliable)
    api_key = os.environ.get("YOUTUBE_API_KEY", "")
    if api_key:
        return _get_comments_api(video_id, max_comments, sort_by, api_key)

    # Fall back to yt-dlp comment extraction
    if not _yt_dlp_available():
        return (
            "[TOOL_ERROR] Neither YOUTUBE_API_KEY nor yt-dlp available. "
            "Set YOUTUBE_API_KEY in .env or install yt-dlp."
        )

    return _get_comments_ytdlp(video_id, max_comments, sort_by)


# ═══════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════


def _clean_srt(srt_text: str) -> str:
    """Clean SRT subtitle text — remove timestamps, numbers, HTML tags, dedup."""
    lines = srt_text.strip().split("\n")
    cleaned = []
    seen = set()

    for line in lines:
        line = line.strip()
        # Skip sequence numbers
        if re.match(r"^\d+$", line):
            continue
        # Skip timestamps
        if re.match(r"\d{2}:\d{2}:\d{2}", line):
            continue
        # Skip empty/WEBVTT header lines
        if (
            not line
            or line.startswith("WEBVTT")
            or line.startswith("Kind:")
            or line.startswith("Language:")
        ):
            continue
        # Remove HTML tags
        line = re.sub(r"<[^>]+>", "", line)
        # Remove position/alignment tags
        line = re.sub(r"\{[^}]+\}", "", line)
        line = line.strip()
        if line and line not in seen:
            seen.add(line)
            cleaned.append(line)

    return " ".join(cleaned)


def _get_video_title(url: str) -> str:
    """Get video title via yt-dlp."""
    try:
        cmd = ["yt-dlp", "--get-title", "--no-playlist"]
        proxy_url = _build_proxy_url()
        if proxy_url:
            cmd.extend(["--proxy", proxy_url])
        cmd.append(url)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip() or "Unknown Title"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "Unknown Title"


def _whisper_transcribe(url: str, video_id: str, language: str, tmpdir: str) -> str:
    """Download audio and transcribe with whisper."""
    audio_path = Path(tmpdir) / "audio"
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "5",  # Lower quality = faster download, sufficient for speech
        "-o",
        str(audio_path) + ".%(ext)s",
        "--no-playlist",
    ]
    proxy_url = _build_proxy_url()
    if proxy_url:
        cmd.extend(["--proxy", proxy_url])
    cmd.append(url)

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return "[TOOL_ERROR] Audio download timed out (5 min)."

    audio_files = list(Path(tmpdir).glob("audio.*"))
    if not audio_files:
        return "[TOOL_ERROR] Failed to download audio."

    try:
        import whisper

        model = whisper.load_model("base")
        result = model.transcribe(
            str(audio_files[0]),
            language=language if language != "auto" else None,
        )
        transcript = result.get("text", "")

        title = _get_video_title(url)
        _cache_transcript(video_id, language, transcript, title)

        return (
            f"**{title}** (whisper transcription)\n"
            f"Video: {url}\n"
            f"Language: {result.get('language', language)}\n"
            f"---\n\n"
            f"{transcript}"
        )
    except Exception as exc:
        return f"[TOOL_ERROR] Whisper transcription failed: {exc}"


def _cache_transcript(
    video_id: str, language: str, transcript: str, title: str
) -> None:
    """Store transcript in persistent cache."""
    try:
        from cache import cache_put

        cache_put(
            url=f"youtube-transcript://{video_id}/{language}",
            content=transcript,
            content_type="text/plain",
            source_type="video_transcript",
            title=title,
            summary=transcript[:500],
            tags=["youtube", "transcript", language],
        )
    except ImportError:
        logger.debug("Cache module not available — transcript not cached")


def _get_comments_api(
    video_id: str,
    max_comments: int,
    sort_by: str,
    api_key: str,
) -> str:
    """Extract comments via YouTube Data API v3."""
    import httpx

    order = "relevance" if sort_by == "relevance" else "time"
    comments = []
    next_page = None

    while len(comments) < max_comments:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(100, max_comments - len(comments)),
            "order": order,
            "textFormat": "plainText",
            "key": api_key,
        }
        if next_page:
            params["pageToken"] = next_page

        try:
            resp = httpx.get(
                "https://www.googleapis.com/youtube/v3/commentThreads",
                params=params,
                timeout=30,
            )

            if resp.status_code == 403:
                return (
                    "[TOOL_ERROR] YouTube API returned 403 — comments may be "
                    "disabled on this video, or the API key lacks permissions."
                )

            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            if comments:
                break  # Return what we have
            return f"[TOOL_ERROR] YouTube API failed: {exc}"

        for item in data.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append(
                {
                    "author": snippet.get("authorDisplayName", "Anonymous"),
                    "text": snippet.get("textDisplay", ""),
                    "likes": snippet.get("likeCount", 0),
                    "replies": item["snippet"].get("totalReplyCount", 0),
                    "published": snippet.get("publishedAt", "")[:10],
                }
            )

        next_page = data.get("nextPageToken")
        if not next_page:
            break

    if not comments:
        return f"No comments found for video {video_id}."

    return _format_comments(video_id, comments)


def _get_comments_ytdlp(video_id: str, max_comments: int, sort_by: str) -> str:
    """Extract comments via yt-dlp (slower but no API key needed)."""
    url = f"https://www.youtube.com/watch?v={video_id}"

    sort_arg = "top" if sort_by == "relevance" else "new"

    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--dump-json",
                "--no-download",
                "--write-comments",
                "--extractor-args",
                f"youtube:comment_sort={sort_arg};max_comments={max_comments}",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=180,  # Comments can be slow
        )

        if result.returncode != 0:
            return (
                f"[TOOL_ERROR] yt-dlp comment extraction failed: {result.stderr[:500]}"
            )

        info = json.loads(result.stdout)
        raw_comments = info.get("comments", [])

        if not raw_comments:
            return f"No comments found for video {video_id}."

        comments = []
        for c in raw_comments[:max_comments]:
            comments.append(
                {
                    "author": c.get("author", "Anonymous"),
                    "text": c.get("text", ""),
                    "likes": c.get("like_count", 0),
                    "replies": 0,  # yt-dlp doesn't report reply count easily
                    "published": c.get("timestamp", 0),
                }
            )

        return _format_comments(video_id, comments)

    except subprocess.TimeoutExpired:
        return "[TOOL_ERROR] Comment extraction timed out (3 min). Try fewer comments."
    except json.JSONDecodeError:
        return "[TOOL_ERROR] Failed to parse yt-dlp comment output."


def _format_comments(video_id: str, comments: list[dict]) -> str:
    """Format extracted comments for display."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    lines = [f"**Comments from {url}** ({len(comments)} extracted):\n"]

    for i, c in enumerate(comments, 1):
        likes = f" [{c['likes']} likes]" if c.get("likes") else ""
        replies = f" [{c['replies']} replies]" if c.get("replies") else ""
        text = c["text"].replace("\n", " ").strip()
        if len(text) > 500:
            text = text[:500] + "..."
        lines.append(f"  {i}. **{c['author']}**{likes}{replies}\n     {text}\n")

    return "\n".join(lines)


# ── Tool registry ─────────────────────────────────────────────────────

YOUTUBE_TOOLS = [
    youtube_download_transcript,
    youtube_search,
    youtube_channel_search,
    youtube_search_transcripts,
    youtube_video_info,
    youtube_channel_list,
    youtube_bulk_transcribe,
    youtube_get_comments,
]

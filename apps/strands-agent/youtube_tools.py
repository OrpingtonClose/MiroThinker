# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
YouTube intelligence tools for the Strands research agent.

Provides:
1. Video download & transcription (yt-dlp + whisper/captions)
2. Bulk channel analysis (download all videos, transcribe, extract insights)
3. Comment extraction (YouTube Data API v3)

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

import httpx
from strands import tool
from async_http import async_get

logger = logging.getLogger(__name__)

# ── TranscriptAPI REST helpers ────────────────────────────────────────

_TRANSCRIPTAPI_BASE = "https://transcriptapi.com/api/v2/youtube"
_TRANSCRIPTAPI_TIMEOUT = 30.0


def _transcriptapi_headers() -> dict[str, str]:
    """Build Authorization header for TranscriptAPI."""
    key = os.environ.get("TRANSCRIPTAPI_KEY", "")
    return {"Authorization": f"Bearer {key}"}


def _transcriptapi_get(path: str, params: dict[str, str | int]) -> dict:
    """GET request to TranscriptAPI REST endpoint."""
    url = f"{_TRANSCRIPTAPI_BASE}/{path}"
    resp = httpx.get(
        url,
        params=params,
        headers=_transcriptapi_headers(),
        timeout=_TRANSCRIPTAPI_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


# ── TranscriptAPI search tools (native fallback for MCP) ─────────────


@tool
def search_youtube(
    query: str,
    result_type: str = "video",
    limit: int = 10,
) -> str:
    """Search YouTube for videos, channels, or playlists via TranscriptAPI.

    This is the PRIMARY tool for YouTube research — use it FIRST before
    any web search when looking for YouTube channels or content on a topic.
    Returns actual YouTube search results with video IDs, titles, channels,
    view counts, and publish dates.

    Args:
        query: Search query (e.g. "insulin protocol bodybuilding").
        result_type: Type of results — "video", "channel", or "playlist".
        limit: Maximum number of results (1-50, default 10).

    Returns:
        JSON array of search results with video/channel metadata.
    """
    key = os.environ.get("TRANSCRIPTAPI_KEY", "")
    if not key:
        return json.dumps({"error": "TRANSCRIPTAPI_KEY not configured"})
    try:
        data = _transcriptapi_get(
            "search",
            {"q": query, "type": result_type, "limit": min(limit, 50)},
        )
        results = data.get("results", data)
        return json.dumps(results, indent=2, ensure_ascii=False)
    except httpx.HTTPStatusError as exc:
        return json.dumps({"error": f"TranscriptAPI HTTP {exc.response.status_code}", "detail": str(exc)})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool
def search_channel_videos(
    channel: str,
    query: str,
    limit: int = 10,
) -> str:
    """Search INSIDE a specific YouTube channel for videos matching a query.

    Use this to find which videos within a known channel cover a specific
    topic. Searches across video titles, descriptions, and transcript content.
    Returns matching videos with metadata.

    Args:
        channel: Channel handle (e.g. "@MorePlatesMoreDates") or channel ID.
        query: Search query to find within the channel's videos.
        limit: Maximum number of results (1-50, default 10).

    Returns:
        JSON array of matching videos from the specified channel.
    """
    key = os.environ.get("TRANSCRIPTAPI_KEY", "")
    if not key:
        return json.dumps({"error": "TRANSCRIPTAPI_KEY not configured"})
    try:
        data = _transcriptapi_get(
            "channel/search",
            {"channel": channel, "q": query, "limit": min(limit, 50)},
        )
        results = data.get("results", data)
        return json.dumps(results, indent=2, ensure_ascii=False)
    except httpx.HTTPStatusError as exc:
        return json.dumps({"error": f"TranscriptAPI HTTP {exc.response.status_code}", "detail": str(exc)})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool
def get_channel_latest_videos(
    channel: str,
    limit: int = 30,
) -> str:
    """Get the latest videos from a YouTube channel via TranscriptAPI.

    Use this to browse a channel's recent uploads — useful for assessing
    how active a channel is and what topics they've covered recently.

    Args:
        channel: Channel handle (e.g. "@MorePlatesMoreDates") or channel ID.
        limit: Maximum number of videos to return (default 30).

    Returns:
        JSON array of recent videos with metadata.
    """
    key = os.environ.get("TRANSCRIPTAPI_KEY", "")
    if not key:
        return json.dumps({"error": "TRANSCRIPTAPI_KEY not configured"})
    try:
        data = _transcriptapi_get(
            "channel/latest",
            {"channel": channel, "limit": min(limit, 50)},
        )
        results = data.get("results", data)
        return json.dumps(results, indent=2, ensure_ascii=False)
    except httpx.HTTPStatusError as exc:
        return json.dumps({"error": f"TranscriptAPI HTTP {exc.response.status_code}", "detail": str(exc)})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


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
# VIDEO DOWNLOAD & TRANSCRIPTION
# ═══════════════════════════════════════════════════════════════════════


@tool
def youtube_download_transcript(
    video_url: str,
    language: str = "en",
) -> str:
    """Download a YouTube video's transcript/captions.

    Tries in order:
    1. Manual (human-written) subtitles in the requested language
    2. Auto-generated subtitles
    3. Falls back to audio download + whisper transcription

    This is the fastest way to get text from a YouTube video. Use this
    for transcript hunting — finding specific mentions, quotes, or
    information in video content.

    Args:
        video_url: YouTube video URL or video ID.
        language: Preferred subtitle language code (default "en").
            Use "fil" for Filipino, "th" for Thai, "id" for Indonesian, etc.

    Returns:
        The transcript text, or error message if unavailable.
    """
    if not _yt_dlp_available():
        return (
            "[TOOL_ERROR] yt-dlp not installed. "
            "Install with: pip install yt-dlp"
        )

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

    with tempfile.TemporaryDirectory() as tmpdir:
        # Try to get subtitles first (much faster than audio transcription)
        sub_path = Path(tmpdir) / "subs"
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            "--sub-langs", f"{language}.*",
            "--sub-format", "vtt",
            "--convert-subs", "srt",
            "-o", str(sub_path),
            url,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return "[TOOL_ERROR] yt-dlp subtitle download timed out (120s)."

        # Look for downloaded subtitle files
        srt_files = list(Path(tmpdir).glob("*.srt"))
        vtt_files = list(Path(tmpdir).glob("*.vtt"))
        sub_files = srt_files or vtt_files

        if sub_files:
            raw_text = sub_files[0].read_text(encoding="utf-8", errors="replace")
            # Clean SRT formatting
            transcript = _clean_srt(raw_text)

            # Get video title
            title = _get_video_title(url)

            # Store in cache
            _cache_transcript(video_id, language, transcript, title)

            return (
                f"**{title}**\n"
                f"Video: {url}\n"
                f"Language: {language}\n"
                f"---\n\n"
                f"{transcript}"
            )

        # No subtitles available — try audio download + whisper
        if _whisper_available():
            return _whisper_transcribe(url, video_id, language, tmpdir)

        return (
            f"No subtitles found for {url} in language '{language}', "
            f"and whisper is not installed for audio transcription. "
            f"Install with: pip install openai-whisper"
        )


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
        if not line or line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
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
        result = subprocess.run(
            ["yt-dlp", "--get-title", "--no-playlist", url],
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
        "--audio-format", "mp3",
        "--audio-quality", "5",  # Lower quality = faster download, sufficient for speech
        "-o", str(audio_path) + ".%(ext)s",
        "--no-playlist",
        url,
    ]

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


def _cache_transcript(video_id: str, language: str, transcript: str, title: str) -> None:
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
    if not _yt_dlp_available():
        return "[TOOL_ERROR] yt-dlp not installed."

    channel_id = _extract_channel_id(channel_url)
    max_videos = min(max_videos, 500)

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
                "--playlist-end", str(max_videos),
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

        formatted = [f"Channel videos ({len(videos)} found):\n"]
        for i, v in enumerate(videos, 1):
            vid = v.get("id", "???")
            title = v.get("title", "Unknown")[:80]
            duration = v.get("duration")
            dur_str = f"{duration // 60}m" if duration else "?"
            views = v.get("view_count")
            view_str = f"{views:,}" if views else "?"
            formatted.append(
                f"  {i}. [{vid}] {title} ({dur_str}, {view_str} views)"
            )

        return "\n".join(formatted)

    except subprocess.TimeoutExpired:
        return "[TOOL_ERROR] Channel listing timed out (120s)."


@tool
def youtube_bulk_transcribe(
    video_ids: str,
    language: str = "en",
) -> str:
    """Download transcripts for multiple YouTube videos at once.

    Use this after youtube_channel_list to bulk-transcribe selected videos.
    Results are cached for future sessions.

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
                continue
        except ImportError:
            pass

        # Download transcript
        url = f"https://www.youtube.com/watch?v={vid_id}"
        title = _get_video_title(url)

        with tempfile.TemporaryDirectory() as tmpdir:
            sub_path = Path(tmpdir) / "subs"
            try:
                subprocess.run(
                    [
                        "yt-dlp",
                        "--skip-download",
                        "--write-subs",
                        "--write-auto-subs",
                        "--sub-langs", f"{language}.*",
                        "--sub-format", "vtt",
                        "--convert-subs", "srt",
                        "-o", str(sub_path),
                        url,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            except subprocess.TimeoutExpired:
                results.append(f"  [{vid_id}] TIMEOUT — {title}")
                continue

            srt_files = list(Path(tmpdir).glob("*.srt"))
            vtt_files = list(Path(tmpdir).glob("*.vtt"))
            sub_files = srt_files or vtt_files

            if sub_files:
                raw = sub_files[0].read_text(encoding="utf-8", errors="replace")
                transcript = _clean_srt(raw)
                _cache_transcript(vid_id, language, transcript, title)
                results.append(
                    f"  [{vid_id}] OK — {title} ({len(transcript):,} chars)"
                )
            else:
                results.append(f"  [{vid_id}] NO SUBS — {title}")

    return f"Bulk transcription ({len(ids)} videos):\n" + "\n".join(results)


# ═══════════════════════════════════════════════════════════════════════
# YOUTUBE COMMENT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════


@tool
async def youtube_get_comments(
    video_url: str,
    max_comments: int = 100,
    sort_by: str = "relevance",
) -> str:
    """Extract comments from a YouTube video.

    Comments are a premium intelligence source — real people sharing opinions,
    recommendations, corrections, and anecdotes. A Filipino viewer commenting
    on a banana variety video is ground-truth intelligence.

    Uses yt-dlp's comment extraction (no API key needed, but slower).
    For faster extraction with more comments, set YOUTUBE_API_KEY.

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
        return await _get_comments_api(video_id, max_comments, sort_by, api_key)

    # Fall back to yt-dlp comment extraction
    if not _yt_dlp_available():
        return (
            "[TOOL_ERROR] Neither YOUTUBE_API_KEY nor yt-dlp available. "
            "Set YOUTUBE_API_KEY in .env or install yt-dlp."
        )

    return _get_comments_ytdlp(video_id, max_comments, sort_by)


async def _get_comments_api(
    video_id: str,
    max_comments: int,
    sort_by: str,
    api_key: str,
) -> str:
    """Extract comments via YouTube Data API v3."""

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
            resp = await async_get(
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
            comments.append({
                "author": snippet.get("authorDisplayName", "Anonymous"),
                "text": snippet.get("textDisplay", ""),
                "likes": snippet.get("likeCount", 0),
                "replies": item["snippet"].get("totalReplyCount", 0),
                "published": snippet.get("publishedAt", "")[:10],
            })

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
            return f"[TOOL_ERROR] yt-dlp comment extraction failed: {result.stderr[:500]}"

        info = json.loads(result.stdout)
        raw_comments = info.get("comments", [])

        if not raw_comments:
            return f"No comments found for video {video_id}."

        comments = []
        for c in raw_comments[:max_comments]:
            comments.append({
                "author": c.get("author", "Anonymous"),
                "text": c.get("text", ""),
                "likes": c.get("like_count", 0),
                "replies": 0,  # yt-dlp doesn't report reply count easily
                "published": c.get("timestamp", 0),
            })

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


# ═══════════════════════════════════════════════════════════════════════
# CHANNEL HARVESTER — bulk download transcripts + comments via
# Apify / Bright Data / yt-dlp cascade.  Results stored in Miro's cache
# as first-class resources for later swarm synthesis.
# ═══════════════════════════════════════════════════════════════════════


@tool
def youtube_harvest_channel(
    channel: str,
    max_videos: int = 0,
    language: str = "en",
    include_comments: bool = True,
) -> str:
    """Download ALL transcripts and comments from a YouTube channel for later swarm processing.

    This is the bulk harvester — it enumerates every video on the channel,
    downloads transcripts and top comments via multiple backends
    (Apify → Bright Data → yt-dlp → youtube-transcript-api) with automatic
    fallback, and stores everything in Miro's cache so it's recognized as a
    first-class research resource.

    Use this when you want to stockpile an entire channel's knowledge for
    later analysis.  The corpus can then be exported and fed to the swarm
    engine for deep synthesis.

    Backends tried in order:
    1. Apify visita/youtube-scraper ($8/1K videos — transcripts + comments + metadata)
    2. Bright Data YouTube Scraper API (comments only — separate dataset)
    3. yt-dlp (needs cookies or proxy for cloud IPs)
    4. youtube-transcript-api (needs proxy for cloud IPs)

    Args:
        channel: YouTube channel URL, handle (@MorePlatesMoreDates),
            or channel ID (UCxxxx).
        max_videos: Maximum videos to harvest (0 = all videos on channel).
        language: Preferred transcript language code (default "en").
        include_comments: Whether to also download top comments per video.

    Returns:
        Summary of harvest results including counts, chars, and any errors.
    """
    import asyncio as _asyncio

    # Import harvester library
    try:
        import sys as _sys
        _sys.path.insert(0, os.path.join(
            os.path.dirname(__file__), "..", "..", "tools",
        ))
        from youtube_harvester import Harvester, HarvestConfig
    except ImportError:
        return (
            "[TOOL_ERROR] youtube_harvester module not found. "
            "Ensure tools/youtube_harvester.py exists in the MiroThinker repo."
        )

    # Wire in Miro cache
    cache_fn = None
    try:
        from cache import cache_put
        cache_fn = cache_put
    except ImportError:
        logger.debug("Cache module not available — transcripts will not be cached")

    config = HarvestConfig.from_env()
    config.language = language
    config.fetch_comments = include_comments

    harvester = Harvester(config=config, cache_fn=cache_fn)

    # Run the async harvester — strands tools always run in sync threads
    # so asyncio.run() is safe and avoids SQLite check_same_thread issues
    # that would occur with ThreadPoolExecutor
    try:
        result = _asyncio.run(
            harvester.harvest_channel(channel, max_videos=max_videos)
        )
    except Exception as exc:
        return f"[TOOL_ERROR] Harvest failed: {exc}"

    # Format output
    lines = [
        f"# Channel Harvest Complete: {result.channel_name}\n",
        result.summary(),
    ]

    if result.errors:
        lines.append("\n**Errors:**")
        for err in result.errors[:10]:
            lines.append(f"- {err}")

    # Show backend distribution
    backend_counts: dict[str, int] = {}
    for v in result.videos:
        if v.status == "downloaded":
            backend_counts[v.backend_used] = backend_counts.get(v.backend_used, 0) + 1
    if backend_counts:
        lines.append("\n**Backends used:**")
        for backend, count in sorted(backend_counts.items()):
            lines.append(f"- {backend}: {count} videos")

    # Show topic distribution
    topic_counts: dict[str, int] = {}
    for v in result.videos:
        for t in v.topics:
            topic_counts[t] = topic_counts.get(t, 0) + 1
    if topic_counts:
        lines.append("\n**Topics detected:**")
        for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1])[:15]:
            lines.append(f"- {topic}: {count} videos")

    lines.append(
        "\nAll transcripts and comments are now stored in Miro's cache "
        "and available for swarm synthesis via youtube_export_corpus."
    )

    return "\n".join(lines)


@tool
def youtube_export_corpus(
    channel: str = "",
    max_chars: int = 0,
    include_comments: bool = True,
    topics_filter: str = "",
) -> str:
    """Export cached YouTube transcripts as a corpus file ready for swarm processing.

    Searches Miro's cache for previously harvested YouTube transcripts and
    assembles them into a single markdown corpus document.  This can then
    be fed directly to the swarm engine for deep synthesis.

    Args:
        channel: Filter by channel name (empty = all channels).
        max_chars: Maximum corpus size in characters (0 = unlimited).
        include_comments: Whether to include comments in the corpus.
        topics_filter: Comma-separated topic tags to filter by
            (e.g. "steroids,testosterone"). Empty = all topics.

    Returns:
        The assembled corpus text, or a summary if too large to return.
    """
    try:
        from cache import cache_search
    except ImportError:
        return (
            "[TOOL_ERROR] Cache module not available. "
            "Run youtube_harvest_channel first to populate the cache."
        )

    # Search for harvested transcripts — track which source_type matched
    active_source_type = "youtube_harvest"
    results = cache_search(
        source_type="youtube_harvest",
        tag="transcript",
        limit=10000,
    )

    if not results:
        # Also check the older source_type
        active_source_type = "video_transcript"
        results = cache_search(
            source_type="video_transcript",
            tag="youtube",
            limit=10000,
        )

    if not results:
        return "No harvested YouTube transcripts found in cache. Run youtube_harvest_channel first."

    # Apply filters — cache_search returns url, title, content_hash, etc.
    # but NOT metadata. Filter on available fields.
    if channel:
        channel_lower = channel.lower().lstrip("@")
        results = [
            r for r in results
            if channel_lower in (r.get("title", "") or "").lower()
            or channel_lower in (r.get("url", "") or "").lower()
            or channel_lower in (r.get("summary", "") or "").lower()
        ]

    if topics_filter:
        # Topic tags are stored as cache tags — re-query with tag filter
        # Use the same source_type that produced the initial results
        filter_topics = [t.strip().lower() for t in topics_filter.split(",")]
        topic_results: list[dict] = []
        seen_ids: set[int] = set()
        for topic_tag in filter_topics:
            tagged = cache_search(
                source_type=active_source_type,
                tag=topic_tag,
                limit=10000,
            )
            for r in tagged:
                rid = r.get("id", 0)
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    topic_results.append(r)
        # Intersect with current results (by content_hash)
        result_hashes = {r.get("content_hash") for r in results}
        results = [r for r in topic_results if r.get("content_hash") in result_hashes]

    if not results:
        return f"No transcripts match the filter (channel={channel!r}, topics={topics_filter!r})."

    # Assemble corpus
    corpus_parts = []
    total_chars = 0

    for r in results:
        try:
            from cache import read_blob
            content_bytes = read_blob(r["content_hash"])
            if not content_bytes:
                continue
            content = content_bytes.decode("utf-8", errors="replace")
        except Exception:
            continue

        if not include_comments:
            # Strip comments section if present
            comment_idx = content.find("\n### Comments (")
            if comment_idx > 0:
                content = content[:comment_idx]

        if max_chars > 0 and total_chars + len(content) > max_chars:
            break

        corpus_parts.append(content)
        total_chars += len(content)

    if not corpus_parts:
        return "Found entries but could not read transcript content from cache."

    header = (
        f"# YouTube Corpus Export\n\n"
        f"**Videos:** {len(corpus_parts)}\n"
        f"**Total chars:** {total_chars:,}\n"
        f"**Channel filter:** {channel or 'all'}\n"
        f"**Topic filter:** {topics_filter or 'all'}\n\n"
        f"---\n\n"
    )
    corpus = header + "\n\n---\n\n".join(corpus_parts)

    # If corpus is very large, return summary + first chunk
    if len(corpus) > 100_000:
        return (
            f"Corpus assembled: {len(corpus_parts)} videos, {total_chars:,} chars total.\n"
            f"Too large to return inline — use this tool's output as input to the swarm engine.\n\n"
            f"**First 50K chars preview:**\n\n{corpus[:50000]}\n\n"
            f"[...truncated — {len(corpus) - 50000:,} more chars available...]"
        )

    return corpus


# ── Tool registry ─────────────────────────────────────────────────────

YOUTUBE_TOOLS = [
    # TranscriptAPI search tools (native REST — always available when key is set)
    search_youtube,
    search_channel_videos,
    get_channel_latest_videos,
    # YouTube intelligence tools (yt-dlp / harvester backends)
    youtube_download_transcript,
    youtube_video_info,
    youtube_channel_list,
    youtube_bulk_transcribe,
    youtube_get_comments,
    youtube_harvest_channel,
    youtube_export_corpus,
]

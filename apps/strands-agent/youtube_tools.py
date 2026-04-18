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

from strands import tool
from async_http import async_get

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


# ── Tool registry ─────────────────────────────────────────────────────

YOUTUBE_TOOLS = [
    youtube_download_transcript,
    youtube_video_info,
    youtube_channel_list,
    youtube_bulk_transcribe,
    youtube_get_comments,
]

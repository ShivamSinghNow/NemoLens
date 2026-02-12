"""YouTube video downloader using yt-dlp."""

import re
import tempfile
import os
import subprocess
import json


# ── URL helpers ───────────────────────────────────────────────────────────────

_YT_PATTERNS = [
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([\w-]{11})"),
    re.compile(r"(?:https?://)?youtu\.be/([\w-]{11})"),
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/embed/([\w-]{11})"),
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/shorts/([\w-]{11})"),
]

MAX_DURATION_SECONDS = 3600  # 1-hour cap to avoid huge downloads


def extract_video_id(url: str) -> str | None:
    """Extract the 11-character YouTube video ID from a URL.

    Returns None if the URL is not a recognised YouTube format.
    """
    url = url.strip()
    for pat in _YT_PATTERNS:
        m = pat.search(url)
        if m:
            return m.group(1)
    return None


def is_youtube_url(url: str) -> bool:
    """Return True if *url* looks like a valid YouTube link."""
    return extract_video_id(url) is not None


# ── Video metadata ────────────────────────────────────────────────────────────

def get_video_info(url: str) -> dict:
    """Fetch lightweight metadata (title, duration, thumbnail) without downloading.

    Raises RuntimeError on failure.
    """
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--no-download",
                "--print-json",
                "--no-playlist",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "yt-dlp metadata fetch failed")
        info = json.loads(result.stdout)
        return {
            "title": info.get("title", "Unknown"),
            "duration": info.get("duration", 0),
            "thumbnail": info.get("thumbnail"),
            "uploader": info.get("uploader", "Unknown"),
        }
    except FileNotFoundError:
        raise RuntimeError(
            "yt-dlp is not installed. Run:  pip install yt-dlp"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Timed out fetching video info.")


# ── Download ──────────────────────────────────────────────────────────────────

def download_youtube_video(
    url: str,
    max_duration: int = MAX_DURATION_SECONDS,
    preferred_height: int = 720,
) -> tuple[bytes, str]:
    """Download a YouTube video and return (video_bytes, title).

    The video is downloaded as MP4, capped at *preferred_height*p to keep
    file sizes reasonable.

    Args:
        url: Full YouTube URL.
        max_duration: Reject videos longer than this (seconds).
        preferred_height: Maximum vertical resolution (default 720p).

    Returns:
        Tuple of (raw MP4 bytes, video title string).

    Raises:
        ValueError: If the URL is invalid or the video is too long.
        RuntimeError: If the download fails.
    """
    if not is_youtube_url(url):
        raise ValueError("Not a valid YouTube URL.")

    # Fetch metadata first so we can enforce the duration cap
    info = get_video_info(url)
    duration = info.get("duration", 0)
    if duration > max_duration:
        mins = max_duration // 60
        raise ValueError(
            f"Video is {duration // 60} min long — maximum allowed is {mins} min."
        )

    tmp_dir = tempfile.mkdtemp()
    output_template = os.path.join(tmp_dir, "video.%(ext)s")

    try:
        cmd = [
            "yt-dlp",
            "--no-playlist",
            # Pick best mp4 up to preferred_height
            "-f", f"bestvideo[height<={preferred_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={preferred_height}][ext=mp4]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", output_template,
            url,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout for the download itself
        )

        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip() or "yt-dlp download failed"
            )

        # Find the downloaded file
        downloaded = None
        for fname in os.listdir(tmp_dir):
            if fname.endswith(".mp4"):
                downloaded = os.path.join(tmp_dir, fname)
                break

        if downloaded is None:
            raise RuntimeError("Download completed but no MP4 file was found.")

        with open(downloaded, "rb") as f:
            video_bytes = f.read()

        return video_bytes, info.get("title", "YouTube Video")

    except FileNotFoundError:
        raise RuntimeError(
            "yt-dlp is not installed. Run:  pip install yt-dlp"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Download timed out (10-minute limit).")
    finally:
        # Clean up temp files
        for fname in os.listdir(tmp_dir):
            try:
                os.unlink(os.path.join(tmp_dir, fname))
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

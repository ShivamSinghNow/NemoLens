"""Local transcription using OpenAI Whisper."""

import tempfile
import os
import subprocess
import whisper

# Use "base" for speed on M1; "small" or "medium" for accuracy
DEFAULT_MODEL = "base"

_model_cache: dict[str, whisper.Whisper] = {}


def _get_model(model_name: str = DEFAULT_MODEL) -> whisper.Whisper:
    if model_name not in _model_cache:
        _model_cache[model_name] = whisper.load_model(model_name)
    return _model_cache[model_name]


def _verify_audio_stream(video_path: str) -> None:
    """Raise a clear error if the video file has no audio stream.

    Uses ffprobe (bundled with ffmpeg) to check for audio tracks.
    This prevents Whisper from crashing with a cryptic tensor reshape
    error when it tries to process a file with zero audio samples.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = result.stdout.strip()
        if not output:
            raise ValueError(
                "This video file does not contain an audio track. "
                "Whisper requires audio to generate a transcript. "
                "Please try a different video that includes audio, "
                "or re-download with a format that includes sound."
            )
    except FileNotFoundError:
        # ffprobe not installed — skip the check and let Whisper try
        pass
    except subprocess.TimeoutExpired:
        pass  # skip check on timeout


def transcribe_video(
    video_bytes: bytes,
    model_name: str = DEFAULT_MODEL,
) -> dict:
    """Transcribe a video file using Whisper.

    Args:
        video_bytes: Raw video file bytes.
        model_name: Whisper model size ("tiny", "base", "small", "medium").

    Returns:
        Dict with keys:
            - "text": Full transcript string.
            - "segments": List of segment dicts with keys:
                - "start": float (seconds)
                - "end": float (seconds)
                - "text": str
            - "language": Detected language code.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        # Verify the file actually contains an audio stream before sending
        # to Whisper.  Without this check, an audio-less video causes a
        # cryptic "cannot reshape tensor of 0 elements" crash.
        _verify_audio_stream(tmp_path)

        model = _get_model(model_name)
        result = model.transcribe(
            tmp_path,
            verbose=False,
            word_timestamps=False,
            fp16=False,
        )

        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            })

        return {
            "text": result.get("text", "").strip(),
            "segments": segments,
            "language": result.get("language", "en"),
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def get_transcript_for_range(
    segments: list[dict], start: float, end: float
) -> str:
    """Extract transcript text that overlaps a given time range."""
    parts = []
    for seg in segments:
        if seg["end"] > start and seg["start"] < end:
            parts.append(seg["text"])
    return " ".join(parts)


def format_transcript_with_timestamps(segments: list[dict]) -> str:
    """Format the full transcript with timestamps for display."""
    lines = []
    for seg in segments:
        ts = _fmt(seg["start"])
        lines.append(f"[{ts}] {seg['text']}")
    return "\n".join(lines)


def search_transcript(segments: list[dict], query: str) -> list[dict]:
    """Search transcript segments for a query string (case-insensitive).

    Returns matching segments sorted by start time.
    """
    query_lower = query.lower()
    return [
        seg for seg in segments
        if query_lower in seg["text"].lower()
    ]


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

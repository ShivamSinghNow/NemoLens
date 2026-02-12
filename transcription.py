"""Local transcription using OpenAI Whisper."""

import tempfile
import os
import whisper

# Use "base" for speed on M1; "small" or "medium" for accuracy
DEFAULT_MODEL = "base"

_model_cache: dict[str, whisper.Whisper] = {}


def _get_model(model_name: str = DEFAULT_MODEL) -> whisper.Whisper:
    if model_name not in _model_cache:
        _model_cache[model_name] = whisper.load_model(model_name)
    return _model_cache[model_name]


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

        model = _get_model(model_name)
        result = model.transcribe(
            tmp_path,
            verbose=False,
            word_timestamps=False,
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

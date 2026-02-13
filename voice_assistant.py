"""Voice assistant — Whisper STT + Nemotron Q&A for video context."""

import os
import tempfile

from transcription import _get_model
from nemotron_client import text_completion


def transcribe_audio(audio_bytes: bytes, model_name: str = "base") -> str:
    """Convert spoken audio to text using Whisper.

    Args:
        audio_bytes: Raw audio bytes (WAV format from st.audio_input).
        model_name: Whisper model size ("tiny", "base", "small").

    Returns:
        Transcribed text string.
    """
    # Detect audio container format for the correct temp-file extension.
    # MediaRecorder (floating mic) produces WebM; st.audio_input sends WAV.
    if audio_bytes[:4] == b"\x1aE\xdf\xa3":      # EBML header → WebM
        suffix = ".webm"
    elif audio_bytes[:4] == b"OggS":               # Ogg container
        suffix = ".ogg"
    else:                                           # Default (WAV / other)
        suffix = ".wav"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        model = _get_model(model_name)
        result = model.transcribe(tmp_path, verbose=False, word_timestamps=False)
        return result.get("text", "").strip()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def answer_question(question: str, video_context: dict) -> str:
    """Send a question with video context to Nemotron and return a voice-friendly answer.

    Args:
        question: The user's question text.
        video_context: Dict with keys transcript, chapters, takeaways,
                       visual_descriptions (all formatted strings).

    Returns:
        Answer string optimised for spoken output.
    """
    system_prompt = (
        "You are Nemo, a friendly voice assistant for NemoLens, a video analysis tool.\n"
        "You have full knowledge of the video the user is watching.\n\n"
        f"VIDEO TRANSCRIPT:\n{video_context.get('transcript', 'Not available')}\n\n"
        f"CHAPTERS:\n{video_context.get('chapters', 'Not available')}\n\n"
        f"KEY TAKEAWAYS:\n{video_context.get('takeaways', 'Not available')}\n\n"
        f"VISUAL DESCRIPTIONS:\n{video_context.get('visual_descriptions', 'Not available')}\n\n"
        "Instructions:\n"
        "- Answer based ONLY on the video content above.\n"
        "- Reference specific timestamps in MM:SS format when relevant "
        '(e.g., "At 03:24, the speaker explains...").\n'
        "- Keep responses concise — 2-3 sentences ideal since this will be spoken aloud.\n"
        "- If the answer is not in the video, say so honestly.\n"
        "- Do NOT use markdown, bullet points, or special formatting.\n"
        "- Be conversational and friendly."
    )

    return text_completion(
        prompt=question,
        system=system_prompt,
        max_tokens=512,
    )

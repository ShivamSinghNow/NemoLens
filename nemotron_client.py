"""Nemotron API client via NVIDIA Build (integrate.api.nvidia.com)."""

import os
import time
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger(__name__)

NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "nvidia/nemotron-nano-12b-v2-vl"
MAX_IMAGES_PER_REQUEST = 5
MAX_WORKERS = 3  # parallel API calls
REQUEST_TIMEOUT = 300  # seconds — vision requests with 5 images can be slow
MAX_RETRIES = 3
BASE_RETRY_DELAY = 3  # seconds; grows exponentially per attempt


def _get_key(api_key: str | None = None) -> str:
    return api_key or os.getenv("NVIDIA_API_KEY", "")


def _headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _post(payload: dict, api_key: str, retries: int = MAX_RETRIES) -> dict:
    """POST to NVIDIA Build with retry on 429, timeouts, and connection errors."""
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                NVIDIA_API_URL,
                headers=_headers(api_key),
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 429 and attempt < retries:
                delay = BASE_RETRY_DELAY * (2 ** attempt)
                log.warning("Rate limited (429), retrying in %ds…", delay)
                time.sleep(delay)
                continue
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise RuntimeError(
                    data["error"].get("message", str(data["error"]))
                )
            return data
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as exc:
            last_err = exc
            if attempt < retries:
                delay = BASE_RETRY_DELAY * (2 ** attempt)
                log.warning(
                    "%s on attempt %d/%d, retrying in %ds…",
                    type(exc).__name__, attempt + 1, retries + 1, delay,
                )
                time.sleep(delay)
                continue
            raise RuntimeError(
                f"NVIDIA API request failed after {retries + 1} attempts: {exc}"
            ) from exc
    raise RuntimeError(
        f"NVIDIA API request failed after {retries + 1} attempts: {last_err}"
    )


# ── Vision requests ──────────────────────────────────────────────────────────

def describe_frames(
    frames_b64: list[str],
    prompt: str,
    api_key: str | None = None,
) -> str:
    """Send up to 5 base64 frames + a prompt to Nemotron VL via NVIDIA Build."""
    key = _get_key(api_key)
    if not key:
        raise RuntimeError("NVIDIA_API_KEY is not set.")

    frames_b64 = frames_b64[:MAX_IMAGES_PER_REQUEST]

    content: list[dict] = [{"type": "text", "text": prompt}]
    for b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "/think"},
            {"role": "user", "content": content},
        ],
        "max_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.7,
    }
    data = _post(payload, key)
    return data["choices"][0]["message"]["content"]


def describe_segments_parallel(
    segments: list[dict],
    api_key: str | None = None,
    on_complete=None,
) -> list[dict]:
    """Process multiple video segments in parallel.

    Args:
        segments: List of dicts with keys:
            - "frames_b64": list[str]
            - "start_time": float (seconds)
            - "end_time": float (seconds)
            - "transcript_chunk": str
        on_complete: Optional callback(completed_count, total) for progress.

    Returns:
        List of dicts with original segment info + "visual_description".
    """
    key = _get_key(api_key)
    if not key:
        raise RuntimeError("NVIDIA_API_KEY is not set.")

    results = [None] * len(segments)
    completed = 0

    def _process(idx: int, seg: dict) -> tuple[int, str]:
        transcript_ctx = seg.get("transcript_chunk", "")
        prompt = (
            "You are analyzing a segment of a video. "
            f"This segment spans from {_fmt(seg['start_time'])} to {_fmt(seg['end_time'])}.\n"
        )
        if transcript_ctx:
            prompt += f"The audio transcript for this segment is:\n\"{transcript_ctx}\"\n\n"
        prompt += (
            "Describe what you see in these frames in detail. Include:\n"
            "- Key visual elements, people, text on screen\n"
            "- Actions and changes between frames\n"
            "- Any slides, diagrams, or presentations shown\n"
            "Keep it factual and concise."
        )
        return idx, describe_frames(seg["frames_b64"], prompt, api_key=key)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_process, i, seg): i
            for i, seg in enumerate(segments)
        }
        for future in as_completed(futures):
            idx = futures[future]
            seg = segments[idx]
            try:
                _, description = future.result()
            except Exception as exc:
                # Graceful degradation: use the transcript as a fallback
                # description so one failed segment doesn't kill the
                # entire analysis pipeline.
                log.warning(
                    "Segment %d (%s–%s) failed: %s — using transcript fallback",
                    idx, _fmt(seg["start_time"]), _fmt(seg["end_time"]), exc,
                )
                transcript = seg.get("transcript_chunk", "")
                description = (
                    f"[Visual analysis unavailable for this segment] "
                    f"{transcript}"
                ) if transcript else (
                    "[Visual analysis unavailable for this segment]"
                )
            results[idx] = {
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "transcript_chunk": seg.get("transcript_chunk", ""),
                "visual_description": description,
            }
            completed += 1
            if on_complete:
                on_complete(completed, len(segments))

    return results


# ── Text-only requests ───────────────────────────────────────────────────────

def text_completion(
    prompt: str,
    system: str = "",
    api_key: str | None = None,
    max_tokens: int = 2048,
) -> str:
    """Send a text-only request to Nemotron via NVIDIA Build."""
    key = _get_key(api_key)
    if not key:
        raise RuntimeError("NVIDIA_API_KEY is not set.")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 0.7,
    }
    data = _post(payload, key)
    return data["choices"][0]["message"]["content"]


def chat_with_context(
    context: str,
    question: str,
    chat_history: list[dict] | None = None,
    api_key: str | None = None,
) -> str:
    """Answer a question using full video context + chat history."""
    key = _get_key(api_key)
    if not key:
        raise RuntimeError("NVIDIA_API_KEY is not set.")

    system = (
        "You are a video analysis assistant powered by NVIDIA Nemotron. "
        "You have access to a complete analysis of a video including its transcript, "
        "visual descriptions, and auto-generated chapters.\n\n"
        "IMPORTANT GUIDELINES:\n"
        "- Answer questions accurately based on the video context below.\n"
        "- ALWAYS reference specific timestamps in MM:SS format when mentioning "
        "video content (e.g., 'At 03:15, the speaker explains...'). "
        "Users can click timestamps to jump to that moment in the video.\n"
        "- If multiple moments are relevant, list each with its timestamp.\n"
        "- Be concise but thorough. Quote the transcript when helpful.\n\n"
        f"=== VIDEO ANALYSIS ===\n{context}"
    )

    messages = [{"role": "system", "content": system}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": question})

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.7,
    }
    data = _post(payload, key)
    return data["choices"][0]["message"]["content"]


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

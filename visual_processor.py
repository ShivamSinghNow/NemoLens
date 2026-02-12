"""Video frame extraction and parallel visual analysis via Nemotron."""

import cv2
import base64
import tempfile
import os

from transcription import get_transcript_for_range
from nemotron_client import describe_segments_parallel

SEGMENT_DURATION = 120  # 2 minutes per segment
FRAMES_PER_SEGMENT = 5  # NVIDIA Build allows max 5 images per request
MAX_PX = 512


def extract_segment_frames(
    video_bytes: bytes,
    segment_duration: int = SEGMENT_DURATION,
    frames_per_segment: int = FRAMES_PER_SEGMENT,
) -> list[dict]:
    """Split video into time segments and extract evenly-spaced frames from each.

    Returns:
        List of segment dicts:
            - "start_time": float
            - "end_time": float
            - "frames_b64": list[str]  (base64 JPEGs)
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if total_frames <= 0 or fps <= 0:
            raise ValueError("Invalid video metadata.")

        duration = total_frames / fps
        segments = []

        t = 0.0
        while t < duration:
            seg_end = min(t + segment_duration, duration)
            seg_duration = seg_end - t

            # For very short segments, reduce frame count proportionally
            n_frames = max(1, int(frames_per_segment * seg_duration / segment_duration))

            frames_b64 = []
            if n_frames == 1:
                sample_times = [t + seg_duration / 2]
            else:
                step = seg_duration / (n_frames - 1)
                sample_times = [t + i * step for i in range(n_frames)]

            for st in sample_times:
                frame_idx = min(int(st * fps), total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                h, w = frame.shape[:2]
                scale = MAX_PX / max(h, w)
                if scale < 1.0:
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                frames_b64.append(base64.b64encode(buf).decode("utf-8"))

            segments.append({
                "start_time": t,
                "end_time": seg_end,
                "frames_b64": frames_b64,
            })
            t = seg_end

        cap.release()
        return segments
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def analyze_video(
    video_bytes: bytes,
    transcript_segments: list[dict],
    on_progress=None,
) -> list[dict]:
    """Full visual analysis pipeline: extract frames, pair with transcript, send to Nemotron.

    Args:
        video_bytes: Raw video bytes.
        transcript_segments: Whisper segments with start/end/text.
        on_progress: Optional callback(completed, total) for progress updates.

    Returns:
        List of analyzed segment dicts with keys:
            start_time, end_time, transcript_chunk, visual_description, frames_b64
    """
    raw_segments = extract_segment_frames(video_bytes)

    # Attach transcript chunks to each segment
    for seg in raw_segments:
        seg["transcript_chunk"] = get_transcript_for_range(
            transcript_segments, seg["start_time"], seg["end_time"]
        )

    results = describe_segments_parallel(
        raw_segments,
        on_complete=on_progress,
    )

    # Carry frames_b64 through for potential thumbnail use
    for i, r in enumerate(results):
        r["frames_b64"] = raw_segments[i]["frames_b64"]

    return results

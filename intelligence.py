"""Intelligence layer: chapters, takeaways, search, and context building."""

import json
import re
from nemotron_client import text_completion


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _get_max_time(analyzed_segments: list[dict]) -> float:
    """Return the latest end_time across all analyzed segments."""
    if not analyzed_segments:
        return 0.0
    return max(seg.get("end_time", 0) for seg in analyzed_segments)


def build_full_context(analyzed_segments: list[dict]) -> str:
    """Combine all segment data into a single context string."""
    parts = []
    for seg in analyzed_segments:
        ts = f"{_fmt(seg['start_time'])} - {_fmt(seg['end_time'])}"
        block = f"[{ts}]\n"
        if seg.get("transcript_chunk"):
            block += f"Transcript: {seg['transcript_chunk']}\n"
        if seg.get("visual_description"):
            block += f"Visual: {seg['visual_description']}\n"
        parts.append(block)
    return "\n".join(parts)


def generate_chapters(analyzed_segments: list[dict]) -> list[dict]:
    """Use Nemotron to identify topic changes and generate chapters.

    Returns:
        List of dicts: {"time": float_seconds, "title": str, "summary": str}
    """
    context = build_full_context(analyzed_segments)

    prompt = (
        "You are analyzing a video. Below is the full transcript and visual description "
        "organized by time segments.\n\n"
        f"{context}\n\n"
        "Based on this content, identify the major topic changes and create chapters. "
        "Return ONLY a JSON array of objects with these keys:\n"
        '- "time": start time in seconds (number)\n'
        '- "title": short chapter title (under 60 chars)\n'
        '- "summary": one-sentence summary\n\n'
        "Create between 3-10 chapters depending on video length and content changes. "
        "The first chapter should start at time 0. "
        "Return ONLY valid JSON, no markdown fences."
    )

    raw = text_completion(prompt, max_tokens=2048)
    chapters = _parse_json_array(raw, fallback_chapters(analyzed_segments))

    # Clamp timestamps to actual video duration
    max_time = _get_max_time(analyzed_segments)
    for ch in chapters:
        if isinstance(ch, dict) and "time" in ch:
            ch["time"] = max(0, min(float(ch["time"]), max_time))
    return chapters


def generate_takeaways(analyzed_segments: list[dict]) -> list[str]:
    """Generate key takeaways from the video content.

    Returns:
        List of bullet-point strings.
    """
    context = build_full_context(analyzed_segments)

    prompt = (
        "You are analyzing a video. Below is the full transcript and visual description.\n\n"
        f"{context}\n\n"
        "Extract the 5-8 most important key takeaways from this video. "
        "Return ONLY a JSON array of strings, each being one takeaway. "
        "Be specific and reference concrete details from the content. "
        "Return ONLY valid JSON, no markdown fences."
    )

    raw = text_completion(prompt, max_tokens=1024)
    result = _parse_json_array(raw, ["Could not generate takeaways."])
    # Ensure we have a list of strings
    return [str(item) for item in result]


def generate_study_guide(analyzed_segments: list[dict]) -> list[dict]:
    """Generate a detailed study guide breaking down the video by topic.

    Returns:
        List of dicts with keys:
            topic, timestamp, overview, details, key_terms
    """
    context = build_full_context(analyzed_segments)

    prompt = (
        "You are an expert educator creating a study guide from a video. "
        "Below is the full transcript and visual description organized by time.\n\n"
        f"{context}\n\n"
        "Create a comprehensive study guide that breaks the video into major topics. "
        "For each topic, provide:\n"
        '- "topic": a clear, descriptive topic title\n'
        '- "timestamp": the start time in seconds (number) where this topic begins\n'
        '- "overview": a 1-2 sentence high-level summary of the topic\n'
        '- "details": a thorough 3-6 sentence explanation covering what was discussed, '
        "key arguments, examples given, and any data or visuals shown. "
        "Write this as if explaining to a student who missed the lecture.\n"
        '- "key_terms": an array of 2-5 important terms, concepts, or names mentioned\n\n'
        "Create between 4-8 topics depending on video length. "
        "Cover ALL major content -- do not skip sections. "
        "Return ONLY a valid JSON array, no markdown fences."
    )

    fallback = [
        {
            "topic": seg.get("transcript_chunk", "")[:50] or f"Segment at {_fmt(seg['start_time'])}",
            "timestamp": seg["start_time"],
            "overview": (seg.get("visual_description") or "")[:120],
            "details": seg.get("transcript_chunk", ""),
            "key_terms": [],
        }
        for seg in analyzed_segments
    ]

    raw = text_completion(prompt, max_tokens=3000)
    guide = _parse_json_array(raw, fallback)

    # Clamp timestamps to actual video duration
    max_time = _get_max_time(analyzed_segments)
    for topic in guide:
        if isinstance(topic, dict) and "timestamp" in topic:
            topic["timestamp"] = max(0, min(float(topic["timestamp"]), max_time))
    return guide


def generate_questions(analyzed_segments: list[dict]) -> list[dict]:
    """Generate practice questions to test comprehension of the video content.

    Returns:
        List of dicts with keys:
            question, type, options (for MC), answer, explanation
    """
    context = build_full_context(analyzed_segments)

    prompt = (
        "You are an expert educator creating a quiz to test a student's understanding "
        "of the following video content.\n\n"
        f"{context}\n\n"
        "Generate 8-12 practice questions that test real understanding (not just recall). "
        "Mix the following question types:\n"
        "- Multiple choice (provide 4 options labeled A, B, C, D)\n"
        "- Short answer / open-ended\n"
        "- True/False\n\n"
        "For EACH question return a JSON object with:\n"
        '- "question": the question text\n'
        '- "type": one of "multiple_choice", "short_answer", or "true_false"\n'
        '- "options": array of 4 option strings (only for multiple_choice, omit for others)\n'
        '- "answer": the correct answer (letter for MC, "True"/"False" for T/F, '
        "or a model answer for short answer)\n"
        '- "explanation": 1-2 sentence explanation of WHY this is the correct answer, '
        "referencing specific video content\n\n"
        "Make questions progressively harder: start with basic recall, "
        "then comprehension, then application/analysis. "
        "Return ONLY a valid JSON array, no markdown fences."
    )

    fallback = [
        {
            "question": "What were the main topics discussed in this video?",
            "type": "short_answer",
            "answer": "Review the study guide above for a full breakdown.",
            "explanation": "Questions could not be auto-generated.",
        }
    ]

    raw = text_completion(prompt, max_tokens=3000)
    return _parse_json_array(raw, fallback)


def evaluate_short_answer(
    question: str,
    correct_answer: str,
    user_answer: str,
) -> dict:
    """Use Nemotron to evaluate a student's short-answer response.

    Returns:
        Dict with keys: "correct" (bool), "feedback" (str)
    """
    prompt = (
        "You are a fair but encouraging teacher grading a student's short answer.\n\n"
        f"Question: {question}\n"
        f"Model Answer: {correct_answer}\n"
        f"Student's Answer: {user_answer}\n\n"
        "Evaluate whether the student's answer demonstrates understanding of the concept. "
        "Minor wording differences are fine -- focus on whether the core idea is correct.\n\n"
        "Return ONLY a JSON object with:\n"
        '- "correct": true if the answer is substantially correct, false otherwise\n'
        '- "feedback": a brief 1-2 sentence response. If correct, give a short positive '
        "confirmation. If incorrect, explain what the right answer is and why, in a "
        "helpful and concise way.\n\n"
        "Return ONLY valid JSON, no markdown fences."
    )

    raw = text_completion(prompt, max_tokens=300)

    # Parse response
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = cleaned.strip().rstrip("`")
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        try:
            result = json.loads(cleaned[start : end + 1])
            return {
                "correct": bool(result.get("correct", False)),
                "feedback": str(result.get("feedback", "")),
            }
        except json.JSONDecodeError:
            pass

    return {"correct": False, "feedback": "Could not evaluate your answer. Please review the answer key in the PDF."}


def search_video(
    query: str,
    analyzed_segments: list[dict],
    transcript_segments: list[dict],
    chapters: list[dict] | None = None,
) -> list[dict]:
    """Search for where a topic is discussed using transcript, visual, and chapter data.

    Returns:
        List of dicts: {"time": float, "timestamp": str, "context": str, "source": str}
    """
    results = []
    query_lower = query.lower()

    # Search transcript segments (fine-grained)
    for seg in transcript_segments:
        if query_lower in seg["text"].lower():
            results.append({
                "time": seg["start"],
                "timestamp": _fmt(seg["start"]),
                "context": seg["text"],
                "source": "transcript",
            })

    # Search visual descriptions
    for seg in analyzed_segments:
        desc = seg.get("visual_description", "")
        if query_lower in desc.lower():
            results.append({
                "time": seg["start_time"],
                "timestamp": _fmt(seg["start_time"]),
                "context": desc[:200] + ("..." if len(desc) > 200 else ""),
                "source": "visual",
            })

    # Search chapter titles and summaries
    if chapters:
        for ch in chapters:
            title = ch.get("title", "")
            summary = ch.get("summary", "")
            if query_lower in title.lower() or query_lower in summary.lower():
                context = f"{title}: {summary}" if summary else title
                results.append({
                    "time": ch["time"],
                    "timestamp": _fmt(ch["time"]),
                    "context": context,
                    "source": "chapter",
                })

    # Deduplicate results within 5 seconds of each other
    results.sort(key=lambda r: r["time"])
    deduped = []
    for r in results:
        if not deduped or r["time"] - deduped[-1]["time"] > 5:
            deduped.append(r)

    return deduped


def fallback_chapters(analyzed_segments: list[dict]) -> list[dict]:
    """Create simple time-based chapters if Nemotron fails to generate them."""
    chapters = []
    for seg in analyzed_segments:
        title = (seg.get("transcript_chunk") or "")[:50]
        if not title:
            title = f"Segment at {_fmt(seg['start_time'])}"
        chapters.append({
            "time": seg["start_time"],
            "title": title.strip(),
            "summary": (seg.get("visual_description") or "")[:100],
        })
    return chapters


def _parse_json_array(raw: str, fallback):
    """Extract a JSON array from model output, handling markdown fences."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = cleaned.strip().rstrip("`")

    # Try to find array bounds
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass
    return fallback

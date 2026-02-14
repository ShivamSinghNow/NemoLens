"""NemoLens — Powered by NVIDIA Nemotron."""

import re
import io
import types
import hashlib
import base64
import datetime

# ── PyTorch / Streamlit compatibility patch ──────────────────────────────────
# Streamlit's file watcher inspects every imported module's __path__._path to
# discover source files.  PyTorch's `torch.classes` uses a custom __getattr__
# that tries to look up *any* attribute as a C++ registered class, so accessing
# __path__._path raises a RuntimeError.  The patch below gives torch.classes a
# normal __path__ object so Streamlit's watcher skips it silently.
import torch.classes as _torch_classes  # noqa: E402
_torch_classes.__path__ = types.SimpleNamespace(_path=[])

import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from transcription import (
    transcribe_video,
    format_transcript_with_timestamps,
    search_transcript,
)
from visual_processor import analyze_video
from intelligence import (
    build_full_context,
    generate_chapters,
    generate_takeaways,
    generate_study_guide,
    generate_questions,
    evaluate_short_answer,
    fallback_chapters,
    search_video,
)
from nemotron_client import chat_with_context
from youtube_downloader import is_youtube_url, download_youtube_video, get_video_info
from voice_assistant import transcribe_audio, answer_question
from components.floating_mic import floating_mic

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NemoLens",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Timestamp badges (non-clickable display) ─────────────────── */
    .ts-badge {
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
        color: #76b900;
        font-weight: 600;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        background: rgba(118, 185, 0, 0.1);
        font-size: 0.85rem;
        white-space: nowrap;
    }

    /* ── Chapter cards ─────────────────────────────────────────────── */
    .chapter-card {
        padding: 0.75rem 1rem;
        border-left: 3px solid #76b900;
        margin-bottom: 0.5rem;
        background: rgba(118, 185, 0, 0.05);
        border-radius: 0 6px 6px 0;
    }
    .chapter-title {
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .chapter-summary {
        font-size: 0.9rem;
        color: #888;
    }

    /* ── Takeaways ─────────────────────────────────────────────────── */
    .takeaway-item {
        padding: 0.5rem 0.75rem;
        border-left: 2px solid #76b900;
        margin-bottom: 0.4rem;
        font-size: 0.95rem;
    }

    /* ── Search results ────────────────────────────────────────────── */
    .search-result {
        padding: 0.75rem 1rem;
        border-left: 3px solid #f0ad4e;
        margin-bottom: 0.5rem;
        background: rgba(240, 173, 78, 0.05);
        border-radius: 0 6px 6px 0;
    }
    .search-badge {
        font-size: 0.7rem;
        padding: 0.1rem 0.4rem;
        border-radius: 3px;
        background: rgba(240, 173, 78, 0.15);
        color: #f0ad4e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .search-badge-chapter {
        background: rgba(118, 185, 0, 0.15);
        color: #76b900;
    }
    .highlight {
        background: #76b900;
        color: white;
        padding: 0 3px;
        border-radius: 3px;
        font-weight: 600;
    }

    /* ── Header ────────────────────────────────────────────────────── */
    .nvidia-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(118, 185, 0, 0.3);
    }
    .nvidia-header h1 {
        color: #ffffff;
        margin-bottom: 0.25rem;
    }
    .nvidia-header p {
        color: #76b900;
        font-size: 0.95rem;
        margin: 0;
    }

    /* ── Status badges ─────────────────────────────────────────────── */
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-ready {
        background: rgba(118, 185, 0, 0.15);
        color: #76b900;
    }
    .status-processing {
        background: rgba(240, 173, 78, 0.15);
        color: #f0ad4e;
    }

    /* ── Now-playing indicator ─────────────────────────────────────── */
    .now-playing {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        background: rgba(118, 185, 0, 0.1);
        border: 1px solid rgba(118, 185, 0, 0.3);
        font-family: monospace;
        font-size: 0.85rem;
        color: #76b900;
        margin: 0.5rem 0;
    }
    .now-playing .pulse-dot {
        width: 8px;
        height: 8px;
        background: #76b900;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    /* ── NVIDIA badge ──────────────────────────────────────────────── */
    .nvidia-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        background: linear-gradient(135deg, #76b900, #5a8f00);
        color: white !important;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    /* ── Voice assistant ──────────────────────────────────────────── */
    .nemo-response {
        padding: 1rem;
        border-left: 3px solid #76b900;
        background: rgba(118, 185, 0, 0.05);
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .nemo-label {
        color: #76b900;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.25rem;
    }
    .voice-history-q {
        font-weight: 600;
        margin-bottom: 0.15rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ───────────────────────────────────────────────────

DEFAULTS = {
    "video_name": None,
    "video_bytes": None,
    "video_source": None,       # "upload" or "youtube"
    "youtube_url": None,        # Original YouTube URL for embedding
    "transcript": None,
    "analyzed_segments": None,
    "chapters": None,
    "takeaways": None,
    "study_guide": None,
    "questions": None,
    "quiz_results": {},
    "full_context": None,
    "chat_history": [],
    "voice_history": [],
    "processing_done": False,
    "video_start_time": 0,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fmt(seconds: float) -> str:
    """Format seconds as MM:SS or H:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _ts_badge(seconds: float) -> str:
    """Return an HTML badge showing a formatted timestamp (non-clickable)."""
    return f'<span class="ts-badge">{_fmt(seconds)}</span>'


def _jump_to(seconds: float):
    """on_click callback — set the video start time in session state."""
    st.session_state.video_start_time = int(seconds)


def _highlight_timestamps(text: str) -> str:
    """Highlight MM:SS and H:MM:SS patterns in text with styled badges."""
    def _repl(m):
        return f'<span class="ts-badge">{m.group(1)}</span>'
    return re.sub(r'(?<!\d)(\d{1,2}:\d{2}(?::\d{2})?)(?!\d)', _repl, text)


def _extract_timestamps(text: str) -> list[tuple[str, int]]:
    """Find all unique MM:SS / H:MM:SS timestamps in text.

    Returns list of (display_string, total_seconds) tuples.
    """
    seen: set[str] = set()
    results: list[tuple[str, int]] = []
    for m in re.finditer(r'(?<!\d)(\d{1,2}:\d{2}(?::\d{2})?)(?!\d)', text):
        ts_str = m.group(1)
        if ts_str in seen:
            continue
        seen.add(ts_str)
        parts = ts_str.split(":")
        try:
            if len(parts) == 3:
                secs = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            else:
                secs = int(parts[0]) * 60 + int(parts[1])
            results.append((ts_str, secs))
        except ValueError:
            pass
    return results


def _sanitize_text(text: str) -> str:
    """Replace common Unicode characters with latin-1 safe equivalents."""
    replacements = {
        "\u2014": "--",   # em-dash
        "\u2013": "-",    # en-dash
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2026": "...",  # ellipsis
        "\u2022": "-",    # bullet
        "\u00a0": " ",    # non-breaking space
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    # Drop any remaining non-latin-1 characters
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _build_export_pdf() -> bytes:
    """Build a comprehensive study-guide PDF with all analysis results.

    Sections: Cover, Key Takeaways, Study Guide, Chapter Timeline,
    Practice Questions, Answer Key, Full Transcript.

    Returns raw PDF bytes suitable for st.download_button.
    """

    # ── Color palette ──────────────────────────────────────────────────
    GREEN = (118, 185, 0)
    DARK_BG = (30, 30, 50)
    DARK_TEXT = (40, 40, 40)
    MID_TEXT = (80, 80, 80)
    LIGHT_TEXT = (120, 120, 120)
    ACCENT_BG = (245, 250, 240)

    class _StudyPDF(FPDF):
        """FPDF subclass with branded header/footer."""

        _skip_header = False  # suppress header on cover page

        def header(self):
            if self._skip_header:
                return
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(*GREEN)
            self.cell(0, 7, "NemoLens  |  Study Guide",
                      align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_draw_color(*GREEN)
            self.set_line_width(0.3)
            self.line(10, self.get_y(), self.w - 10, self.get_y())
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(160, 160, 160)
            self.cell(
                0, 10,
                "NemoLens  --  "
                f"Powered by NVIDIA Nemotron  |  Page {self.page_no()}/{{nb}}",
                align="C",
            )

    pdf = _StudyPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    video_name = _sanitize_text(st.session_state.video_name or "Unknown")
    analyzed_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Shared helpers ─────────────────────────────────────────────────
    def _section_heading(title: str):
        """Draw a green-accented section heading."""
        pdf.set_draw_color(*GREEN)
        pdf.set_line_width(0.6)
        pdf.line(10, pdf.get_y(), 65, pdf.get_y())
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(*DARK_TEXT)
        pdf.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    def _body_text(text: str, size: int = 10, bold: bool = False):
        style = "B" if bold else ""
        pdf.set_font("Helvetica", style, size)
        pdf.set_text_color(*MID_TEXT)
        pdf.multi_cell(0, 5.5, _sanitize_text(text))

    def _green_label(text: str):
        pdf.set_font("Courier", "B", 9)
        pdf.set_text_color(*GREEN)
        pdf.cell(22, 6, text)

    # ══════════════════════════════════════════════════════════════════
    #  PAGE 1 — COVER
    # ══════════════════════════════════════════════════════════════════
    pdf._skip_header = True
    pdf.add_page()

    # Big dark banner
    pdf.ln(40)
    pdf.set_fill_color(*DARK_BG)
    pdf.rect(0, 30, pdf.w, 80, "F")

    pdf.set_y(42)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 14, "NemoLens",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(*GREEN)
    pdf.cell(0, 10, "Comprehensive Study Guide",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Accent line
    pdf.ln(4)
    pdf.set_draw_color(*GREEN)
    pdf.set_line_width(0.8)
    line_w = 60
    pdf.line((pdf.w - line_w) / 2, pdf.get_y(),
             (pdf.w + line_w) / 2, pdf.get_y())
    pdf.ln(8)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(200, 200, 200)
    pdf.cell(0, 7, f"Video:  {video_name}",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Below the banner — metadata
    pdf.set_y(125)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*LIGHT_TEXT)
    pdf.cell(0, 6, f"Generated:  {analyzed_at}",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, "Powered by NVIDIA Nemotron Nano 12B",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Contents preview
    pdf.ln(16)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*DARK_TEXT)
    pdf.cell(0, 8, "What's Inside", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    toc_items = [
        "Key Takeaways -- The most important points at a glance",
        "Study Guide -- Detailed topic-by-topic breakdown",
        "Chapter Timeline -- Navigate the video by section",
        "Practice Questions -- Test your understanding",
        "Answer Key -- Check your answers with explanations",
        "Full Transcript -- Complete word-for-word reference",
    ]
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*MID_TEXT)
    for item in toc_items:
        pdf.cell(0, 6, f"   {item}",
                 align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf._skip_header = False  # re-enable header for remaining pages

    # ══════════════════════════════════════════════════════════════════
    #  KEY TAKEAWAYS
    # ══════════════════════════════════════════════════════════════════
    pdf.add_page()
    _section_heading("Key Takeaways")

    if st.session_state.takeaways:
        for i, tw in enumerate(st.session_state.takeaways, 1):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*GREEN)
            pdf.cell(8, 6, f"{i}.")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(*MID_TEXT)
            pdf.multi_cell(0, 6, _sanitize_text(tw))
            pdf.ln(2)
    else:
        _body_text("No takeaways were generated for this video.")

    # ══════════════════════════════════════════════════════════════════
    #  STUDY GUIDE
    # ══════════════════════════════════════════════════════════════════
    pdf.add_page()
    _section_heading("Study Guide")

    study_guide = st.session_state.study_guide or []
    if study_guide:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(*LIGHT_TEXT)
        pdf.multi_cell(0, 5,
            "A detailed breakdown of every major topic covered in the video. "
            "Use this as your primary review material.")
        pdf.ln(4)

        for idx, topic in enumerate(study_guide, 1):
            # Topic header with number + timestamp
            ts = topic.get("timestamp", 0)
            title = _sanitize_text(str(topic.get("topic", f"Topic {idx}")))

            # Green accent bar
            y_before = pdf.get_y()
            pdf.set_fill_color(*ACCENT_BG)
            pdf.rect(10, y_before, pdf.w - 20, 10, "F")
            pdf.set_draw_color(*GREEN)
            pdf.set_line_width(0.6)
            pdf.line(10, y_before, 10, y_before + 10)

            pdf.set_x(14)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(*DARK_TEXT)
            pdf.cell(0, 10, f"{idx}.  {title}",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            # Timestamp
            pdf.set_x(14)
            _green_label(f"[{_fmt(ts)}]")
            overview = _sanitize_text(str(topic.get("overview", "")))
            if overview:
                pdf.set_font("Helvetica", "I", 10)
                pdf.set_text_color(*LIGHT_TEXT)
                pdf.multi_cell(0, 5, overview)
            else:
                pdf.ln(6)

            pdf.ln(2)

            # Detailed explanation
            details = _sanitize_text(str(topic.get("details", "")))
            if details:
                pdf.set_x(14)
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(*MID_TEXT)
                pdf.multi_cell(pdf.w - 28, 5.5, details)
                pdf.ln(2)

            # Key terms
            terms = topic.get("key_terms") or []
            if terms:
                pdf.set_x(14)
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(*GREEN)
                pdf.cell(22, 5, "Key Terms:")
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(*MID_TEXT)
                terms_str = _sanitize_text("  |  ".join(str(t) for t in terms))
                pdf.multi_cell(0, 5, terms_str)

            pdf.ln(5)
    else:
        _body_text("Study guide could not be generated for this video.")

    # ══════════════════════════════════════════════════════════════════
    #  CHAPTER TIMELINE
    # ══════════════════════════════════════════════════════════════════
    pdf.add_page()
    _section_heading("Chapter Timeline")

    chapters = st.session_state.chapters or []
    if chapters:
        for ch in chapters:
            _green_label(f"[{_fmt(ch['time'])}]")
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(*DARK_TEXT)
            pdf.multi_cell(0, 7, _sanitize_text(str(ch.get("title", ""))))
            if ch.get("summary"):
                pdf.set_x(pdf.l_margin + 22)
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(*LIGHT_TEXT)
                pdf.multi_cell(0, 5, _sanitize_text(ch["summary"]))
            pdf.ln(3)
    else:
        _body_text("No chapters were generated.")

    # ══════════════════════════════════════════════════════════════════
    #  PRACTICE QUESTIONS
    # ══════════════════════════════════════════════════════════════════
    pdf.add_page()
    _section_heading("Practice Questions")

    questions = st.session_state.questions or []
    if questions:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(*LIGHT_TEXT)
        pdf.multi_cell(0, 5,
            "Test your understanding of the video. "
            "Try answering before checking the Answer Key on the next page.")
        pdf.ln(4)

        for qi, q in enumerate(questions, 1):
            qtype = str(q.get("type", "short_answer"))
            type_label = {
                "multiple_choice": "Multiple Choice",
                "true_false": "True / False",
                "short_answer": "Short Answer",
            }.get(qtype, qtype.replace("_", " ").title())

            # Question number + type badge
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(*DARK_TEXT)
            pdf.cell(10, 7, f"{qi}.")
            pdf.set_font("Courier", "", 8)
            pdf.set_text_color(*GREEN)
            pdf.cell(0, 7, f"[{type_label}]",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            # Question text
            pdf.set_x(pdf.l_margin + 10)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(*MID_TEXT)
            pdf.multi_cell(pdf.w - 30, 6,
                           _sanitize_text(str(q.get("question", ""))))

            # Options (for multiple choice)
            options = q.get("options") or []
            if options and qtype == "multiple_choice":
                pdf.ln(1)
                for opt in options:
                    pdf.set_x(pdf.l_margin + 16)
                    pdf.set_font("Helvetica", "", 10)
                    pdf.set_text_color(*MID_TEXT)
                    pdf.multi_cell(pdf.w - 36, 5.5, _sanitize_text(str(opt)))
                    pdf.ln(0.5)

            # Writing space for short answer
            if qtype == "short_answer":
                pdf.ln(1)
                pdf.set_draw_color(200, 200, 200)
                pdf.set_line_width(0.2)
                for _ in range(3):
                    y = pdf.get_y()
                    pdf.line(pdf.l_margin + 10, y, pdf.w - 10, y)
                    pdf.ln(6)

            pdf.ln(3)
    else:
        _body_text("Practice questions could not be generated.")

    # ══════════════════════════════════════════════════════════════════
    #  ANSWER KEY
    # ══════════════════════════════════════════════════════════════════
    if questions:
        pdf.add_page()
        _section_heading("Answer Key")

        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*LIGHT_TEXT)
        pdf.multi_cell(0, 5,
            "Review your answers, then read the explanations "
            "to deepen your understanding.")
        pdf.ln(4)

        for qi, q in enumerate(questions, 1):
            # Question number + answer
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*DARK_TEXT)
            pdf.cell(10, 6, f"{qi}.")
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*GREEN)
            answer_text = _sanitize_text(str(q.get("answer", "N/A")))
            pdf.multi_cell(0, 6, answer_text)

            # Explanation
            explanation = q.get("explanation", "")
            if explanation:
                pdf.set_x(pdf.l_margin + 10)
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(*LIGHT_TEXT)
                pdf.multi_cell(pdf.w - 30, 5, _sanitize_text(str(explanation)))
            pdf.ln(3)

    # ══════════════════════════════════════════════════════════════════
    #  FULL TRANSCRIPT
    # ══════════════════════════════════════════════════════════════════
    if st.session_state.transcript and st.session_state.transcript.get("segments"):
        pdf.add_page()
        _section_heading("Full Transcript")

        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*LIGHT_TEXT)
        pdf.multi_cell(0, 5, "Complete word-for-word transcript for reference.")
        pdf.ln(3)

        for seg in st.session_state.transcript["segments"]:
            pdf.set_font("Courier", "", 8)
            pdf.set_text_color(*GREEN)
            pdf.cell(18, 5, f"[{_fmt(seg['start'])}]")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(100, 100, 100)
            pdf.multi_cell(0, 5, _sanitize_text(seg["text"]))
            pdf.ln(0.5)

    # ── Write to bytes ─────────────────────────────────────────────────
    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


def reset_state():
    """Reset all session state to defaults."""
    for key, val in DEFAULTS.items():
        st.session_state[key] = val
    # Clear voice-assistant internal tracking state
    for key in ("_voice_audio_hash", "_voice_question", "_voice_answer", "_voice_response_id"):
        st.session_state.pop(key, None)


def _build_voice_context() -> dict:
    """Assemble video analysis data into a dict for the voice assistant."""
    transcript_text = ""
    if st.session_state.transcript:
        transcript_text = format_transcript_with_timestamps(
            st.session_state.transcript["segments"]
        )

    chapters_text = ""
    if st.session_state.chapters:
        chapters_text = "\n".join(
            f"[{_fmt(ch['time'])}] {ch['title']}: {ch.get('summary', '')}"
            for ch in st.session_state.chapters
        )

    takeaways_text = ""
    if st.session_state.takeaways:
        takeaways_text = "\n".join(f"- {tw}" for tw in st.session_state.takeaways)

    visual_text = ""
    if st.session_state.analyzed_segments:
        visual_text = "\n".join(
            f"[{_fmt(seg['start_time'])} - {_fmt(seg['end_time'])}] "
            f"{seg.get('visual_description', '')}"
            for seg in st.session_state.analyzed_segments
        )

    return {
        "transcript": transcript_text,
        "chapters": chapters_text,
        "takeaways": takeaways_text,
        "visual_descriptions": visual_text,
    }


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚡ NemoLens")
    st.caption("Vision-Language Model via NVIDIA Build")
    st.divider()

    if st.session_state.processing_done:
        st.markdown(
            '<span class="status-badge status-ready">✓ Analysis Complete</span>',
            unsafe_allow_html=True,
        )

        # Clickable chapters in sidebar
        if st.session_state.chapters:
            st.markdown("#### 📑 Chapters")
            for i, ch in enumerate(st.session_state.chapters):
                st.button(
                    f"▶ {_fmt(ch['time'])}  —  {ch['title']}",
                    key=f"ch_sb_{i}",
                    on_click=_jump_to,
                    args=(ch["time"],),
                    use_container_width=True,
                )
                if ch.get("summary"):
                    st.caption(ch["summary"])

        st.divider()

        # Export study guide as PDF
        pdf_bytes = _build_export_pdf()
        video_name = st.session_state.video_name or "video"
        base_name = video_name.rsplit(".", 1)[0]
        st.download_button(
            "📥 Export Study Guide (PDF)",
            pdf_bytes,
            file_name=f"{base_name}_study_guide.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="sidebar_pdf_export",
        )
    else:
        st.markdown(
            '<span class="status-badge status-processing">⏳ Waiting for video</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    whisper_model = st.selectbox(
        "Whisper model",
        ["tiny", "base", "small"],
        index=1,
        help="Larger = more accurate but slower. 'base' is a good default.",
    )

    # NVIDIA badge at bottom
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; padding:0.5rem 0;">'
        '<span class="nvidia-badge">⚡ Powered by NVIDIA Nemotron Nano 12B</span>'
        "</div>",
        unsafe_allow_html=True,
    )


# ── Header ───────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="nvidia-header">'
    "<h1>🎬 NemoLens</h1>"
    "<p>Powered by NVIDIA Nemotron Nano 12B</p>"
    "</div>",
    unsafe_allow_html=True,
)


# ── Video Input ──────────────────────────────────────────────────────────────

input_tab_upload, input_tab_youtube = st.tabs(
    ["📁 Upload Video", "🔗 YouTube URL"]
)

with input_tab_upload:
    uploaded = st.file_uploader(
        "Upload a video file",
        type=["mp4", "mov", "avi"],
        help="Supported formats: MP4, MOV, AVI",
    )

    if uploaded is not None:
        # New video — reset everything
        if st.session_state.video_name != uploaded.name:
            reset_state()
            st.session_state.video_name = uploaded.name
            st.session_state.video_bytes = uploaded.getvalue()
            st.session_state.video_source = "upload"

with input_tab_youtube:
    st.markdown(
        "Paste a YouTube URL below to download and analyze the video. "
        "Videos up to **60 minutes** are supported."
    )
    yt_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        key="yt_url_input",
    )

    if yt_url:
        if not is_youtube_url(yt_url):
            st.error("That doesn't look like a valid YouTube URL. Please check and try again.")
        else:
            # Only fetch info / download if this is a new URL
            already_loaded = (
                st.session_state.youtube_url == yt_url
                and st.session_state.video_bytes is not None
            )

            if not already_loaded:
                # Show metadata preview
                try:
                    with st.spinner("Fetching video info..."):
                        info = get_video_info(yt_url)
                    dur_min = info["duration"] // 60
                    dur_sec = info["duration"] % 60
                    st.markdown(
                        f"**{info['title']}**  \n"
                        f"👤 {info['uploader']}  •  "
                        f"⏱️ {dur_min}m {dur_sec}s"
                    )
                    if info.get("thumbnail"):
                        st.image(info["thumbnail"], width=360)
                except Exception as e:
                    st.error(f"Could not fetch video info: {e}")
                    info = None

                if info and st.button(
                    "⬇️ Download & Load Video",
                    type="primary",
                    use_container_width=True,
                    key="yt_download_btn",
                ):
                    try:
                        with st.spinner(
                            "⏳ Downloading video from YouTube (this may take a moment)..."
                        ):
                            video_bytes, title = download_youtube_video(yt_url)
                        reset_state()
                        st.session_state.video_name = f"{title}.mp4"
                        st.session_state.video_bytes = video_bytes
                        st.session_state.video_source = "youtube"
                        st.session_state.youtube_url = yt_url
                        st.success(
                            f"✅ Downloaded **{title}** "
                            f"({len(video_bytes) / (1024 * 1024):.1f} MB)"
                        )
                        st.rerun()
                    except (ValueError, RuntimeError) as e:
                        st.error(f"Download failed: {e}")
            else:
                st.success(
                    f"✅ **{st.session_state.video_name}** is loaded and ready."
                )


# ── Video Player & Analysis ──────────────────────────────────────────────────

has_video = st.session_state.video_bytes is not None

if has_video:
    # Video player with timestamp seeking support
    st.video(
        st.session_state.video_bytes,
        start_time=int(st.session_state.video_start_time),
    )

    # "Now playing from" indicator when user jumps to a timestamp
    if st.session_state.video_start_time > 0:
        st.markdown(
            f'<div class="now-playing">'
            f'<div class="pulse-dot"></div>'
            f"Playing from {_fmt(st.session_state.video_start_time)}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Processing pipeline ──────────────────────────────────────────────

    if not st.session_state.processing_done:
        if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
            video_bytes = st.session_state.video_bytes
            progress = st.progress(0, text="🚀 Starting analysis pipeline...")

            # Step 1 — Transcription
            progress.progress(5, text="🎙️ Transcribing audio with Whisper...")
            try:
                transcript = transcribe_video(video_bytes, model_name=whisper_model)
                st.session_state.transcript = transcript
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                st.stop()

            n_segs = len(transcript["segments"])
            progress.progress(
                30,
                text=(
                    f"✅ Transcription complete — {n_segs} segments "
                    f"in {transcript['language'].upper()}"
                ),
            )

            # Step 2 — Visual analysis (parallel)
            status_text = st.empty()
            status_text.info(
                "🔬 Analyzing video segments with Nemotron (parallel)..."
            )

            def _on_visual_progress(done, total):
                pct = 30 + int(50 * done / max(total, 1))
                progress.progress(
                    pct, text=f"🔬 Visual analysis: {done}/{total} segments"
                )

            try:
                analyzed = analyze_video(
                    video_bytes,
                    transcript["segments"],
                    on_progress=_on_visual_progress,
                )
                st.session_state.analyzed_segments = analyzed
            except Exception as e:
                status_text.empty()
                st.error(f"Visual analysis failed: {e}")
                st.stop()

            status_text.empty()
            progress.progress(80, text="🧠 Generating chapters and takeaways...")

            # Step 3 — Intelligence layer
            context = build_full_context(analyzed)
            st.session_state.full_context = context

            try:
                chapters = generate_chapters(analyzed)
                st.session_state.chapters = chapters
            except Exception as e:
                st.warning(
                    f"AI chapter generation failed ({e}). "
                    "Using time-based fallback chapters."
                )
                st.session_state.chapters = fallback_chapters(analyzed)

            progress.progress(85, text="💡 Extracting key takeaways...")

            try:
                takeaways = generate_takeaways(analyzed)
                st.session_state.takeaways = takeaways
            except Exception as e:
                st.warning(f"Takeaway generation failed ({e}).")
                st.session_state.takeaways = [
                    "Takeaways could not be generated -- "
                    "please use the Q&A chat to ask about the video."
                ]

            progress.progress(90, text="📖 Building study guide...")

            try:
                study_guide = generate_study_guide(analyzed)
                st.session_state.study_guide = study_guide
            except Exception as e:
                st.warning(f"Study guide generation failed ({e}).")
                st.session_state.study_guide = []

            progress.progress(95, text="❓ Generating practice questions...")

            try:
                questions = generate_questions(analyzed)
                st.session_state.questions = questions
            except Exception as e:
                st.warning(f"Question generation failed ({e}).")
                st.session_state.questions = []

            st.session_state.processing_done = True
            progress.progress(100, text="✅ Analysis complete!")
            st.rerun()

    # ── Results UI ───────────────────────────────────────────────────────

    if st.session_state.processing_done:

        # ── Analysis Tabs ─────────────────────────────────────────────────

        tab_overview, tab_study, tab_quiz, tab_transcript, tab_search, tab_chat = st.tabs(
            ["📊 Overview", "📖 Study Guide", "❓ Practice Quiz",
             "📝 Transcript", "🔍 Search", "💬 Q&A Chat"]
        )

        # ── Overview tab ─────────────────────────────────────────────────

        with tab_overview:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 💡 Key Takeaways")
                if st.session_state.takeaways:
                    for tw in st.session_state.takeaways:
                        st.markdown(
                            f'<div class="takeaway-item">{tw}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No takeaways generated.")

            with col2:
                st.markdown("### 📑 Chapters")
                if st.session_state.chapters:
                    for i, ch in enumerate(st.session_state.chapters):
                        c_ts, c_info = st.columns([1.4, 5])
                        with c_ts:
                            st.button(
                                f"▶ {_fmt(ch['time'])}",
                                key=f"ch_ov_{i}",
                                on_click=_jump_to,
                                args=(ch["time"],),
                                use_container_width=True,
                            )
                        with c_info:
                            st.markdown(f"**{ch['title']}**")
                            if ch.get("summary"):
                                st.caption(ch["summary"])
                else:
                    st.info("No chapters generated.")

            # Segment details with jump-to buttons
            st.markdown("### 🎞️ Segment Analysis")
            for i, seg in enumerate(st.session_state.analyzed_segments):
                label = (
                    f"Segment {i + 1}: "
                    f"{_fmt(seg['start_time'])} – {_fmt(seg['end_time'])}"
                )
                with st.expander(label):
                    sc1, sc2, sc3 = st.columns([1, 1, 4])
                    with sc1:
                        st.button(
                            f"▶ {_fmt(seg['start_time'])}",
                            key=f"seg_s_{i}",
                            on_click=_jump_to,
                            args=(seg["start_time"],),
                            use_container_width=True,
                        )
                    with sc2:
                        st.button(
                            f"▶ {_fmt(seg['end_time'])}",
                            key=f"seg_e_{i}",
                            on_click=_jump_to,
                            args=(seg["end_time"],),
                            use_container_width=True,
                        )
                    if seg.get("transcript_chunk"):
                        st.markdown("**Transcript:**")
                        st.text(seg["transcript_chunk"])
                    if seg.get("visual_description"):
                        st.markdown("**Visual Description:**")
                        st.write(seg["visual_description"])

        # ── Study Guide tab ────────────────────────────────────────────────

        with tab_study:
            st.markdown("### 📖 Study Guide")
            st.caption(
                "A detailed breakdown of every major topic covered in this video."
            )

            study_guide = st.session_state.study_guide or []
            if study_guide:
                for idx, topic in enumerate(study_guide, 1):
                    ts = topic.get("timestamp", 0)
                    title = topic.get("topic", f"Topic {idx}")
                    overview = topic.get("overview", "")
                    details = topic.get("details", "")
                    terms = topic.get("key_terms") or []

                    with st.expander(f"**{idx}. {title}**  —  {_fmt(ts)}", expanded=True):
                        # Jump button
                        st.button(
                            f"▶ Jump to {_fmt(ts)}",
                            key=f"sg_jump_{idx}",
                            on_click=_jump_to,
                            args=(ts,),
                        )
                        if overview:
                            st.markdown(f"*{overview}*")
                        if details:
                            st.markdown(details)
                        if terms:
                            term_str = " · ".join(f"**{t}**" for t in terms)
                            st.markdown(f"🏷️ Key Terms: {term_str}")
            else:
                st.info("No study guide was generated for this video.")

        # ── Practice Quiz tab ──────────────────────────────────────────────

        with tab_quiz:
            st.markdown("### ❓ Practice Quiz")
            st.caption(
                "Select or type your answers, then submit each one to see "
                "if you got it right. Short answers are evaluated by AI!"
            )

            questions = st.session_state.questions or []
            quiz_results = st.session_state.quiz_results

            if questions:
                # ── Score bar ──────────────────────────────────────────
                total_q = len(questions)
                answered = len(quiz_results)
                correct = sum(
                    1 for r in quiz_results.values() if r.get("correct")
                )
                if answered:
                    sc1, sc2, sc3 = st.columns(3)
                    with sc1:
                        st.metric("Answered", f"{answered}/{total_q}")
                    with sc2:
                        st.metric("Correct", f"{correct}/{answered}")
                    with sc3:
                        pct = int(100 * correct / answered)
                        st.metric("Score", f"{pct}%")
                    st.progress(answered / total_q)

                # ── Reset button ───────────────────────────────────────
                if answered and st.button("🔄 Reset Quiz", key="quiz_reset"):
                    st.session_state.quiz_results = {}
                    st.rerun()

                st.divider()

                # ── Questions ──────────────────────────────────────────
                for qi, q in enumerate(questions, 1):
                    qtype = str(q.get("type", "short_answer"))
                    type_labels = {
                        "multiple_choice": "Multiple Choice",
                        "true_false": "True / False",
                        "short_answer": "Short Answer",
                    }
                    badge = type_labels.get(
                        qtype, qtype.replace("_", " ").title()
                    )
                    qtext = str(q.get("question", ""))
                    correct_answer = str(q.get("answer", ""))
                    explanation = str(q.get("explanation", ""))
                    result_key = f"q_{qi}"
                    already_answered = result_key in quiz_results

                    # Header icon
                    if already_answered:
                        icon = "✅" if quiz_results[result_key]["correct"] else "❌"
                    else:
                        icon = f"Q{qi}"

                    with st.expander(
                        f"{icon}  {qtext}",
                        expanded=not already_answered,
                    ):
                        st.caption(badge)

                        # ── Multiple Choice ────────────────────────────
                        if qtype == "multiple_choice":
                            options = q.get("options") or []
                            if not options:
                                options = ["A", "B", "C", "D"]

                            selection = st.radio(
                                "Select your answer:",
                                options,
                                index=None,
                                key=f"mc_{qi}",
                                disabled=already_answered,
                            )

                            if not already_answered:
                                if st.button(
                                    "Submit Answer",
                                    key=f"submit_{qi}",
                                ):
                                    if selection is None:
                                        st.warning(
                                            "Please select an option first."
                                        )
                                    else:
                                        # Compare first letter of selection
                                        # with first letter of correct answer
                                        sel_letter = selection.strip()[0].upper()
                                        ans_letter = correct_answer.strip()[0].upper()
                                        is_correct = sel_letter == ans_letter
                                        quiz_results[result_key] = {
                                            "correct": is_correct,
                                            "user_answer": selection,
                                            "feedback": explanation,
                                        }
                                        st.rerun()

                        # ── True / False ───────────────────────────────
                        elif qtype == "true_false":
                            selection = st.radio(
                                "Select:",
                                ["True", "False"],
                                index=None,
                                key=f"tf_{qi}",
                                disabled=already_answered,
                            )

                            if not already_answered:
                                if st.button(
                                    "Submit Answer",
                                    key=f"submit_{qi}",
                                ):
                                    if selection is None:
                                        st.warning(
                                            "Please select True or False."
                                        )
                                    else:
                                        is_correct = (
                                            selection.lower()
                                            == correct_answer.strip().lower()
                                        )
                                        quiz_results[result_key] = {
                                            "correct": is_correct,
                                            "user_answer": selection,
                                            "feedback": explanation,
                                        }
                                        st.rerun()

                        # ── Short Answer ───────────────────────────────
                        elif qtype == "short_answer":
                            user_text = st.text_area(
                                "Type your answer:",
                                key=f"sa_{qi}",
                                height=100,
                                disabled=already_answered,
                            )

                            if not already_answered:
                                if st.button(
                                    "Submit Answer",
                                    key=f"submit_{qi}",
                                ):
                                    if not user_text or not user_text.strip():
                                        st.warning(
                                            "Please type an answer first."
                                        )
                                    else:
                                        with st.spinner(
                                            "🧠 AI is evaluating your answer..."
                                        ):
                                            eval_result = evaluate_short_answer(
                                                qtext,
                                                correct_answer,
                                                user_text.strip(),
                                            )
                                        quiz_results[result_key] = {
                                            "correct": eval_result["correct"],
                                            "user_answer": user_text.strip(),
                                            "feedback": eval_result["feedback"],
                                        }
                                        st.rerun()

                        # ── Show result after submission ───────────────
                        if already_answered:
                            result = quiz_results[result_key]
                            if result["correct"]:
                                st.success("✅ Correct!")
                            else:
                                st.error("❌ Incorrect")
                            feedback = result.get("feedback", "")
                            if feedback:
                                st.info(f"**Explanation:** {feedback}")
                            # Show the model answer for reference
                            if not result["correct"]:
                                st.caption(
                                    f"Correct answer: **{correct_answer}**"
                                )
            else:
                st.info("No practice questions were generated for this video.")

        # ── Transcript tab ───────────────────────────────────────────────

        with tab_transcript:
            st.markdown("### 📝 Full Transcript")

            transcript_data = st.session_state.transcript
            if transcript_data and transcript_data.get("segments"):
                segments = transcript_data["segments"]

                # Search / filter box
                transcript_filter = st.text_input(
                    "🔍 Filter transcript",
                    placeholder="Type to filter transcript lines...",
                    key="transcript_filter",
                )

                # Filter segments
                if transcript_filter:
                    filtered = [
                        s
                        for s in segments
                        if transcript_filter.lower() in s["text"].lower()
                    ]
                    st.caption(
                        f"Showing {len(filtered)} of {len(segments)} lines "
                        f'matching "{transcript_filter}"'
                    )
                else:
                    filtered = segments

                if not filtered:
                    st.warning("No matching lines found.")
                else:
                    # Render each transcript line with a jump button
                    for idx, seg in enumerate(filtered):
                        tc1, tc2 = st.columns([1, 7])
                        with tc1:
                            st.button(
                                f"▶ {_fmt(seg['start'])}",
                                key=f"tr_{int(seg['start'] * 100)}",
                                on_click=_jump_to,
                                args=(seg["start"],),
                                use_container_width=True,
                            )
                        with tc2:
                            text = seg["text"]
                            if transcript_filter:
                                pattern = re.compile(
                                    re.escape(transcript_filter),
                                    re.IGNORECASE,
                                )
                                text = pattern.sub(
                                    lambda m: f'<span class="highlight">'
                                    f"{m.group()}</span>",
                                    text,
                                )
                                st.markdown(text, unsafe_allow_html=True)
                            else:
                                st.write(text)

                # Download buttons
                st.divider()
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    formatted = format_transcript_with_timestamps(segments)
                    st.download_button(
                        "📥 Download Transcript (.txt)",
                        formatted,
                        file_name="transcript.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                with col_dl2:
                    pdf_bytes = _build_export_pdf()
                    vname = st.session_state.video_name or "video"
                    st.download_button(
                        "📥 Export Study Guide (PDF)",
                        pdf_bytes,
                        file_name=f"{vname.rsplit('.', 1)[0]}_study_guide.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="transcript_tab_pdf_export",
                    )
            else:
                st.info("No transcript available.")

        # ── Search tab ───────────────────────────────────────────────────

        with tab_search:
            st.markdown("### 🔍 Search Video Content")
            st.caption(
                "Search through transcript text, visual descriptions, and "
                "chapter titles. Click any timestamp to jump!"
            )

            search_query = st.text_input(
                "Find where they discuss...",
                placeholder='e.g. "machine learning", "revenue growth", "diagram"',
                key="search_query",
            )

            if search_query:
                results = search_video(
                    search_query,
                    st.session_state.analyzed_segments,
                    st.session_state.transcript["segments"],
                    chapters=st.session_state.chapters,
                )

                if results:
                    st.markdown(f"**{len(results)} result(s) found:**")
                    for ri, r in enumerate(results):
                        source = r["source"]
                        badge_cls = (
                            "search-badge search-badge-chapter"
                            if source == "chapter"
                            else "search-badge"
                        )
                        context = r["context"]
                        pattern = re.compile(
                            re.escape(search_query), re.IGNORECASE
                        )
                        context_hl = pattern.sub(
                            lambda m: f'<span class="highlight">'
                            f"{m.group()}</span>",
                            context,
                        )

                        sr1, sr2 = st.columns([1.4, 6])
                        with sr1:
                            st.button(
                                f"▶ {r['timestamp']}",
                                key=f"sr_{ri}",
                                on_click=_jump_to,
                                args=(r["time"],),
                                use_container_width=True,
                            )
                        with sr2:
                            st.markdown(
                                f'<span class="{badge_cls}">{source}</span> '
                                f"{context_hl}",
                                unsafe_allow_html=True,
                            )
                else:
                    st.warning(f'No results found for "{search_query}".')

        # ── Q&A Chat tab ─────────────────────────────────────────────────

        with tab_chat:
            st.markdown("### 💬 Ask About This Video")
            st.caption(
                "Ask any question about the video content. "
                "The AI will reference specific timestamps — "
                "click the buttons to jump to that moment!"
            )

            # Display chat history with jump buttons for timestamps
            for msg_idx, msg in enumerate(st.session_state.chat_history):
                with st.chat_message(msg["role"]):
                    if msg["role"] == "assistant":
                        st.markdown(
                            _highlight_timestamps(msg["content"]),
                            unsafe_allow_html=True,
                        )
                        # Extract timestamps and show jump buttons
                        ts_list = _extract_timestamps(msg["content"])
                        if ts_list:
                            ts_cols = st.columns(min(len(ts_list), 6))
                            for j, (ts_str, secs) in enumerate(ts_list):
                                with ts_cols[j % len(ts_cols)]:
                                    st.button(
                                        f"▶ {ts_str}",
                                        key=f"chat_{msg_idx}_{j}",
                                        on_click=_jump_to,
                                        args=(secs,),
                                        use_container_width=True,
                                    )
                    else:
                        st.markdown(msg["content"])

            # Chat input
            if question := st.chat_input("Ask anything about the video..."):
                with st.chat_message("user"):
                    st.markdown(question)
                st.session_state.chat_history.append(
                    {"role": "user", "content": question}
                )

                with st.chat_message("assistant"):
                    with st.spinner("🧠 Thinking..."):
                        try:
                            answer = chat_with_context(
                                st.session_state.full_context,
                                question,
                                chat_history=st.session_state.chat_history[:-1],
                            )
                        except Exception as e:
                            answer = f"Error: {e}"
                    st.markdown(
                        _highlight_timestamps(answer),
                        unsafe_allow_html=True,
                    )
                    # Show jump buttons for timestamps in the answer
                    ts_list = _extract_timestamps(answer)
                    if ts_list:
                        ts_cols = st.columns(min(len(ts_list), 6))
                        for j, (ts_str, secs) in enumerate(ts_list):
                            with ts_cols[j % len(ts_cols)]:
                                st.button(
                                    f"▶ {ts_str}",
                                    key=f"chat_new_{j}",
                                    on_click=_jump_to,
                                    args=(secs,),
                                    use_container_width=True,
                                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )

else:
    st.info(
        "👆 Upload a video file or paste a YouTube URL above to get started."
    )


# ── Floating Voice Assistant (Nemo) ──────────────────────────────────────────
# A draggable mic FAB that hovers over the page.  Tap to record a question;
# the answer appears in a floating panel with TTS.

if st.session_state.processing_done:
    audio_b64 = floating_mic(
        response=st.session_state.get("_voice_answer", ""),
        question=st.session_state.get("_voice_question", ""),
        response_id=st.session_state.get("_voice_response_id", ""),
        key="nemo_floating_mic",
    )

    # Process new audio when the component sends recorded data
    if audio_b64:
        audio_hash = hashlib.md5(
            audio_b64.encode() if isinstance(audio_b64, str) else audio_b64
        ).hexdigest()

        if st.session_state.get("_voice_audio_hash") != audio_hash:
            st.session_state["_voice_audio_hash"] = audio_hash

            try:
                audio_bytes = base64.b64decode(audio_b64)
                question_text = transcribe_audio(
                    audio_bytes, model_name=whisper_model
                )
            except Exception as e:
                question_text = ""
                st.toast(f"Transcription failed: {e}", icon="⚠️")

            if question_text.strip():
                st.session_state["_voice_question"] = question_text
                try:
                    ctx = _build_voice_context()
                    answer_text = answer_question(question_text, ctx)
                except Exception as e:
                    answer_text = (
                        "Sorry, I couldn't process that right now. "
                        f"Error: {e}"
                    )
                st.session_state["_voice_answer"] = answer_text
                st.session_state["_voice_response_id"] = hashlib.md5(
                    answer_text.encode()
                ).hexdigest()
                st.session_state.voice_history.append(
                    {"question": question_text, "answer": answer_text}
                )
            else:
                st.session_state["_voice_question"] = ""
                st.session_state["_voice_answer"] = ""
                st.toast(
                    "Couldn't catch that — please try recording again.",
                    icon="🔇",
                )

            st.rerun()

# NemoLens

**AI-Powered Video Analysis & Study Companion** built with NVIDIA Nemotron Nano 12B

NemoLens transforms any video into a structured learning experience. Upload a video or paste a YouTube URL and NemoLens will transcribe the audio, analyze the visuals frame-by-frame, generate chapters, key takeaways, a full study guide, practice quizzes, and let you ask questions with a floating voice assistant — all powered by NVIDIA's Nemotron vision-language model.

---

## Features

- **Multi-Modal Analysis** — Combines audio transcription (Whisper) with visual frame analysis (Nemotron Vision) for a complete understanding of video content
- **Auto-Generated Chapters** — AI creates timestamped chapters you can click to jump to
- **Key Takeaways** — Extracts the 5-8 most important points from the video
- **Study Guide** — Detailed topic-by-topic breakdown exported as a downloadable PDF
- **Practice Quiz** — Multiple choice, true/false, and short-answer questions with AI grading
- **Full-Text Search** — Search across transcript, visual descriptions, and chapters simultaneously
- **Q&A Chat** — Context-aware chat interface grounded in the video's content
- **Voice Assistant ("Nemo")** — Draggable floating microphone for hands-free questions with browser text-to-speech responses
- **YouTube Integration** — Paste a YouTube URL and NemoLens downloads and processes it automatically
- **Timestamp Navigation** — Click any timestamp to jump directly to that moment in the video

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit, Custom HTML/JS Components |
| AI – Vision & Language | NVIDIA Nemotron Nano 12B (via NVIDIA Build API) |
| AI – Transcription | OpenAI Whisper (runs locally) |
| Video Processing | OpenCV, FFmpeg |
| YouTube Downloads | yt-dlp |
| PDF Export | fpdf2 |
| Voice I/O | Browser MediaRecorder API, Web Speech API |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        app.py (Streamlit UI)                 │
│  Tabs: Overview │ Study Guide │ Quiz │ Transcript │ Search │ Chat │
└────────┬──────────────────────┬──────────────────────┬───────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐
│ transcription.py │  │ visual_processor │  │ voice_assistant.py │
│ (Whisper local)  │  │ (OpenCV frames)  │  │ (Mic → Whisper →  │
│                  │  │                  │  │  Nemotron → TTS)  │
└────────┬─────────┘  └────────┬─────────┘  └─────────┬─────────┘
         │                     │                       │
         ▼                     ▼                       ▼
┌──────────────────────────────────────────────────────────────┐
│                    nemotron_client.py                         │
│       NVIDIA Build API wrapper (vision + text completions)   │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│                     intelligence.py                           │
│  Chapters • Takeaways • Study Guide • Questions • Search     │
└──────────────────────────────────────────────────────────────┘
```

**Data Flow:**
1. **Input** — User uploads a video file (MP4/MOV/AVI) or pastes a YouTube URL
2. **Download** — `youtube_downloader.py` handles YouTube URLs via yt-dlp (720p max, 60-min limit)
3. **Transcription** — `transcription.py` runs Whisper locally to produce timestamped text
4. **Visual Analysis** — `visual_processor.py` splits the video into 2-minute segments, extracts 3 frames per segment (384px, JPEG q60), and sends them to Nemotron Vision in parallel (3 workers)
5. **Intelligence** — `intelligence.py` combines transcript + visual descriptions and prompts Nemotron for chapters, takeaways, study guides, and quiz questions
6. **Presentation** — `app.py` renders everything in a tabbed Streamlit interface with clickable timestamps and PDF export

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **FFmpeg** — Required for audio extraction and YouTube downloads
- **NVIDIA API Key** — Free tier available at [build.nvidia.com](https://build.nvidia.com)

### 1. Clone the repository

```bash
git clone https://github.com/ShivamSinghNow/NemoLens.git
cd NemoLens
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg

# Windows — download from https://ffmpeg.org/download.html and add to PATH
```

### 5. Set up your environment variables

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your NVIDIA API key:

```
NVIDIA_API_KEY=nvapi-your_actual_key_here
```

You can get a free API key by signing up at [build.nvidia.com](https://build.nvidia.com) and navigating to the Nemotron model page.

### 6. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Project Structure

```
NemoLens/
├── app.py                  # Main Streamlit application & UI
├── intelligence.py         # AI generation (chapters, takeaways, study guide, quiz)
├── nemotron_client.py      # NVIDIA Build API wrapper with retry logic
├── visual_processor.py     # Frame extraction & Nemotron Vision analysis
├── transcription.py        # Whisper-based audio transcription
├── youtube_downloader.py   # YouTube URL validation & download (yt-dlp)
├── voice_assistant.py      # Voice Q&A pipeline (speech → text → AI → TTS)
├── components/
│   ├── tts_component.py    # Browser text-to-speech helper
│   └── floating_mic/       # Custom Streamlit component
│       └── frontend/
│           └── index.html  # Draggable mic button with audio recording
├── requirements.txt
├── .env.example
├── LICENSE                 # MIT
└── README.md
```

---

## Architecture Decisions & Why I Made Them

### Local Whisper vs. Cloud Transcription APIs
I chose to run OpenAI's Whisper model locally rather than calling a cloud transcription API. This eliminates per-minute API costs and keeps audio data on the user's machine. The trade-off is longer processing time on machines without a GPU, but Whisper's `base` model provides a solid balance of speed and accuracy for most content.

### NVIDIA Nemotron Nano 12B as the Core LLM
Nemotron Nano 12B was selected because it supports both vision (image) and text modalities through a single model endpoint on the NVIDIA Build API. This means the same model that reads video frames also generates study guides and answers questions — keeping the architecture simple and the context coherent. The free tier on build.nvidia.com makes it accessible for anyone to try.

### 2-Minute Segments with 3 Frames Each
Videos are split into 2-minute segments with 3 evenly-spaced frames extracted per segment. This was a deliberate balance: too few frames and you miss visual context, too many and you hit API payload limits and rate caps. Three frames per segment captures the beginning, middle, and end of each chunk without overwhelming the vision model.

### Parallel Visual Processing (3 Workers)
Visual analysis runs segments through a thread pool with 3 concurrent workers. This significantly reduces wall-clock time for longer videos while respecting the NVIDIA API's rate limits. The worker count was tuned through testing to avoid 429 (rate limit) errors.

### Custom Streamlit Component for Voice
The floating microphone is a fully custom Streamlit component built with raw HTML/JS rather than a third-party library. I built it from scratch because no existing Streamlit component offered a draggable, always-visible mic button with integrated audio recording and TTS playback. It uses the browser's `MediaRecorder` API for capture and `SpeechSynthesis` for reading answers aloud.

### Graceful Fallbacks Throughout
Every major pipeline stage has a fallback path:
- If visual analysis fails for a segment, it falls back to transcript-only descriptions
- If chapter generation fails, time-based fallback chapters are generated automatically
- If a video has no audio track, the app detects this before attempting transcription
- API rate limits (429) trigger exponential backoff and automatic retries

### PDF Export with fpdf2
I used fpdf2 over alternatives like ReportLab or WeasyPrint because it's lightweight, pure Python, and doesn't require system-level dependencies. It generates clean, readable study guide PDFs without adding complexity to the install process.

---

## Challenges I Faced

### Rate Limiting & API Reliability
The NVIDIA Build API enforces rate limits that become a real constraint when processing longer videos with many segments. I implemented exponential backoff with jitter, retry logic, and capped the parallel workers at 3 — but tuning this balance between speed and reliability took significant iteration.

### Frame Size & API Payload Limits
Sending high-resolution frames to the vision API quickly exceeds payload size limits. I had to find the sweet spot: 384px max dimension and JPEG quality 60 keeps frame data small enough while preserving enough visual detail for the model to describe what's happening on screen.

### Whisper Model Selection
Larger Whisper models produce better transcripts but take dramatically longer on CPU. The app defaults to the `base` model as a practical compromise, but users on GPU-equipped machines can switch to `small` or `medium` for better accuracy — especially for technical content or accented speech.

### YouTube Download Reliability
YouTube frequently changes its internals, which can break downloaders. Using yt-dlp (actively maintained) over the legacy youtube-dl mitigates this, but I still had to add robust error handling around the download pipeline, including duration validation, format selection fallbacks, and proper temp-file cleanup.

### Keeping Voice Assistant Responsive
The voice assistant chains multiple operations: audio recording in the browser, upload to the server, Whisper transcription, Nemotron inference, and TTS playback. Each step adds latency. I optimized by keeping Whisper's model cached in memory and instructing Nemotron to produce concise, spoken-friendly answers to minimize the round-trip time.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

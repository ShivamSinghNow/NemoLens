"""Browser text-to-speech via the Web Speech API."""

import json

import streamlit.components.v1 as components


def speak_text(text: str, rate: float = 1.0, pitch: float = 1.0):
    """Speak *text* aloud using the browser's speechSynthesis API.

    Renders a zero-height HTML snippet that triggers TTS on load.
    """
    safe_text = json.dumps(text)

    components.html(f"""
        <script>
            window.speechSynthesis.cancel();

            const text = {safe_text};
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = {rate};
            utterance.pitch = {pitch};

            function speak() {{
                const voices = window.speechSynthesis.getVoices();
                const preferred = voices.find(v =>
                    v.name.includes('Samantha') ||
                    v.name.includes('Google US English') ||
                    v.name.includes('Natural') ||
                    (v.lang.startsWith('en') && v.localService)
                );
                if (preferred) utterance.voice = preferred;
                window.speechSynthesis.speak(utterance);
            }}

            if (window.speechSynthesis.getVoices().length > 0) {{
                speak();
            }} else {{
                window.speechSynthesis.onvoiceschanged = speak;
            }}
        </script>
    """, height=0)

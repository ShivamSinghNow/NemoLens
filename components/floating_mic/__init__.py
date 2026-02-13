"""Floating draggable microphone button — Streamlit custom component."""

import os
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
_component_func = components.declare_component("floating_mic", path=_COMPONENT_DIR)


def floating_mic(response="", question="", response_id="", key=None):
    """Render a floating draggable microphone FAB for voice Q&A.

    Args:
        response: The latest answer text to display in the panel.
        question: The latest question text to display in the panel.
        response_id: Unique ID for the response (triggers TTS on change).
        key: Streamlit widget key.

    Returns:
        Base64-encoded audio string when user records, or None.
    """
    return _component_func(
        response=response or "",
        question=question or "",
        _response_id=response_id or "",
        key=key,
        default=None,
    )

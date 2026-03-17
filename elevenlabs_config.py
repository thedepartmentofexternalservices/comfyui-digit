"""Shared ElevenLabs configuration for DIGIT nodes.

Resolution order for API key:
  1. Node input (if user typed something)
  2. DIGIT_ELEVENLABS_API_KEY env var
  3. ELEVENLABS_API_KEY env var
  4. Error
"""

import os

ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"


def resolve_api_key(node_value=""):
    """Resolve ElevenLabs API key from node input → env vars."""
    key = node_value.strip() if node_value else ""
    if key:
        return key

    key = (
        os.environ.get("DIGIT_ELEVENLABS_API_KEY")
        or os.environ.get("ELEVENLABS_API_KEY")
        or ""
    )
    if key:
        return key

    raise ValueError(
        "ElevenLabs API key is required. Set it in the node, or set the "
        "DIGIT_ELEVENLABS_API_KEY or ELEVENLABS_API_KEY environment variable."
    )


def default_api_key():
    """Return the default API key for node UI fields."""
    return (
        os.environ.get("DIGIT_ELEVENLABS_API_KEY")
        or os.environ.get("ELEVENLABS_API_KEY")
        or ""
    )

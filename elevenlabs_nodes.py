"""DIGIT ElevenLabs nodes — direct API access via env-var API key.

Replaces the built-in ComfyUI ElevenLabs nodes that require ComfyUI org
proxy auth.  These nodes call the ElevenLabs API directly using an API key
resolved from DIGIT_ELEVENLABS_API_KEY or ELEVENLABS_API_KEY.
"""

import io
import json
import logging
import struct
import uuid

import numpy as np
import requests
import torch

from .elevenlabs_config import ELEVENLABS_API_BASE, default_api_key, resolve_api_key

logger = logging.getLogger(__name__)

# ── Predefined voices ────────────────────────────────────────────────────────
# (voice_id, display_name, gender, accent)
ELEVENLABS_VOICES = [
    ("CwhRBWXzGAHq8TQ4Fs17", "Roger", "male", "american"),
    ("EXAVITQu4vr4xnSDxMaL", "Sarah", "female", "american"),
    ("FGY2WhTYpPnrIDTdsKH5", "Laura", "female", "american"),
    ("IKne3meq5aSn9XLyUdCD", "Charlie", "male", "australian"),
    ("JBFqnCBsd6RMkjVDRZzb", "George", "male", "british"),
    ("N2lVS1w4EtoT3dr4eOWO", "Callum", "male", "american"),
    ("SAz9YHcvj6GT2YYXdXww", "River", "neutral", "american"),
    ("SOYHLrjzK2X1ezoPC6cr", "Harry", "male", "american"),
    ("TX3LPaxmHKxFdv7VOQHJ", "Liam", "male", "american"),
    ("Xb7hH8MSUJpSbSDYk0k2", "Alice", "female", "british"),
    ("XrExE9yKIg1WjnnlVkGX", "Matilda", "female", "american"),
    ("bIHbv24MWmeRgasZH58o", "Will", "male", "american"),
    ("cgSgspJ2msm6clMCkdW9", "Jessica", "female", "american"),
    ("cjVigY5qzO86Huf0OWal", "Eric", "male", "american"),
    ("hpp4J3VqNfWAUOO0d1Us", "Bella", "female", "american"),
    ("iP95p4xoKVk53GoZ742B", "Chris", "male", "american"),
    ("nPczCjzI2devNBz1zQrb", "Brian", "male", "american"),
    ("onwK4e9ZLuTAKqWW03F9", "Daniel", "male", "british"),
    ("pFZP5JQG7iQjIQuC4Bku", "Lily", "female", "british"),
    ("pNInz6obpgDQGcFmaJgB", "Adam", "male", "american"),
    ("pqHfZKP75CvOlQylNhV4", "Bill", "male", "american"),
]

VOICE_OPTIONS = [f"{name} ({gender}, {accent})" for _, name, gender, accent in ELEVENLABS_VOICES]
VOICE_MAP = {f"{name} ({gender}, {accent})": vid for vid, name, gender, accent in ELEVENLABS_VOICES}


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _audio_tensor_to_wav_bytes(waveform, sample_rate):
    """Convert ComfyUI audio waveform tensor to WAV bytes."""
    # waveform shape: (channels, samples) or (batch, channels, samples)
    if waveform.dim() == 3:
        waveform = waveform[0]
    audio_np = waveform.cpu().numpy()
    # Mix to mono if multi-channel
    if audio_np.shape[0] > 1:
        audio_np = audio_np.mean(axis=0, keepdims=True)
    audio_np = audio_np[0]  # (samples,)
    # Convert to 16-bit PCM
    audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    # Write WAV header
    num_samples = len(audio_int16)
    data_size = num_samples * 2  # 16-bit = 2 bytes per sample
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(audio_int16.tobytes())
    return buf.getvalue()


def _pcm_bytes_to_audio_tensor(pcm_bytes, sample_rate=44100):
    """Convert raw PCM 16-bit signed LE bytes to ComfyUI AUDIO dict."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.from_numpy(samples).unsqueeze(0)  # (1, num_samples)
    return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}


def _headers(api_key):
    return {"xi-api-key": api_key}


# ── Voice Selector ────────────────────────────────────────────────────────────

class DigitElevenLabsVoiceSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voice": (VOICE_OPTIONS, {"tooltip": "Choose a predefined ElevenLabs voice."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("voice_id",)
    FUNCTION = "select"
    CATEGORY = "DIGIT/ElevenLabs"

    def select(self, voice):
        voice_id = VOICE_MAP.get(voice)
        if not voice_id:
            raise ValueError(f"Unknown voice: {voice}")
        return (voice_id,)


# ── Text to Speech ────────────────────────────────────────────────────────────

class DigitElevenLabsTTS:
    MODELS = ["eleven_multilingual_v2", "eleven_v3"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True, "tooltip": "Text to convert to speech."}),
                "voice_id": ("STRING", {"default": "", "tooltip": "Voice ID. Connect from Voice Selector or paste directly."}),
                "model": (cls.MODELS, {"default": "eleven_multilingual_v2"}),
                "stability": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "similarity_boost": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.7, "max": 1.3, "step": 0.01}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 2147483647}),
            },
            "optional": {
                "api_key": ("STRING", {"default": default_api_key(), "tooltip": "ElevenLabs API key. Auto-detected from ELEVENLABS_API_KEY env var."}),
                "language_code": ("STRING", {"default": "", "tooltip": "ISO-639-1/3 language code. Leave empty for auto-detect."}),
                "style": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.2, "step": 0.01}),
                "use_speaker_boost": ("BOOLEAN", {"default": False}),
                "apply_text_normalization": (["auto", "on", "off"], {"default": "auto"}),
                "output_format": (["pcm_44100", "mp3_44100_192", "opus_48000_192"], {"default": "pcm_44100"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "DIGIT/ElevenLabs"

    def generate(self, text, voice_id, model, stability, similarity_boost, speed, seed,
                 api_key="", language_code="", style=0.0, use_speaker_boost=False,
                 apply_text_normalization="auto", output_format="pcm_44100"):
        if not text.strip():
            raise ValueError("Text is required")
        if not voice_id.strip():
            raise ValueError("Voice ID is required. Connect a Voice Selector or enter an ID.")

        key = resolve_api_key(api_key)
        url = f"{ELEVENLABS_API_BASE}/text-to-speech/{voice_id}"
        params = {"output_format": output_format}

        body = {
            "text": text,
            "model_id": model,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
                "speed": speed,
                "style": style,
                "use_speaker_boost": use_speaker_boost,
            },
            "seed": seed,
            "apply_text_normalization": apply_text_normalization,
        }
        if language_code.strip():
            body["language_code"] = language_code.strip()

        resp = requests.post(url, headers=_headers(key), json=body, params=params, timeout=300)
        resp.raise_for_status()
        return (_pcm_bytes_to_audio_tensor(resp.content),)


# ── Speech to Text ────────────────────────────────────────────────────────────

class DigitElevenLabsSTT:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model": (["scribe_v2"], {"default": "scribe_v2"}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 2147483647}),
            },
            "optional": {
                "api_key": ("STRING", {"default": default_api_key(), "tooltip": "ElevenLabs API key. Auto-detected from ELEVENLABS_API_KEY env var."}),
                "language_code": ("STRING", {"default": "", "tooltip": "ISO-639-1/3 language code. Leave empty for auto-detect."}),
                "tag_audio_events": ("BOOLEAN", {"default": False, "tooltip": "Annotate sounds like (laughter) in transcript."}),
                "diarize": ("BOOLEAN", {"default": False, "tooltip": "Annotate which speaker is talking."}),
                "diarization_threshold": ("FLOAT", {"default": 0.22, "min": 0.1, "max": 0.4, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "timestamps_granularity": (["word", "character", "none"], {"default": "word"}),
                "num_speakers": ("INT", {"default": 0, "min": 0, "max": 32}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "language_code", "words_json")
    FUNCTION = "transcribe"
    CATEGORY = "DIGIT/ElevenLabs"

    def transcribe(self, audio, model, seed, api_key="", language_code="",
                   tag_audio_events=False, diarize=False, diarization_threshold=0.22,
                   temperature=0.0, timestamps_granularity="word", num_speakers=0):
        key = resolve_api_key(api_key)

        if diarize and num_speakers > 0:
            raise ValueError(
                "Cannot specify num_speakers when diarization is enabled. "
                "Set num_speakers to 0 or disable diarize."
            )

        # Convert audio tensor to WAV bytes
        wav_bytes = _audio_tensor_to_wav_bytes(audio["waveform"], audio["sample_rate"])

        url = f"{ELEVENLABS_API_BASE}/speech-to-text"

        # Build multipart form data
        data = {
            "model_id": model,
            "tag_audio_events": str(tag_audio_events).lower(),
            "timestamps_granularity": timestamps_granularity,
            "seed": str(seed),
        }
        if language_code.strip():
            data["language_code"] = language_code.strip()
        if num_speakers > 0:
            data["num_speakers"] = str(num_speakers)
        if diarize:
            data["diarize"] = "true"
            data["diarization_threshold"] = str(diarization_threshold)
        if temperature > 0:
            data["temperature"] = str(temperature)

        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}

        resp = requests.post(url, headers=_headers(key), data=data, files=files, timeout=300)
        resp.raise_for_status()
        result = resp.json()

        text = result.get("text", "")
        lang = result.get("language_code", "")
        words = result.get("words", [])
        words_json = json.dumps(words, indent=2) if words else "[]"

        return (text, lang, words_json)


# ── Text to Sound Effects ─────────────────────────────────────────────────────

class DigitElevenLabsSFX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True, "tooltip": "Describe the sound effect to generate."}),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.5, "max": 30.0, "step": 0.1}),
                "prompt_influence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "api_key": ("STRING", {"default": default_api_key(), "tooltip": "ElevenLabs API key. Auto-detected from ELEVENLABS_API_KEY env var."}),
                "loop": ("BOOLEAN", {"default": False, "tooltip": "Create a smoothly looping sound effect."}),
                "output_format": (["pcm_44100", "mp3_44100_192", "opus_48000_192"], {"default": "pcm_44100"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "DIGIT/ElevenLabs"

    def generate(self, text, duration, prompt_influence, api_key="", loop=False,
                 output_format="pcm_44100"):
        if not text.strip():
            raise ValueError("Text description is required")

        key = resolve_api_key(api_key)
        url = f"{ELEVENLABS_API_BASE}/sound-generation"
        params = {"output_format": output_format}

        body = {
            "text": text,
            "duration_seconds": duration,
            "prompt_influence": prompt_influence,
        }
        if loop:
            body["loop"] = True

        resp = requests.post(url, headers=_headers(key), json=body, params=params, timeout=300)
        resp.raise_for_status()
        return (_pcm_bytes_to_audio_tensor(resp.content),)


# ── Voice Isolation ───────────────────────────────────────────────────────────

class DigitElevenLabsVoiceIsolation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "api_key": ("STRING", {"default": default_api_key(), "tooltip": "ElevenLabs API key. Auto-detected from ELEVENLABS_API_KEY env var."}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "isolate"
    CATEGORY = "DIGIT/ElevenLabs"

    def isolate(self, audio, api_key=""):
        key = resolve_api_key(api_key)
        wav_bytes = _audio_tensor_to_wav_bytes(audio["waveform"], audio["sample_rate"])

        url = f"{ELEVENLABS_API_BASE}/audio-isolation"
        files = {"audio": ("audio.wav", wav_bytes, "audio/wav")}
        params = {"output_format": "pcm_44100"}

        resp = requests.post(url, headers=_headers(key), files=files, params=params, timeout=300)
        resp.raise_for_status()
        return (_pcm_bytes_to_audio_tensor(resp.content),)


# ── Instant Voice Clone ───────────────────────────────────────────────────────

class DigitElevenLabsVoiceClone:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio1": ("AUDIO",),
                "remove_background_noise": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "api_key": ("STRING", {"default": default_api_key(), "tooltip": "ElevenLabs API key. Auto-detected from ELEVENLABS_API_KEY env var."}),
                "voice_name": ("STRING", {"default": "", "tooltip": "Name for the cloned voice. Auto-generated if empty."}),
                "audio2": ("AUDIO",),
                "audio3": ("AUDIO",),
                "audio4": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("voice_id",)
    FUNCTION = "clone"
    CATEGORY = "DIGIT/ElevenLabs"

    def clone(self, audio1, remove_background_noise, api_key="", voice_name="",
              audio2=None, audio3=None, audio4=None):
        key = resolve_api_key(api_key)
        name = voice_name.strip() if voice_name.strip() else str(uuid.uuid4())

        url = f"{ELEVENLABS_API_BASE}/voices/add"

        files = []
        for i, audio in enumerate([audio1, audio2, audio3, audio4]):
            if audio is not None:
                wav_bytes = _audio_tensor_to_wav_bytes(audio["waveform"], audio["sample_rate"])
                files.append(("files", (f"sample_{i+1}.wav", wav_bytes, "audio/wav")))

        data = {
            "name": name,
            "remove_background_noise": str(remove_background_noise).lower(),
        }

        resp = requests.post(url, headers=_headers(key), data=data, files=files, timeout=300)
        resp.raise_for_status()
        result = resp.json()
        return (result["voice_id"],)


# ── Speech to Speech ──────────────────────────────────────────────────────────

class DigitElevenLabsSTS:
    MODELS = ["eleven_multilingual_sts_v2", "eleven_english_sts_v2"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "voice_id": ("STRING", {"default": "", "tooltip": "Target voice ID. Connect from Voice Selector or paste directly."}),
                "model": (cls.MODELS, {"default": "eleven_multilingual_sts_v2"}),
                "stability": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "similarity_boost": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295}),
            },
            "optional": {
                "api_key": ("STRING", {"default": default_api_key(), "tooltip": "ElevenLabs API key. Auto-detected from ELEVENLABS_API_KEY env var."}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.7, "max": 1.3, "step": 0.01}),
                "style": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.2, "step": 0.01}),
                "use_speaker_boost": ("BOOLEAN", {"default": False}),
                "remove_background_noise": ("BOOLEAN", {"default": False}),
                "output_format": (["pcm_44100", "mp3_44100_192", "opus_48000_192"], {"default": "pcm_44100"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "transform"
    CATEGORY = "DIGIT/ElevenLabs"

    def transform(self, audio, voice_id, model, stability, similarity_boost, seed,
                  api_key="", speed=1.0, style=0.0, use_speaker_boost=False,
                  remove_background_noise=False, output_format="pcm_44100"):
        if not voice_id.strip():
            raise ValueError("Voice ID is required.")

        key = resolve_api_key(api_key)
        wav_bytes = _audio_tensor_to_wav_bytes(audio["waveform"], audio["sample_rate"])

        url = f"{ELEVENLABS_API_BASE}/speech-to-speech/{voice_id}"
        params = {"output_format": output_format}

        voice_settings = json.dumps({
            "stability": stability,
            "similarity_boost": similarity_boost,
            "speed": speed,
            "style": style,
            "use_speaker_boost": use_speaker_boost,
        })

        data = {
            "model_id": model,
            "voice_settings": voice_settings,
            "seed": str(seed),
            "remove_background_noise": str(remove_background_noise).lower(),
        }
        files = {"audio": ("audio.wav", wav_bytes, "audio/wav")}

        resp = requests.post(url, headers=_headers(key), data=data, files=files,
                             params=params, timeout=300)
        resp.raise_for_status()
        return (_pcm_bytes_to_audio_tensor(resp.content),)


# ── Text to Dialogue ──────────────────────────────────────────────────────────

class DigitElevenLabsDialogue:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text1": ("STRING", {"default": "", "multiline": True, "tooltip": "Dialogue line 1."}),
                "voice_id1": ("STRING", {"default": "", "tooltip": "Voice ID for line 1."}),
                "num_entries": ("INT", {"default": 2, "min": 1, "max": 10, "tooltip": "Number of dialogue entries to use."}),
                "model": (["eleven_v3"], {"default": "eleven_v3"}),
                "stability": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 4294967295}),
            },
            "optional": {
                "api_key": ("STRING", {"default": default_api_key(), "tooltip": "ElevenLabs API key. Auto-detected from ELEVENLABS_API_KEY env var."}),
                "text2": ("STRING", {"default": "", "multiline": True}),
                "voice_id2": ("STRING", {"default": ""}),
                "text3": ("STRING", {"default": "", "multiline": True}),
                "voice_id3": ("STRING", {"default": ""}),
                "text4": ("STRING", {"default": "", "multiline": True}),
                "voice_id4": ("STRING", {"default": ""}),
                "text5": ("STRING", {"default": "", "multiline": True}),
                "voice_id5": ("STRING", {"default": ""}),
                "text6": ("STRING", {"default": "", "multiline": True}),
                "voice_id6": ("STRING", {"default": ""}),
                "text7": ("STRING", {"default": "", "multiline": True}),
                "voice_id7": ("STRING", {"default": ""}),
                "text8": ("STRING", {"default": "", "multiline": True}),
                "voice_id8": ("STRING", {"default": ""}),
                "text9": ("STRING", {"default": "", "multiline": True}),
                "voice_id9": ("STRING", {"default": ""}),
                "text10": ("STRING", {"default": "", "multiline": True}),
                "voice_id10": ("STRING", {"default": ""}),
                "language_code": ("STRING", {"default": ""}),
                "apply_text_normalization": (["auto", "on", "off"], {"default": "auto"}),
                "output_format": (["pcm_44100", "mp3_44100_192", "opus_48000_192"], {"default": "pcm_44100"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "DIGIT/ElevenLabs"

    def generate(self, text1, voice_id1, num_entries, model, stability, seed,
                 api_key="", language_code="", apply_text_normalization="auto",
                 output_format="pcm_44100", **kwargs):
        key = resolve_api_key(api_key)

        # Build dialogue inputs from numbered text/voice pairs
        inputs = []
        all_texts = {1: text1}
        all_voices = {1: voice_id1}
        for i in range(2, 11):
            all_texts[i] = kwargs.get(f"text{i}", "")
            all_voices[i] = kwargs.get(f"voice_id{i}", "")

        for i in range(1, num_entries + 1):
            t = all_texts.get(i, "").strip()
            v = all_voices.get(i, "").strip()
            if not t:
                raise ValueError(f"Text for dialogue entry {i} is required.")
            if not v:
                raise ValueError(f"Voice ID for dialogue entry {i} is required.")
            inputs.append({"text": t, "voice_id": v})

        url = f"{ELEVENLABS_API_BASE}/text-to-dialogue"
        params = {"output_format": output_format}

        body = {
            "inputs": inputs,
            "model_id": model,
            "settings": {"stability": stability},
            "seed": seed,
            "apply_text_normalization": apply_text_normalization,
        }
        if language_code.strip():
            body["language_code"] = language_code.strip()

        resp = requests.post(url, headers=_headers(key), json=body, params=params, timeout=300)
        resp.raise_for_status()
        return (_pcm_bytes_to_audio_tensor(resp.content),)

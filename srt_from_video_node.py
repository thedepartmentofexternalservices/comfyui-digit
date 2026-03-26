"""DIGIT SRT From Video — transcribe video files to SRT using Gemini.

Features:
- Single-file and batch transcription (recursive folder scan)
- Multi-format output: SRT, VTT, ASS/SSA, plain text
- Burn-in with full styling (font, size, color, outline, shadow, position)
- Frame-accurate padding and snap-to-frame
- Line-length enforcement (broadcast/Netflix standards)
- Language selection and translation
- Hallucination / repeat detection
"""

import logging
import math
import os
import re
import subprocess
import tempfile
import time

import comfy.utils

from .gcp_config import resolve_gcp_config, default_project, default_region
from .projekts_utils import PROJEKTS_ROOTS, scan_projects

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".qt", ".m4v", ".mkv", ".avi", ".mxf"}

GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
]

SUBTITLE_OUTPUT_MODES = ["srt_only", "burn_in_only", "both"]
OUTPUT_FORMATS = ["srt", "vtt", "ass", "txt", "all"]
LANGUAGES = [
    "auto", "en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh",
    "ar", "hi", "ru", "nl", "pl", "sv", "da", "no", "fi", "tr",
    "th", "vi", "id", "ms", "tl", "uk", "cs", "ro", "hu", "el",
    "he",
]
BURN_IN_ALIGNMENTS = [
    "bottom_center", "bottom_left", "bottom_right",
    "top_center", "top_left", "top_right",
    "middle_center",
]
# ASS alignment values: 1=BL, 2=BC, 3=BR, 5=TL, 6=TC, 7=TR, 8=MC (approx)
_ALIGNMENT_MAP = {
    "bottom_center": 2, "bottom_left": 1, "bottom_right": 3,
    "top_center": 6, "top_left": 5, "top_right": 7,
    "middle_center": 10,
}

SRT_TIMESTAMP_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)


# ── Timestamp helpers ────────────────────────────────────────────────────────

def _srt_ts_to_seconds(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _seconds_to_srt_ts(seconds):
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _seconds_to_vtt_ts(seconds):
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _seconds_to_ass_ts(seconds):
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# ── SRT parsing ──────────────────────────────────────────────────────────────

def _parse_srt(srt_text):
    """Parse SRT text into a list of (index, start_sec, end_sec, text) tuples."""
    entries = []
    blocks = re.split(r"\n\n+", srt_text.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        ts_match = SRT_TIMESTAMP_RE.search(lines[1])
        if not ts_match:
            continue
        start = _srt_ts_to_seconds(
            ts_match.group(1), ts_match.group(2),
            ts_match.group(3), ts_match.group(4))
        end = _srt_ts_to_seconds(
            ts_match.group(5), ts_match.group(6),
            ts_match.group(7), ts_match.group(8))
        text = "\n".join(lines[2:])
        try:
            idx = int(lines[0].strip())
        except ValueError:
            idx = 0
        entries.append((idx, start, end, text))
    return entries


def _entries_to_srt(entries):
    """Convert parsed entries back to SRT text."""
    parts = []
    for i, (_, start, end, text) in enumerate(entries, 1):
        parts.append(f"{i}\n{_seconds_to_srt_ts(start)} --> {_seconds_to_srt_ts(end)}\n{text}")
    return "\n\n".join(parts) + "\n"


# ── Post-processing ──────────────────────────────────────────────────────────

def _pad_srt(srt_text, pad_frames, frame_rate):
    """Pad each subtitle entry by pad_frames on both sides."""
    if pad_frames <= 0 or frame_rate <= 0:
        return srt_text
    pad_seconds = pad_frames / frame_rate
    entries = _parse_srt(srt_text)
    padded = []
    for idx, start, end, text in entries:
        padded.append((idx, max(0, start - pad_seconds), end + pad_seconds, text))
    return _entries_to_srt(padded)


def _snap_to_frames(srt_text, frame_rate):
    """Snap all SRT timestamps to nearest frame boundary."""
    if frame_rate <= 0:
        return srt_text
    frame_dur = 1.0 / frame_rate
    entries = _parse_srt(srt_text)
    snapped = []
    for idx, start, end, text in entries:
        start = round(start / frame_dur) * frame_dur
        end = round(end / frame_dur) * frame_dur
        if end <= start:
            end = start + frame_dur
        snapped.append((idx, start, end, text))
    return _entries_to_srt(snapped)


def _enforce_line_length(srt_text, max_chars_per_line=42, max_lines=2):
    """Split overlong subtitle lines at word boundaries."""
    entries = _parse_srt(srt_text)
    fixed = []
    for idx, start, end, text in entries:
        # Process each subtitle's text
        raw_lines = text.split("\n")
        new_lines = []
        for line in raw_lines:
            if len(line) <= max_chars_per_line:
                new_lines.append(line)
            else:
                # Word-wrap the line
                words = line.split()
                current = ""
                for word in words:
                    test = f"{current} {word}".strip()
                    if len(test) <= max_chars_per_line:
                        current = test
                    else:
                        if current:
                            new_lines.append(current)
                        current = word
                if current:
                    new_lines.append(current)

        # Enforce max_lines by splitting into multiple entries if needed
        if len(new_lines) <= max_lines:
            fixed.append((idx, start, end, "\n".join(new_lines)))
        else:
            # Split into chunks of max_lines, distribute time proportionally
            chunks = [new_lines[i:i + max_lines] for i in range(0, len(new_lines), max_lines)]
            total_chars = sum(len(line) for line in new_lines)
            if total_chars == 0:
                total_chars = 1
            duration = end - start
            t = start
            for chunk in chunks:
                chunk_chars = sum(len(line) for line in chunk)
                chunk_dur = duration * (chunk_chars / total_chars)
                if chunk_dur < 0.5:
                    chunk_dur = 0.5
                fixed.append((idx, t, t + chunk_dur, "\n".join(chunk)))
                t += chunk_dur

    return _entries_to_srt(fixed)


def _detect_hallucinations(srt_text):
    """Detect and remove repeated/hallucinated subtitle entries.

    Returns (cleaned_srt, removed_count, warnings).
    """
    entries = _parse_srt(srt_text)
    if not entries:
        return srt_text, 0, []

    cleaned = []
    removed = 0
    warnings = []
    prev_text = None
    repeat_count = 0

    for idx, start, end, text in entries:
        normalized = re.sub(r"\s+", " ", text.strip().lower())

        # Check for exact repeats
        if normalized == prev_text and normalized:
            repeat_count += 1
            if repeat_count >= 2:
                removed += 1
                warnings.append(f"Removed repeat #{repeat_count} at {_seconds_to_srt_ts(start)}: {text[:50]}")
                continue
        else:
            repeat_count = 0

        # Check for suspiciously short entries with generic content
        if len(normalized) < 3 and end - start > 5:
            removed += 1
            warnings.append(f"Removed suspicious entry at {_seconds_to_srt_ts(start)}: '{text.strip()}'")
            continue

        cleaned.append((idx, start, end, text))
        prev_text = normalized

    return _entries_to_srt(cleaned), removed, warnings


# ── Format conversion ────────────────────────────────────────────────────────

def _srt_to_vtt(srt_text):
    """Convert SRT to WebVTT format."""
    entries = _parse_srt(srt_text)
    lines = ["WEBVTT", ""]
    for i, (_, start, end, text) in enumerate(entries, 1):
        lines.append(str(i))
        lines.append(f"{_seconds_to_vtt_ts(start)} --> {_seconds_to_vtt_ts(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _srt_to_ass(srt_text, font_name="Arial", font_size=20, primary_color="&H00FFFFFF",
                outline_color="&H00000000", back_color="&H80000000",
                outline_width=2, shadow_depth=1, alignment=2, margin_v=30):
    """Convert SRT to ASS/SSA format with styling."""
    entries = _parse_srt(srt_text)

    header = f"""[Script Info]
Title: DIGIT Auto-Generated Subtitles
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary_color},&H000000FF,{outline_color},{back_color},0,0,0,0,100,100,0,0,1,{outline_width},{shadow_depth},{alignment},20,20,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header.strip()]
    for _, start, end, text in entries:
        # ASS uses \N for line breaks
        ass_text = text.replace("\n", "\\N")
        lines.append(
            f"Dialogue: 0,{_seconds_to_ass_ts(start)},{_seconds_to_ass_ts(end)},"
            f"Default,,0,0,0,,{ass_text}"
        )
    return "\n".join(lines) + "\n"


def _srt_to_txt(srt_text):
    """Convert SRT to plain text (just the spoken words)."""
    entries = _parse_srt(srt_text)
    return "\n".join(text for _, _, _, text in entries) + "\n"


def _color_name_to_ass(color_name):
    """Convert common color names to ASS &HAABBGGRR format."""
    colors = {
        "white": "&H00FFFFFF",
        "black": "&H00000000",
        "red": "&H000000FF",
        "green": "&H0000FF00",
        "blue": "&H00FF0000",
        "yellow": "&H0000FFFF",
        "cyan": "&H00FFFF00",
        "magenta": "&H00FF00FF",
        "orange": "&H000080FF",
        "gray": "&H00808080",
        "dark_gray": "&H00404040",
    }
    return colors.get(color_name, color_name)


# ── Audio extraction ─────────────────────────────────────────────────────────

def _extract_audio(video_path):
    """Extract audio from video to a temp WAV file using ffmpeg."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        tmp.name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        os.unlink(tmp.name)
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr.strip()}")
    return tmp.name


def _clean_srt(text):
    """Strip markdown code fences that Gemini sometimes wraps around output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines).strip()
    return text


# ── Transcription ────────────────────────────────────────────────────────────

def _transcribe_single(video_path, client, model, system_prompt, user_prompt):
    """Extract audio from a video file and transcribe it via Gemini."""
    from google.genai import types

    audio_path = _extract_audio(video_path)
    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
    finally:
        os.unlink(audio_path)

    audio_size_mb = len(audio_bytes) / (1024 * 1024)
    print(f"[DigitSRT] Audio: {audio_size_mb:.1f} MB — sending to {model}...")

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.1,
    )

    audio_part = types.Part.from_bytes(
        data=audio_bytes,
        mime_type="audio/wav",
    )

    response = client.models.generate_content(
        model=model,
        contents=[audio_part, user_prompt],
        config=config,
    )

    srt_text = _clean_srt(response.text)
    if not srt_text:
        raise ValueError("Gemini returned empty transcription.")
    return srt_text


# ── Burn-in ──────────────────────────────────────────────────────────────────

def _burn_in_subtitles(video_path, srt_or_ass_path, output_path,
                       font_name="Arial", font_size=24,
                       primary_color="white", outline_color="black",
                       outline_width=2, shadow_depth=1,
                       alignment="bottom_center", margin_v=30):
    """Burn subtitles into the video using ffmpeg.

    Uses ASS file directly if provided, otherwise applies force_style to SRT.
    """
    escaped_path = srt_or_ass_path.replace("'", r"'\''").replace(":", r"\:")

    if srt_or_ass_path.endswith(".ass"):
        # ASS already has styling baked in
        vf = f"ass='{escaped_path}'"
    else:
        # Build force_style for SRT
        ass_align = _ALIGNMENT_MAP.get(alignment, 2)
        ass_primary = _color_name_to_ass(primary_color)
        ass_outline = _color_name_to_ass(outline_color)
        style = (
            f"FontName={font_name},"
            f"FontSize={font_size},"
            f"PrimaryColour={ass_primary},"
            f"OutlineColour={ass_outline},"
            f"Outline={outline_width},"
            f"Shadow={shadow_depth},"
            f"Alignment={ass_align},"
            f"MarginV={margin_v}"
        )
        vf = f"subtitles='{escaped_path}':force_style='{style}'"

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", vf,
        "-c:a", "copy",
        output_path,
    ]
    print(f"[DigitSRT] Burning subtitles into: {os.path.basename(output_path)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg burn-in failed: {result.stderr.strip()}")
    return output_path


# ── Post-process pipeline ────────────────────────────────────────────────────

def _post_process_srt(srt_text, pad_frames=0, frame_rate=23.976,
                      snap_frames=False, max_chars_per_line=42,
                      max_lines=2, remove_hallucinations=True):
    """Run all post-processing steps on SRT text.

    Returns (processed_srt, warnings_list).
    """
    warnings = []

    # 1. Hallucination detection
    if remove_hallucinations:
        srt_text, removed, hal_warnings = _detect_hallucinations(srt_text)
        if removed:
            warnings.extend(hal_warnings)
            warnings.append(f"Removed {removed} hallucinated/repeated entries")

    # 2. Line-length enforcement
    if max_chars_per_line > 0:
        srt_text = _enforce_line_length(srt_text, max_chars_per_line, max_lines)

    # 3. Frame padding
    if pad_frames > 0:
        srt_text = _pad_srt(srt_text, pad_frames, frame_rate)

    # 4. Snap to frame boundaries
    if snap_frames and frame_rate > 0:
        srt_text = _snap_to_frames(srt_text, frame_rate)

    return srt_text, warnings


def _save_formats(srt_text, base_path, output_format,
                  font_name="Arial", font_size=20,
                  primary_color="white", outline_color="black",
                  outline_width=2, shadow_depth=1,
                  alignment="bottom_center", margin_v=30):
    """Save SRT text in requested format(s). Returns list of saved paths."""
    saved = []
    base_dir = os.path.dirname(base_path)
    base_name = os.path.splitext(os.path.basename(base_path))[0]

    formats_to_write = [output_format] if output_format != "all" else ["srt", "vtt", "ass", "txt"]

    ass_primary = _color_name_to_ass(primary_color)
    ass_outline = _color_name_to_ass(outline_color)
    ass_align = _ALIGNMENT_MAP.get(alignment, 2)

    for fmt in formats_to_write:
        path = os.path.join(base_dir, f"{base_name}.{fmt}")
        if fmt == "srt":
            content = srt_text
        elif fmt == "vtt":
            content = _srt_to_vtt(srt_text)
        elif fmt == "ass":
            content = _srt_to_ass(
                srt_text, font_name=font_name, font_size=font_size,
                primary_color=ass_primary, outline_color=ass_outline,
                outline_width=outline_width, shadow_depth=shadow_depth,
                alignment=ass_align, margin_v=margin_v)
        elif fmt == "txt":
            content = _srt_to_txt(srt_text)
        else:
            continue

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        saved.append(path)

    return saved


# ── Translation ──────────────────────────────────────────────────────────────

def _translate_srt(srt_text, client, model, target_language):
    """Translate SRT subtitle text to target language, preserving timestamps."""
    from google.genai import types

    prompt = f"""Translate the following SRT subtitle file into {target_language}.

Rules:
- Keep ALL SRT formatting intact: entry numbers, timestamps, blank lines.
- Only translate the subtitle text lines — do NOT modify timestamps or numbering.
- Preserve line breaks within subtitle entries.
- Output ONLY the translated SRT content. No code fences, no explanations.

SRT content:
{srt_text}"""

    config = types.GenerateContentConfig(temperature=0.2)
    response = client.models.generate_content(
        model=model, contents=prompt, config=config)
    translated = _clean_srt(response.text)
    return translated if translated else srt_text


# ── System prompt ────────────────────────────────────────────────────────────

TRANSCRIBE_SYSTEM_PROMPT = """You are a precise audio transcription engine.

Given an audio file extracted from a video, you must:
1. Transcribe ALL spoken words exactly as heard.
2. Provide accurate timestamps for each subtitle entry.
3. Output valid SRT subtitle format.

Rules for timing:
- Timestamps must reflect the ACTUAL timing of speech in the audio.
- Listen carefully and place timestamps where words are actually spoken.
- Leave a small gap between subtitle entries when there is a natural pause.

Rules for text:
- Transcribe exactly what is spoken — do not paraphrase or summarize.
- Keep each subtitle entry to a maximum of 2 lines and ~42 characters per line for readability.
- Split long sentences into multiple subtitle entries at natural phrase boundaries.
- If you can identify different speakers, prefix lines with SPEAKER 1:, SPEAKER 2:, etc.

Output ONLY the raw SRT content. No markdown code fences, no explanations, no preamble.
Start with subtitle number 1.

Example output format:
1
00:00:01,200 --> 00:00:03,800
Hey, how are you doing today?

2
00:00:04,100 --> 00:00:06,500
I'm great, thanks for asking.

"""


# ── Shared input definitions ─────────────────────────────────────────────────

def _styling_inputs():
    """Return the common burn-in styling inputs."""
    return {
        "font_name": ("STRING", {
            "default": "Arial",
            "tooltip": "Font family for burn-in subtitles.",
        }),
        "font_size": ("INT", {
            "default": 24, "min": 8, "max": 120, "step": 1,
            "tooltip": "Font size for burn-in subtitles.",
        }),
        "font_color": ([
            "white", "yellow", "cyan", "green", "red",
            "orange", "magenta", "blue", "black", "gray",
        ], {
            "default": "white",
            "tooltip": "Subtitle text color.",
        }),
        "outline_color": ([
            "black", "dark_gray", "gray", "white", "red",
            "blue", "green", "yellow",
        ], {
            "default": "black",
            "tooltip": "Subtitle outline/border color.",
        }),
        "outline_width": ("INT", {
            "default": 2, "min": 0, "max": 8, "step": 1,
            "tooltip": "Outline thickness around subtitle text.",
        }),
        "shadow_depth": ("INT", {
            "default": 1, "min": 0, "max": 8, "step": 1,
            "tooltip": "Shadow depth behind subtitle text.",
        }),
        "position": (BURN_IN_ALIGNMENTS, {
            "default": "bottom_center",
            "tooltip": "Where to place subtitles on screen.",
        }),
        "margin_v": ("INT", {
            "default": 30, "min": 0, "max": 200, "step": 5,
            "tooltip": "Vertical margin from screen edge (pixels at 1080p).",
        }),
    }


def _processing_inputs():
    """Return the common post-processing inputs."""
    return {
        "pad_frames": ("INT", {
            "default": 0, "min": 0, "max": 120, "step": 1,
            "tooltip": "Extend each subtitle by this many frames on both sides (head and tail).",
        }),
        "frame_rate": ("FLOAT", {
            "default": 23.976, "min": 1.0, "max": 120.0, "step": 0.001,
            "tooltip": "Frame rate of the video. Used for pad_frames and snap-to-frame.",
        }),
        "snap_to_frames": ("BOOLEAN", {
            "default": False,
            "tooltip": "Snap all timestamps to nearest frame boundary. Prevents subtitle flicker on frame-accurate systems.",
        }),
        "max_chars_per_line": ("INT", {
            "default": 42, "min": 0, "max": 80, "step": 1,
            "tooltip": "Max characters per subtitle line. 42 = Netflix/broadcast standard. 0 = no enforcement.",
        }),
        "max_lines": ("INT", {
            "default": 2, "min": 1, "max": 4, "step": 1,
            "tooltip": "Max lines per subtitle entry. Entries exceeding this get split.",
        }),
        "remove_hallucinations": ("BOOLEAN", {
            "default": True,
            "tooltip": "Detect and remove repeated/hallucinated subtitle entries.",
        }),
        "output_format": (OUTPUT_FORMATS, {
            "default": "srt",
            "tooltip": "Output format(s). 'all' saves SRT + VTT + ASS + TXT.",
        }),
        "language": (LANGUAGES, {
            "default": "auto",
            "tooltip": "Language of the audio. 'auto' lets Gemini detect. Improves accuracy when specified.",
        }),
        "translate_to": (["none"] + [l for l in LANGUAGES if l != "auto"], {
            "default": "none",
            "tooltip": "Translate subtitles to this language after transcription. 'none' = no translation.",
        }),
    }


# ── Single file node ─────────────────────────────────────────────────────────

class DigitSRTFromVideo:
    MODELS = GEMINI_MODELS
    CATEGORY = "DIGIT"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("srt_filepath", "srt_text")
    FUNCTION = "transcribe_video"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        available_roots = [r for r in PROJEKTS_ROOTS if os.path.isdir(r)]
        if not available_roots:
            available_roots = PROJEKTS_ROOTS

        first_root = available_roots[0]
        projects = scan_projects(first_root)

        optional = {
            "identify_speakers": ("BOOLEAN", {
                "default": True,
                "tooltip": "Try to identify and label different speakers.",
            }),
        }
        optional.update(_processing_inputs())
        optional.update(_styling_inputs())

        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to MP4 / MOV file",
                }),
                "model": (cls.MODELS, {"default": "gemini-2.5-flash"}),
                "subtitle_output": (SUBTITLE_OUTPUT_MODES, {
                    "default": "srt_only",
                    "tooltip": "srt_only: sidecar file(s). burn_in_only: hardcode subs into video. both: file(s) + burned-in video.",
                }),
                "extra_instructions": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Optional: additional transcription instructions",
                }),
                "projekts_root": (available_roots,),
                "project": (projects,),
                "filename": ("STRING", {"default": "transcription"}),
                "gcp_project_id": ("STRING", {
                    "default": default_project(),
                    "tooltip": "GCP project ID.",
                }),
                "gcp_region": ("STRING", {
                    "default": default_region(),
                    "tooltip": "GCP region.",
                }),
            },
            "optional": optional,
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def transcribe_video(self, video_path, model, subtitle_output, extra_instructions,
                         projekts_root, project, filename,
                         gcp_project_id="", gcp_region="global",
                         identify_speakers=True,
                         pad_frames=0, frame_rate=23.976, snap_to_frames=False,
                         max_chars_per_line=42, max_lines=2,
                         remove_hallucinations=True,
                         output_format="srt", language="auto", translate_to="none",
                         font_name="Arial", font_size=24,
                         font_color="white", outline_color="black",
                         outline_width=2, shadow_depth=1,
                         position="bottom_center", margin_v=30):

        from google import genai

        video_path = video_path.strip()
        if not video_path:
            raise ValueError("[DigitSRTFromVideo] video_path is required.")
        if not os.path.isfile(video_path):
            raise ValueError(f"[DigitSRTFromVideo] File not found: {video_path}")

        print(f"[DigitSRTFromVideo] Extracting audio from: {video_path}")

        gcp_project, gcp_reg = resolve_gcp_config(gcp_project_id, gcp_region)
        client = genai.Client(vertexai=True, project=gcp_project, location=gcp_reg)

        # Build system prompt
        system_prompt = TRANSCRIBE_SYSTEM_PROMPT
        if not identify_speakers:
            system_prompt += "\nDo NOT label or identify speakers. Just transcribe the words.\n"
        if language != "auto":
            system_prompt += f"\nThe audio is in {language}. Transcribe in that language.\n"

        user_prompt = "Transcribe this audio into SRT subtitle format with accurate timestamps."
        if extra_instructions and extra_instructions.strip():
            user_prompt += f"\n\nAdditional instructions:\n{extra_instructions.strip()}"

        srt_text = _transcribe_single(video_path, client, model, system_prompt, user_prompt)

        # Post-process
        srt_text, warnings = _post_process_srt(
            srt_text, pad_frames=pad_frames, frame_rate=frame_rate,
            snap_frames=snap_to_frames, max_chars_per_line=max_chars_per_line,
            max_lines=max_lines, remove_hallucinations=remove_hallucinations)

        for w in warnings:
            print(f"[DigitSRTFromVideo] {w}")

        # Translation
        if translate_to != "none":
            print(f"[DigitSRTFromVideo] Translating to {translate_to}...")
            srt_text = _translate_srt(srt_text, client, model, translate_to)

        # Save
        target_dir = os.path.join(projekts_root, project, "assets", "auto_srt")
        os.makedirs(target_dir, exist_ok=True)
        base_path = os.path.join(target_dir, f"{filename}.srt")

        if subtitle_output in ("srt_only", "both"):
            saved = _save_formats(
                srt_text, base_path, output_format,
                font_name=font_name, font_size=font_size,
                primary_color=font_color, outline_color=outline_color,
                outline_width=outline_width, shadow_depth=shadow_depth,
                alignment=position, margin_v=margin_v)
            entry_count = len(_parse_srt(srt_text))
            print(f"[DigitSRTFromVideo] Saved {entry_count} entries: {', '.join(os.path.basename(p) for p in saved)}")

        # Burn in
        if subtitle_output in ("burn_in_only", "both"):
            # Use ASS for burn-in if we have it, otherwise temp SRT
            ass_path = os.path.join(target_dir, f"{filename}.ass")
            if os.path.isfile(ass_path):
                burn_file = ass_path
                tmp_burn = None
            elif subtitle_output == "both" and output_format in ("ass", "all"):
                burn_file = ass_path
                tmp_burn = None
            else:
                # Write temp file for burn-in
                tmp_burn = tempfile.NamedTemporaryFile(suffix=".srt", delete=False, mode="w")
                tmp_burn.write(srt_text)
                tmp_burn.close()
                burn_file = tmp_burn.name

            try:
                ext = os.path.splitext(video_path)[1]
                burned_path = os.path.join(target_dir, f"{filename}_subtitled{ext}")
                _burn_in_subtitles(
                    video_path, burn_file, burned_path,
                    font_name=font_name, font_size=font_size,
                    primary_color=font_color, outline_color=outline_color,
                    outline_width=outline_width, shadow_depth=shadow_depth,
                    alignment=position, margin_v=margin_v)
                print(f"[DigitSRTFromVideo] Burned-in video: {burned_path}")
            finally:
                if tmp_burn and os.path.isfile(tmp_burn.name):
                    os.unlink(tmp_burn.name)

        return {
            "ui": {"filepath_text": [base_path]},
            "result": (base_path, srt_text),
        }


# ── Batch node ───────────────────────────────────────────────────────────────

class DigitBatchSRTFromVideo:
    """Batch transcribe a folder of video files to SRT using Gemini."""

    MODELS = GEMINI_MODELS
    CATEGORY = "DIGIT"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("log", "transcribed_count", "output_folder")
    FUNCTION = "batch_transcribe"
    OUTPUT_NODE = True
    DESCRIPTION = "Batch transcribe video files in a folder to SRT subtitles using Gemini via Vertex AI."

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "extra_instructions": ("STRING", {
                "default": "",
                "multiline": True,
                "placeholder": "Optional: additional transcription instructions applied to all videos",
            }),
            "identify_speakers": ("BOOLEAN", {
                "default": True,
                "tooltip": "Try to identify and label different speakers.",
            }),
        }
        optional.update(_processing_inputs())
        optional.update(_styling_inputs())
        optional["delay_seconds"] = ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 30.0, "step": 0.5,
            "tooltip": "Delay between API calls to avoid rate limiting.",
        })
        optional["projekts_root"] = (
            [r for r in PROJEKTS_ROOTS if os.path.isdir(r)] or PROJEKTS_ROOTS,
        )
        optional["project"] = (
            scan_projects(
                next((r for r in PROJEKTS_ROOTS if os.path.isdir(r)), PROJEKTS_ROOTS[0])
            ),
        )

        return {
            "required": {
                "video_folder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to folder of video files",
                    "tooltip": "Folder containing video files to transcribe. Scans recursively.",
                }),
                "file_types": (["all", "mp4", "mov", "mxf", "mkv", "avi", "m4v", "qt"], {
                    "default": "all",
                    "tooltip": "Which video file types to process. 'all' includes mp4, mov, qt, m4v, mkv, avi, mxf.",
                }),
                "subtitle_output": (SUBTITLE_OUTPUT_MODES, {
                    "default": "srt_only",
                    "tooltip": "srt_only: sidecar file(s). burn_in_only: hardcode subs into video. both: file(s) + burned-in video.",
                }),
                "model": (cls.MODELS, {"default": "gemini-2.5-flash"}),
                "output_mode": (["alongside_video", "projekts_auto_srt"], {
                    "default": "alongside_video",
                    "tooltip": "alongside_video: save next to each video. projekts_auto_srt: save all to project auto_srt folder.",
                }),
                "overwrite": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Overwrite existing output files. If false, skips videos that already have output.",
                }),
                "gcp_project_id": ("STRING", {
                    "default": default_project(),
                    "tooltip": "GCP project ID.",
                }),
                "gcp_region": ("STRING", {
                    "default": default_region(),
                    "tooltip": "GCP region.",
                }),
            },
            "optional": optional,
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def batch_transcribe(self, video_folder, file_types, subtitle_output, model,
                         output_mode, overwrite,
                         gcp_project_id="", gcp_region="global",
                         extra_instructions="", identify_speakers=True,
                         pad_frames=0, frame_rate=23.976, snap_to_frames=False,
                         max_chars_per_line=42, max_lines=2,
                         remove_hallucinations=True,
                         output_format="srt", language="auto", translate_to="none",
                         font_name="Arial", font_size=24,
                         font_color="white", outline_color="black",
                         outline_width=2, shadow_depth=1,
                         position="bottom_center", margin_v=30,
                         delay_seconds=1.0, projekts_root="", project=""):

        from google import genai

        video_folder = video_folder.strip()
        if not os.path.isdir(video_folder):
            raise ValueError(f"[DigitBatchSRT] Folder not found: {video_folder}")

        # Build extension filter
        if file_types == "all":
            allowed_exts = VIDEO_EXTENSIONS
        else:
            allowed_exts = {f".{file_types}"}

        # Recursively find video files matching the filter
        video_files = []
        for root, _dirs, files in os.walk(video_folder):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in allowed_exts:
                    video_files.append(os.path.join(root, f))
        video_files.sort()

        if not video_files:
            msg = f"No video files found in {video_folder}"
            return {"ui": {"log_text": [msg]}, "result": (msg, 0, video_folder)}

        # Determine output folder
        if output_mode == "projekts_auto_srt":
            if not projekts_root or not project or project == "(no projects found)":
                raise ValueError(
                    "[DigitBatchSRT] projekts_root and project are required for projekts_auto_srt output mode."
                )
            out_dir = os.path.join(projekts_root, project, "assets", "auto_srt")
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = video_folder

        # Setup Gemini
        gcp_project, gcp_reg = resolve_gcp_config(gcp_project_id, gcp_region)
        client = genai.Client(vertexai=True, project=gcp_project, location=gcp_reg)

        system_prompt = TRANSCRIBE_SYSTEM_PROMPT
        if not identify_speakers:
            system_prompt += "\nDo NOT label or identify speakers. Just transcribe the words.\n"
        if language != "auto":
            system_prompt += f"\nThe audio is in {language}. Transcribe in that language.\n"

        user_prompt = "Transcribe this audio into SRT subtitle format with accurate timestamps."
        if extra_instructions and extra_instructions.strip():
            user_prompt += f"\n\nAdditional instructions:\n{extra_instructions.strip()}"

        # Process
        total = len(video_files)
        pbar = comfy.utils.ProgressBar(total)
        log_lines = []
        transcribed = 0
        skipped = 0
        errors = 0

        print(f"[DigitBatchSRT] Processing {total} video files from: {video_folder}")

        for idx, video_path in enumerate(video_files):
            vf = os.path.relpath(video_path, video_folder)
            base_name = os.path.splitext(os.path.basename(video_path))[0]

            if output_mode == "alongside_video":
                srt_path = os.path.join(os.path.dirname(video_path), f"{base_name}.srt")
            else:
                srt_path = os.path.join(out_dir, f"{base_name}.srt")

            # Skip if output already exists and overwrite is off
            if not overwrite:
                ext = os.path.splitext(video_path)[1]
                burned_check = os.path.join(
                    os.path.dirname(srt_path), f"{base_name}_subtitled{ext}")
                srt_exists = os.path.isfile(srt_path)
                burned_exists = os.path.isfile(burned_check)

                skip = False
                if subtitle_output == "srt_only" and srt_exists:
                    skip = True
                elif subtitle_output == "burn_in_only" and burned_exists:
                    skip = True
                elif subtitle_output == "both" and srt_exists and burned_exists:
                    skip = True

                if skip:
                    skipped += 1
                    log_lines.append(f"[{idx + 1}/{total}] {vf} -> SKIPPED (exists)")
                    pbar.update_absolute(idx + 1)
                    continue

            try:
                print(f"[DigitBatchSRT] [{idx + 1}/{total}] Processing: {vf}")

                srt_text = _transcribe_single(video_path, client, model,
                                              system_prompt, user_prompt)

                # Post-process
                srt_text, warnings = _post_process_srt(
                    srt_text, pad_frames=pad_frames, frame_rate=frame_rate,
                    snap_frames=snap_to_frames,
                    max_chars_per_line=max_chars_per_line,
                    max_lines=max_lines,
                    remove_hallucinations=remove_hallucinations)
                for w in warnings:
                    print(f"[DigitBatchSRT]   {w}")

                # Translation
                if translate_to != "none":
                    srt_text = _translate_srt(srt_text, client, model, translate_to)

                entry_count = len(_parse_srt(srt_text))
                parts = []

                # Write output files
                if subtitle_output in ("srt_only", "both"):
                    saved = _save_formats(
                        srt_text, srt_path, output_format,
                        font_name=font_name, font_size=font_size,
                        primary_color=font_color, outline_color=outline_color,
                        outline_width=outline_width, shadow_depth=shadow_depth,
                        alignment=position, margin_v=margin_v)
                    fmts = [os.path.splitext(p)[1] for p in saved]
                    parts.append(f"{entry_count} entries ({','.join(fmts)})")

                # Burn in subtitles
                if subtitle_output in ("burn_in_only", "both"):
                    # Check if we have an ASS file to use
                    ass_check = os.path.join(
                        os.path.dirname(srt_path), f"{base_name}.ass")
                    if os.path.isfile(ass_check):
                        burn_file = ass_check
                        tmp_burn = None
                    else:
                        tmp_burn = tempfile.NamedTemporaryFile(
                            suffix=".srt", delete=False, mode="w")
                        tmp_burn.write(srt_text)
                        tmp_burn.close()
                        burn_file = tmp_burn.name

                    try:
                        ext = os.path.splitext(video_path)[1]
                        burned_dir = os.path.dirname(srt_path)
                        burned_path = os.path.join(
                            burned_dir, f"{base_name}_subtitled{ext}")
                        _burn_in_subtitles(
                            video_path, burn_file, burned_path,
                            font_name=font_name, font_size=font_size,
                            primary_color=font_color, outline_color=outline_color,
                            outline_width=outline_width, shadow_depth=shadow_depth,
                            alignment=position, margin_v=margin_v)
                        parts.append("burned in")
                    finally:
                        if tmp_burn and os.path.isfile(tmp_burn.name):
                            os.unlink(tmp_burn.name)

                transcribed += 1
                status = f"[{idx + 1}/{total}] {vf} -> OK ({', '.join(parts)})"
                log_lines.append(status)
                logger.info("DigitBatchSRT: %s", status)

            except Exception as e:
                errors += 1
                status = f"[{idx + 1}/{total}] {vf} -> ERROR: {e}"
                log_lines.append(status)
                logger.error("DigitBatchSRT: %s", status)

            pbar.update_absolute(idx + 1)

            # Rate limit delay between files
            if delay_seconds > 0 and idx < total - 1:
                time.sleep(delay_seconds)

        # Summary
        summary = (
            f"Done. Transcribed: {transcribed}, Skipped: {skipped}, "
            f"Errors: {errors}, Total: {total}"
        )
        log_lines.append("")
        log_lines.append(summary)
        log_text = "\n".join(log_lines)

        print(f"[DigitBatchSRT] {summary}")
        logger.info("DigitBatchSRT: %s", summary)

        return {
            "ui": {"log_text": [summary]},
            "result": (log_text, transcribed, out_dir),
        }


# ── SRT Tools node ───────────────────────────────────────────────────────────

class DigitSRTTools:
    """Post-process and convert SRT subtitle files."""

    CATEGORY = "DIGIT"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("output_text", "output_filepath", "log")
    FUNCTION = "process"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        optional.update(_processing_inputs())
        optional.update(_styling_inputs())

        return {
            "required": {
                "srt_input": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Paste SRT text here, or use srt_filepath input",
                }),
                "action": ([
                    "post_process",
                    "convert_format",
                    "time_offset",
                    "merge",
                    "renumber",
                ], {
                    "default": "post_process",
                    "tooltip": (
                        "post_process: apply line-length, hallucination removal, padding, snap. "
                        "convert_format: convert to VTT/ASS/TXT. "
                        "time_offset: shift all timestamps. "
                        "merge: merge adjacent entries with small gaps. "
                        "renumber: re-number all entries sequentially."
                    ),
                }),
            },
            "optional": {
                "srt_filepath": ("STRING", {
                    "default": "",
                    "tooltip": "Path to an SRT file. If provided, overrides srt_input text.",
                }),
                "save_filepath": ("STRING", {
                    "default": "",
                    "tooltip": "Path to save output. Leave empty to not save (output via text only).",
                }),
                "time_offset_ms": ("INT", {
                    "default": 0, "min": -600000, "max": 600000, "step": 100,
                    "tooltip": "Milliseconds to shift all timestamps (positive = later, negative = earlier). Used with time_offset action.",
                }),
                "merge_gap_ms": ("INT", {
                    "default": 500, "min": 0, "max": 5000, "step": 100,
                    "tooltip": "Merge adjacent subtitles with gaps smaller than this (ms). Used with merge action.",
                }),
                **optional,
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def process(self, srt_input, action,
                srt_filepath="", save_filepath="",
                time_offset_ms=0, merge_gap_ms=500,
                pad_frames=0, frame_rate=23.976, snap_to_frames=False,
                max_chars_per_line=42, max_lines=2,
                remove_hallucinations=True,
                output_format="srt", language="auto", translate_to="none",
                font_name="Arial", font_size=24,
                font_color="white", outline_color="black",
                outline_width=2, shadow_depth=1,
                position="bottom_center", margin_v=30):

        # Get input SRT
        if srt_filepath and srt_filepath.strip() and os.path.isfile(srt_filepath.strip()):
            with open(srt_filepath.strip(), "r", encoding="utf-8") as f:
                srt_text = f.read()
        elif srt_input.strip():
            srt_text = srt_input.strip()
        else:
            raise ValueError("[DigitSRTTools] Provide SRT text or a filepath.")

        log_parts = []

        if action == "post_process":
            srt_text, warnings = _post_process_srt(
                srt_text, pad_frames=pad_frames, frame_rate=frame_rate,
                snap_frames=snap_to_frames,
                max_chars_per_line=max_chars_per_line,
                max_lines=max_lines,
                remove_hallucinations=remove_hallucinations)
            log_parts.extend(warnings)
            log_parts.append(f"Post-processed: {len(_parse_srt(srt_text))} entries")

        elif action == "convert_format":
            # Output will be saved in requested format
            log_parts.append(f"Converting to {output_format}")

        elif action == "time_offset":
            offset_sec = time_offset_ms / 1000.0
            entries = _parse_srt(srt_text)
            shifted = []
            for idx, start, end, text in entries:
                shifted.append((idx, max(0, start + offset_sec), max(0, end + offset_sec), text))
            srt_text = _entries_to_srt(shifted)
            log_parts.append(f"Shifted {len(entries)} entries by {time_offset_ms}ms")

        elif action == "merge":
            gap_sec = merge_gap_ms / 1000.0
            entries = _parse_srt(srt_text)
            if entries:
                merged = [entries[0]]
                for idx, start, end, text in entries[1:]:
                    prev_idx, prev_start, prev_end, prev_text = merged[-1]
                    if start - prev_end <= gap_sec:
                        # Merge
                        combined_text = prev_text + "\n" + text
                        merged[-1] = (prev_idx, prev_start, end, combined_text)
                    else:
                        merged.append((idx, start, end, text))
                before = len(entries)
                srt_text = _entries_to_srt(merged)
                log_parts.append(f"Merged {before} -> {len(merged)} entries (gap <= {merge_gap_ms}ms)")
            else:
                log_parts.append("No entries to merge")

        elif action == "renumber":
            entries = _parse_srt(srt_text)
            srt_text = _entries_to_srt(entries)
            log_parts.append(f"Renumbered {len(entries)} entries")

        # Save if requested
        output_path = ""
        if save_filepath and save_filepath.strip():
            save_path = save_filepath.strip()
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

            if action == "convert_format":
                saved = _save_formats(
                    srt_text, save_path, output_format,
                    font_name=font_name, font_size=font_size,
                    primary_color=font_color, outline_color=outline_color,
                    outline_width=outline_width, shadow_depth=shadow_depth,
                    alignment=position, margin_v=margin_v)
                output_path = saved[0] if saved else save_path
                log_parts.append(f"Saved: {', '.join(os.path.basename(p) for p in saved)}")
            else:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(srt_text)
                output_path = save_path
                log_parts.append(f"Saved: {os.path.basename(save_path)}")

        log_text = "\n".join(log_parts) if log_parts else "OK"

        return {
            "ui": {"log_text": [log_text]},
            "result": (srt_text, output_path, log_text),
        }


# ── SRT Preview node ─────────────────────────────────────────────────────────

class DigitSRTPreview:
    """Preview and validate SRT subtitle content."""

    CATEGORY = "DIGIT"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("summary", "entry_count", "warnings")
    FUNCTION = "preview"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "srt_input": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Paste SRT text here, or use srt_filepath input",
                }),
            },
            "optional": {
                "srt_filepath": ("STRING", {
                    "default": "",
                    "tooltip": "Path to an SRT file. Overrides srt_input if provided.",
                }),
                "max_chars_per_line": ("INT", {
                    "default": 42, "min": 0, "max": 80,
                    "tooltip": "Flag lines exceeding this length. 0 = no check.",
                }),
                "max_cps": ("FLOAT", {
                    "default": 20.0, "min": 0.0, "max": 40.0, "step": 0.5,
                    "tooltip": "Max characters per second reading speed. Flag entries exceeding this. 0 = no check.",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def preview(self, srt_input, srt_filepath="",
                max_chars_per_line=42, max_cps=20.0):

        if srt_filepath and srt_filepath.strip() and os.path.isfile(srt_filepath.strip()):
            with open(srt_filepath.strip(), "r", encoding="utf-8") as f:
                srt_text = f.read()
        elif srt_input.strip():
            srt_text = srt_input.strip()
        else:
            return {"ui": {"log_text": ["No SRT input provided"]},
                    "result": ("No input", 0, "")}

        entries = _parse_srt(srt_text)
        entry_count = len(entries)
        warnings = []

        if not entries:
            return {"ui": {"log_text": ["Empty SRT — no entries found"]},
                    "result": ("Empty SRT", 0, "")}

        total_duration = entries[-1][2] - entries[0][1]
        total_chars = sum(len(text) for _, _, _, text in entries)

        # Check for issues
        for i, (idx, start, end, text) in enumerate(entries):
            duration = end - start

            # Overlapping timestamps
            if i > 0:
                prev_end = entries[i - 1][2]
                if start < prev_end - 0.01:
                    warnings.append(
                        f"OVERLAP: Entry {i+1} starts at {_seconds_to_srt_ts(start)} "
                        f"but previous ends at {_seconds_to_srt_ts(prev_end)}")

            # Long lines
            if max_chars_per_line > 0:
                for line in text.split("\n"):
                    if len(line) > max_chars_per_line:
                        warnings.append(
                            f"LONG LINE: Entry {i+1} has {len(line)} chars "
                            f"(max {max_chars_per_line}): '{line[:50]}...'")

            # Too many lines
            line_count = len(text.split("\n"))
            if line_count > 2:
                warnings.append(f"TOO MANY LINES: Entry {i+1} has {line_count} lines")

            # CPS check
            if max_cps > 0 and duration > 0:
                char_count = len(text.replace("\n", ""))
                cps = char_count / duration
                if cps > max_cps:
                    warnings.append(
                        f"FAST: Entry {i+1} is {cps:.1f} CPS "
                        f"(max {max_cps}): '{text[:40]}...'")

            # Zero/negative duration
            if duration <= 0:
                warnings.append(f"BAD TIMING: Entry {i+1} has {duration:.3f}s duration")

        # Build summary
        minutes = int(total_duration // 60)
        seconds = int(total_duration % 60)
        summary_lines = [
            f"Entries: {entry_count}",
            f"Duration: {minutes}m {seconds}s",
            f"Total characters: {total_chars}",
            f"Avg chars/entry: {total_chars // max(entry_count, 1)}",
        ]
        if warnings:
            summary_lines.append(f"Warnings: {len(warnings)}")
        else:
            summary_lines.append("No issues found")

        summary = "\n".join(summary_lines)
        warnings_text = "\n".join(warnings) if warnings else ""

        print(f"[DigitSRTPreview] {entry_count} entries, {minutes}m{seconds}s, {len(warnings)} warnings")

        return {
            "ui": {"log_text": [summary]},
            "result": (summary, entry_count, warnings_text),
        }

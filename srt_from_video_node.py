"""DIGIT SRT From Video — transcribe video files to SRT using Gemini."""

import logging
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


def _transcribe_single(video_path, client, model, system_prompt, user_prompt):
    """Extract audio from a video file and transcribe it via Gemini.

    Returns the SRT text string.
    """
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


SUBTITLE_OUTPUT_MODES = ["srt_only", "burn_in_only", "both"]

SRT_TIMESTAMP_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)


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


def _pad_srt(srt_text, pad_frames, frame_rate):
    """Pad each subtitle entry by pad_frames on both sides."""
    if pad_frames <= 0 or frame_rate <= 0:
        return srt_text
    pad_seconds = pad_frames / frame_rate

    def _replace(match):
        start = _srt_ts_to_seconds(
            match.group(1), match.group(2), match.group(3), match.group(4))
        end = _srt_ts_to_seconds(
            match.group(5), match.group(6), match.group(7), match.group(8))
        start = max(0, start - pad_seconds)
        end = end + pad_seconds
        return f"{_seconds_to_srt_ts(start)} --> {_seconds_to_srt_ts(end)}"

    return SRT_TIMESTAMP_RE.sub(_replace, srt_text)


def _burn_in_subtitles(video_path, srt_path, output_path):
    """Burn SRT subtitles into the video using ffmpeg subtitles filter.

    Outputs to a new file with _subtitled suffix.
    """
    # The subtitles filter needs the SRT path with special chars escaped
    # Using the subtitles filter with the filename option avoids shell issues
    escaped_srt = srt_path.replace("'", r"'\''").replace(":", r"\:")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"subtitles='{escaped_srt}'",
        "-c:a", "copy",
        output_path,
    ]
    print(f"[DigitSRT] Burning subtitles into: {os.path.basename(output_path)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg burn-in failed: {result.stderr.strip()}")
    return output_path


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
                    "tooltip": "srt_only: SRT sidecar file. burn_in_only: hardcode subs into video. both: SRT file + burned-in video.",
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
                    "tooltip": "GCP project ID. Auto-detected from DIGIT_GCP_PROJECT env var or GCP metadata.",
                }),
                "gcp_region": ("STRING", {
                    "default": default_region(),
                    "tooltip": "GCP region. Auto-detected from DIGIT_GCP_REGION env var or GCP metadata. Defaults to 'global'.",
                }),
            },
            "optional": {
                "identify_speakers": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Try to identify and label different speakers.",
                }),
                "pad_frames": ("INT", {
                    "default": 0, "min": 0, "max": 120, "step": 1,
                    "tooltip": "Extend each subtitle by this many frames on both sides (head and tail).",
                }),
                "frame_rate": ("FLOAT", {
                    "default": 23.976, "min": 1.0, "max": 120.0, "step": 0.001,
                    "tooltip": "Frame rate of the video. Used to convert pad_frames to seconds.",
                }),
            },
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
                         identify_speakers=True, pad_frames=0, frame_rate=23.976):

        from google import genai

        video_path = video_path.strip()
        if not video_path:
            raise ValueError("[DigitSRTFromVideo] video_path is required.")
        if not os.path.isfile(video_path):
            raise ValueError(f"[DigitSRTFromVideo] File not found: {video_path}")

        print(f"[DigitSRTFromVideo] Extracting audio from: {video_path}")

        gcp_project, gcp_reg = resolve_gcp_config(gcp_project_id, gcp_region)
        client = genai.Client(vertexai=True, project=gcp_project, location=gcp_reg)

        system_prompt = TRANSCRIBE_SYSTEM_PROMPT
        if not identify_speakers:
            system_prompt += "\nDo NOT label or identify speakers. Just transcribe the words.\n"

        user_prompt = "Transcribe this audio into SRT subtitle format with accurate timestamps."
        if extra_instructions and extra_instructions.strip():
            user_prompt += f"\n\nAdditional instructions:\n{extra_instructions.strip()}"

        srt_text = _transcribe_single(video_path, client, model, system_prompt, user_prompt)

        # Apply frame padding
        if pad_frames > 0:
            srt_text = _pad_srt(srt_text, pad_frames, frame_rate)
            print(f"[DigitSRTFromVideo] Padded subtitles by {pad_frames} frames @ {frame_rate} fps")

        # Save SRT
        target_dir = os.path.join(projekts_root, project, "assets", "auto_srt")
        os.makedirs(target_dir, exist_ok=True)

        srt_filename = f"{filename}.srt"
        filepath = os.path.join(target_dir, srt_filename)

        if subtitle_output in ("srt_only", "both"):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(srt_text)
            entry_count = len(re.findall(r"^\d+$", srt_text, re.MULTILINE))
            print(f"[DigitSRTFromVideo] Saved {entry_count} entries to: {filepath}")

        # Burn in
        if subtitle_output in ("burn_in_only", "both"):
            # Write temp SRT for burn-in if we didn't save one above
            if subtitle_output == "burn_in_only":
                tmp_srt = tempfile.NamedTemporaryFile(suffix=".srt", delete=False, mode="w")
                tmp_srt.write(srt_text)
                tmp_srt.close()
                burn_srt = tmp_srt.name
            else:
                burn_srt = filepath

            try:
                ext = os.path.splitext(video_path)[1]
                burned_path = os.path.join(target_dir, f"{filename}_subtitled{ext}")
                _burn_in_subtitles(video_path, burn_srt, burned_path)
                print(f"[DigitSRTFromVideo] Burned-in video saved to: {burned_path}")
            finally:
                if subtitle_output == "burn_in_only" and os.path.isfile(burn_srt):
                    os.unlink(burn_srt)

        return {
            "ui": {"filepath_text": [filepath]},
            "result": (filepath, srt_text),
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
        return {
            "required": {
                "video_folder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to folder of video files",
                    "tooltip": "Folder containing MP4/MOV/MXF files to transcribe.",
                }),
                "file_types": (["all", "mp4", "mov", "mxf", "mkv", "avi", "m4v", "qt"], {
                    "default": "all",
                    "tooltip": "Which video file types to process. 'all' includes mp4, mov, qt, m4v, mkv, avi, mxf.",
                }),
                "subtitle_output": (SUBTITLE_OUTPUT_MODES, {
                    "default": "srt_only",
                    "tooltip": "srt_only: SRT sidecar file. burn_in_only: hardcode subs into video. both: SRT file + burned-in video.",
                }),
                "model": (cls.MODELS, {"default": "gemini-2.5-flash"}),
                "output_mode": (["alongside_video", "projekts_auto_srt"], {
                    "default": "alongside_video",
                    "tooltip": "alongside_video: save .srt next to each video. projekts_auto_srt: save all to project auto_srt folder.",
                }),
                "overwrite": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Overwrite existing .srt files. If false, skips videos that already have an SRT.",
                }),
                "gcp_project_id": ("STRING", {
                    "default": default_project(),
                    "tooltip": "GCP project ID. Auto-detected from DIGIT_GCP_PROJECT env var or GCP metadata.",
                }),
                "gcp_region": ("STRING", {
                    "default": default_region(),
                    "tooltip": "GCP region. Auto-detected from DIGIT_GCP_REGION env var or GCP metadata. Defaults to 'global'.",
                }),
            },
            "optional": {
                "extra_instructions": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Optional: additional transcription instructions applied to all videos",
                }),
                "identify_speakers": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Try to identify and label different speakers.",
                }),
                "pad_frames": ("INT", {
                    "default": 0, "min": 0, "max": 120, "step": 1,
                    "tooltip": "Extend each subtitle by this many frames on both sides (head and tail).",
                }),
                "frame_rate": ("FLOAT", {
                    "default": 23.976, "min": 1.0, "max": 120.0, "step": 0.001,
                    "tooltip": "Frame rate of the video. Used to convert pad_frames to seconds.",
                }),
                "delay_seconds": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 30.0, "step": 0.5,
                    "tooltip": "Delay between API calls to avoid rate limiting.",
                }),
                "projekts_root": (
                    [r for r in PROJEKTS_ROOTS if os.path.isdir(r)] or PROJEKTS_ROOTS,
                ),
                "project": (
                    scan_projects(
                        next((r for r in PROJEKTS_ROOTS if os.path.isdir(r)), PROJEKTS_ROOTS[0])
                    ),
                ),
            },
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
                         pad_frames=0, frame_rate=23.976,
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

                if pad_frames > 0:
                    srt_text = _pad_srt(srt_text, pad_frames, frame_rate)

                entry_count = len(re.findall(r"^\d+$", srt_text, re.MULTILINE))
                parts = []

                # Write SRT sidecar
                if subtitle_output in ("srt_only", "both"):
                    with open(srt_path, "w", encoding="utf-8") as f:
                        f.write(srt_text)
                    parts.append(f"{entry_count} entries")

                # Burn in subtitles
                if subtitle_output in ("burn_in_only", "both"):
                    # Write temp SRT if we didn't save a sidecar
                    if subtitle_output == "burn_in_only":
                        tmp_srt = tempfile.NamedTemporaryFile(
                            suffix=".srt", delete=False, mode="w")
                        tmp_srt.write(srt_text)
                        tmp_srt.close()
                        burn_srt = tmp_srt.name
                    else:
                        burn_srt = srt_path

                    try:
                        ext = os.path.splitext(video_path)[1]
                        burned_dir = os.path.dirname(srt_path)
                        burned_path = os.path.join(
                            burned_dir, f"{base_name}_subtitled{ext}")
                        _burn_in_subtitles(video_path, burn_srt, burned_path)
                        parts.append("burned in")
                    finally:
                        if subtitle_output == "burn_in_only" and os.path.isfile(burn_srt):
                            os.unlink(burn_srt)

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

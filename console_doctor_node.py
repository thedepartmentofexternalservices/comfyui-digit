"""DIGIT Console Doctor — read ComfyUI logs, send errors to Gemini for diagnosis."""

import logging
import re

import requests

from .gcp_config import get_gcp_access_token, build_vertex_url, resolve_gcp_config, default_project, default_region

logger = logging.getLogger(__name__)

LOG_LEVELS = ["errors_only", "warnings_and_errors", "all"]

DOCTOR_SYSTEM_PROMPT = """\
You are an expert ComfyUI debugger. You are given recent console log output from a ComfyUI instance.

Your job:
1. Identify errors, warnings, or problems in the logs.
2. Explain what went wrong in plain language.
3. Suggest specific fixes — file paths, node settings, missing dependencies, etc.
4. If there are no errors, say so clearly.

Keep your response concise and actionable. Use bullet points. Do not repeat the raw log text back.
If you see a Python traceback, identify the root cause (the last exception in the chain).
If you see a missing module error, suggest the pip install command.
If you see a CUDA/GPU error, suggest common fixes (VRAM, driver version, etc.).
If you see a node execution error, identify which node failed and why.
"""

# Patterns that indicate errors/problems in logs
ERROR_PATTERNS = re.compile(
    r"(?i)(error|exception|traceback|failed|critical|fatal|"
    r"CUDA out of memory|ModuleNotFoundError|ImportError|"
    r"FileNotFoundError|ValueError|RuntimeError|KeyError|"
    r"AttributeError|TypeError|NameError|OSError|"
    r"ConnectionError|TimeoutError|PermissionError)"
)

WARNING_PATTERNS = re.compile(
    r"(?i)(warning|warn|deprecated|could not|unable to|"
    r"falling back|not found|skipping|retry)"
)


def _get_comfyui_logs():
    """Read logs from ComfyUI's built-in log buffer."""
    try:
        import app.logger
        return list(app.logger.get_logs())
    except (ImportError, AttributeError):
        return []


def _filter_logs(logs, level, node_filter, max_entries):
    """Filter log entries by severity and optional node name."""
    filtered = []

    for entry in logs:
        msg = entry.get("m", "")
        if not msg or not msg.strip():
            continue

        # Apply node filter if set
        if node_filter and node_filter.strip():
            if node_filter.lower() not in msg.lower():
                continue

        # Apply severity filter
        if level == "errors_only":
            if not ERROR_PATTERNS.search(msg):
                continue
        elif level == "warnings_and_errors":
            if not ERROR_PATTERNS.search(msg) and not WARNING_PATTERNS.search(msg):
                continue
        # "all" passes everything through

        filtered.append(entry)

    # Return the most recent entries
    return filtered[-max_entries:]


def _format_logs(entries):
    """Format log entries into readable text."""
    lines = []
    for entry in entries:
        timestamp = entry.get("t", "")
        msg = entry.get("m", "").rstrip()
        if msg:
            # Shorten timestamp to just time portion
            time_part = timestamp.split("T")[-1][:12] if "T" in timestamp else timestamp
            lines.append(f"[{time_part}] {msg}")
    return "\n".join(lines)


class DigitConsoleDoctor:
    """Read ComfyUI console logs, filter for errors, and get AI-powered diagnosis."""

    MODELS = [
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

    CATEGORY = "DIGIT"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("diagnosis", "raw_logs")
    FUNCTION = "diagnose"
    OUTPUT_NODE = True
    DESCRIPTION = "Read ComfyUI console logs, filter for errors/warnings, and get an AI diagnosis via Gemini."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("INT", {
                    "default": 0, "min": 0, "max": 999999,
                    "tooltip": "Change this value to re-run the diagnosis.",
                }),
                "model": (cls.MODELS, {"default": "gemini-2.5-flash"}),
                "log_level": (LOG_LEVELS, {
                    "default": "warnings_and_errors",
                    "tooltip": "Filter logs by severity level.",
                }),
                "max_log_entries": ("INT", {
                    "default": 50, "min": 5, "max": 300,
                    "tooltip": "Maximum number of log entries to send to Gemini.",
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
                "run_after": ("*", {
                    "tooltip": "Connect any output from the last node in your workflow to ensure Console Doctor runs last.",
                }),
                "node_filter": ("STRING", {
                    "default": "",
                    "tooltip": "Only show logs containing this text (e.g. a node name). Leave empty for all.",
                }),
                "extra_context": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Extra context to include with the diagnosis request (e.g. 'I was trying to load a LoRA').",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def diagnose(self, trigger, model, log_level, max_log_entries,
                 run_after=None, node_filter="", extra_context="",
                 gcp_project_id="", gcp_region=""):
        # Get logs from ComfyUI's buffer
        all_logs = _get_comfyui_logs()

        if not all_logs:
            msg = "No logs available. ComfyUI's log buffer may be empty."
            return {"ui": {"doctor_text": [msg]},
                    "result": (msg, "")}

        # Filter
        filtered = _filter_logs(all_logs, log_level, node_filter, max_log_entries)
        raw_text = _format_logs(filtered)

        if not raw_text.strip():
            msg = f"No {log_level.replace('_', ' ')} found in the last {len(all_logs)} log entries."
            if node_filter:
                msg += f" (filtered by: '{node_filter}')"
            return {"ui": {"doctor_text": [msg]},
                    "result": (msg, _format_logs(list(all_logs)[-20:]))}

        # Resolve GCP
        project, region = resolve_gcp_config(gcp_project_id, gcp_region)

        # Build prompt
        user_prompt = f"Here are the recent ComfyUI console logs:\n\n```\n{raw_text}\n```"
        if extra_context and extra_context.strip():
            user_prompt += f"\n\nAdditional context from the user:\n{extra_context.strip()}"

        # Call Gemini
        token = get_gcp_access_token()
        body = {
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.3},
            "systemInstruction": {"parts": [{"text": DOCTOR_SYSTEM_PROMPT}]},
        }

        url = build_vertex_url(project, region, model)

        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=body,
            timeout=120,
        )
        resp.raise_for_status()
        diagnosis = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

        # Truncate display for the node face
        display_lines = diagnosis.split("\n")[:10]
        display_text = "\n".join(display_lines)
        if len(display_lines) < len(diagnosis.split("\n")):
            display_text += "\n..."

        entry_count = len(filtered)
        header = f"Analyzed {entry_count} log entries ({log_level.replace('_', ' ')})"
        if node_filter:
            header += f" [filter: {node_filter}]"

        return {"ui": {"doctor_text": [f"{header}\n\n{display_text}"]},
                "result": (diagnosis, raw_text)}

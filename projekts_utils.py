"""Shared PROJEKTS pipeline utilities used by Image Saver, Video Saver, and SRT Maker."""

import os
import re

# Override with DIGIT_PROJEKTS_ROOTS env var (colon-separated paths).
# Falls back to common mount points, then home directory.
_DEFAULT_ROOTS = [
    os.path.join(os.path.expanduser("~"), "PROJEKTS"),
]

def _resolve_projekts_roots():
    env = os.environ.get("DIGIT_PROJEKTS_ROOTS", "")
    if env:
        return [p.strip() for p in env.split(":") if p.strip()]
    # Auto-detect common mount points
    candidates = [
        "/mnt/projekts/PROJEKTS",
        "/Volumes/projekts/PROJEKTS",
    ]
    found = [c for c in candidates if os.path.isdir(c)]
    return found if found else _DEFAULT_ROOTS

PROJEKTS_ROOTS = _resolve_projekts_roots()

PROJECT_RE = re.compile(r"^\d{5}_")
FRAME_RE = re.compile(r"\.(\d+)\.[^.]+$")


def scan_projects(projekts_root):
    """Return sorted list of project folders matching 5-digit prefix pattern."""
    if not os.path.isdir(projekts_root):
        return ["(no projects found)"]
    folders = [
        d for d in sorted(os.listdir(projekts_root))
        if os.path.isdir(os.path.join(projekts_root, d)) and PROJECT_RE.match(d)
    ]
    return folders if folders else ["(no projects found)"]


def scan_shots(projekts_root, project):
    """Return sorted list of shot folders inside <project>/shots/."""
    shots_dir = os.path.join(projekts_root, project, "shots")
    if not os.path.isdir(shots_dir):
        return ["(no shots found)"]
    folders = sorted(
        d for d in os.listdir(shots_dir)
        if os.path.isdir(os.path.join(shots_dir, d))
    )
    return folders if folders else ["(no shots found)"]


def next_frame(target_dir, prefix, shot, task, ext, start_frame, frame_pad):
    """Find highest existing frame number in target_dir and return next frame number."""
    pat = re.compile(
        rf"^{re.escape(prefix)}_{re.escape(shot)}_{re.escape(task)}\.(\d+)\.{re.escape(ext)}$"
    )
    max_frame = start_frame - 1
    if os.path.isdir(target_dir):
        for f in os.listdir(target_dir):
            m = pat.match(f)
            if m:
                max_frame = max(max_frame, int(m.group(1)))
    return max_frame + 1

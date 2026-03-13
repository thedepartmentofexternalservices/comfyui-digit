"""Shared PROJEKTS pipeline utilities used by Image Saver, Video Saver, and SRT Maker."""

import os
import re

PROJEKTS_ROOTS = [
    "/Volumes/saint/goose/PROJEKTS",
    "/mnt/lucid/PROJEKTS",
]

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

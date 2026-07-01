"""Shared PROJEKTS pipeline utilities used by Image Saver, Video Saver, and SRT Maker."""

import os
import re

# Override with DIGIT_PROJEKTS_ROOTS env var (colon-separated paths).
# Falls back to common mount points, then home directory.
_DEFAULT_ROOTS = [
    os.path.join(os.path.expanduser("~"), "PROJEKTS"),
]

# Known mount points checked on each call so late-mounted volumes are picked up.
_CANDIDATE_ROOTS = [
    "/mnt/projekts/PROJEKTS",
    "/Volumes/projekts/PROJEKTS",
    "/Volumes/saint/goose/PROJEKTS",
    "/mnt/lucid/PROJEKTS",
]


def get_projekts_roots():
    """Return available PROJEKTS roots, re-scanning mount points each call."""
    env = os.environ.get("DIGIT_PROJEKTS_ROOTS", "")
    if env:
        return [p.strip() for p in env.split(":") if p.strip()]
    found = [c for c in _CANDIDATE_ROOTS if os.path.isdir(c)]
    return found if found else _DEFAULT_ROOTS


def get_available_projekts_roots():
    """Roots that currently exist on disk; falls back to configured list."""
    roots = get_projekts_roots()
    available = [r for r in roots if os.path.isdir(r)]
    return available if available else roots


# Back-compat alias; prefer get_projekts_roots() for fresh results.
PROJEKTS_ROOTS = get_projekts_roots()

PROJECT_RE = re.compile(r"^\d{5}_")
FRAME_RE = re.compile(r"\.(\d+)\.[^.]+$")


def is_within_roots(path, roots=None):
    """Return True if `path` resolves to a location inside one of the PROJEKTS roots.

    Both sides are passed through os.path.realpath, so symlinks and ``..``
    traversal cannot escape the configured roots. Used to constrain the
    /digit/browse listing endpoint to the pipeline filespace (security: M1).
    """
    roots = roots if roots is not None else get_projekts_roots()
    try:
        real = os.path.realpath(path)
    except (OSError, ValueError):
        return False
    for root in roots:
        try:
            real_root = os.path.realpath(root)
        except (OSError, ValueError):
            continue
        if real == real_root or real.startswith(real_root + os.sep):
            return True
    return False


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

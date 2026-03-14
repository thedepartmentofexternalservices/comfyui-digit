"""DIGIT Caption Find & Replace — bulk find/replace text across caption files."""

import logging
import os
import re

import comfy.utils

logger = logging.getLogger(__name__)


class DigitCaptionFindReplace:
    """Bulk find and replace text in .txt caption files across a dataset folder."""

    CATEGORY = "DIGIT"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("log", "modified_count")
    FUNCTION = "find_replace"
    OUTPUT_NODE = True
    DESCRIPTION = "Find and replace text across all .txt caption files in a folder."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Path to folder containing .txt caption files.",
                }),
                "find_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Text to search for in captions.",
                }),
                "replace_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Replacement text. Leave empty to delete the found text.",
                }),
                "case_sensitive": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Case-sensitive matching.",
                }),
                "dry_run": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Preview changes without writing files. Turn off to apply.",
                }),
            },
            "optional": {
                "prefix_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Text to prepend to ALL captions (applied after find/replace). Leave empty to skip.",
                }),
                "suffix_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Text to append to ALL captions (applied after find/replace). Leave empty to skip.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def find_replace(self, caption_folder, find_text, replace_text,
                     case_sensitive=True, dry_run=True,
                     prefix_text="", suffix_text=""):
        caption_folder = caption_folder.strip()
        if not os.path.isdir(caption_folder):
            raise ValueError(f"Folder not found: {caption_folder}")

        if not find_text and not prefix_text.strip() and not suffix_text.strip():
            raise ValueError("Provide find_text, prefix_text, or suffix_text.")

        # Find all .txt files
        txt_files = sorted([
            f for f in os.listdir(caption_folder)
            if f.lower().endswith(".txt")
        ])

        if not txt_files:
            return {"ui": {"log_text": ["No .txt files found."]},
                    "result": ("No .txt files found.", 0)}

        log_lines = []
        modified = 0
        total = len(txt_files)
        pbar = comfy.utils.ProgressBar(total)

        for idx, txt_file in enumerate(txt_files):
            txt_path = os.path.join(caption_folder, txt_file)

            with open(txt_path, "r", encoding="utf-8") as f:
                original = f.read()

            new_text = original

            # Find and replace
            if find_text:
                if case_sensitive:
                    new_text = new_text.replace(find_text, replace_text)
                else:
                    new_text = re.sub(
                        re.escape(find_text), replace_text, new_text,
                        flags=re.IGNORECASE,
                    )

            # Apply prefix/suffix
            if prefix_text.strip():
                if not new_text.startswith(prefix_text.strip()):
                    new_text = prefix_text.strip() + "\n\n" + new_text

            if suffix_text.strip():
                if not new_text.endswith(suffix_text.strip()):
                    new_text = new_text + "\n\n" + suffix_text.strip()

            if new_text != original:
                modified += 1
                # Show what changed
                if find_text:
                    if case_sensitive:
                        count = original.count(find_text)
                    else:
                        count = len(re.findall(re.escape(find_text), original, re.IGNORECASE))
                    status = f"[{idx + 1}/{total}] {txt_file} — {count} replacement(s)"
                else:
                    status = f"[{idx + 1}/{total}] {txt_file} — prefix/suffix added"

                if dry_run:
                    status += " (dry run)"
                else:
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(new_text)

                log_lines.append(status)

            pbar.update_absolute(idx + 1)

        mode = "DRY RUN" if dry_run else "APPLIED"
        summary = f"{mode}: {modified}/{total} files modified"
        if find_text:
            summary += f" | '{find_text}' -> '{replace_text}'"
        log_lines.append("")
        log_lines.append(summary)
        log_text = "\n".join(log_lines)

        logger.info("DIGIT Caption Find/Replace: %s", summary)

        return {"ui": {"log_text": [summary]},
                "result": (log_text, modified)}

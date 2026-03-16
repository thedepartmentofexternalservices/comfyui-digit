"""DIGIT Captioner — ComfyUI node for Gemini-based dataset captioning."""

import json
import os
import threading
from pathlib import Path


class DigitCaptioner:
    """Auto-caption training images using Google Gemini."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to dataset directory",
                }),
                "action": ([
                    "caption_all",
                    "caption_uncaptioned",
                    "caption_single",
                    "recaption_all",
                    "preview",
                ], {"default": "caption_uncaptioned"}),
            },
            "optional": {
                "caption_preset": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Name of saved caption preset",
                }),
                "system_prompt": ("STRING", {
                    "default": (
                        "You are an expert image captioner for AI training datasets. "
                        "Describe the image in detail, focusing on subject, composition, "
                        "lighting, colors, style, and mood. Be specific and descriptive. "
                        "Output only the caption, no preamble."
                    ),
                    "multiline": True,
                }),
                "prompt_template": ("STRING", {
                    "default": "Describe this image in detail for AI training:",
                    "multiline": True,
                }),
                "model": ([
                    "gemini-2.5-flash",
                    "gemini-2.5-pro",
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-lite",
                ], {"default": "gemini-2.5-flash"}),
                "temperature": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                "max_tokens": ("INT", {
                    "default": 300, "min": 50, "max": 2000, "step": 50,
                }),
                "overwrite": ("BOOLEAN", {"default": False}),
                "caption_ext": ("STRING", {"default": ".txt"}),
                "single_image_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to single image (for caption_single)",
                }),
                "gcp_project_id": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "GCP project ID (auto-detected if empty)",
                }),
                "gcp_region": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "GCP region (auto-detected if empty)",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT",)
    RETURN_NAMES = ("report", "last_caption", "captioned_count",)
    FUNCTION = "execute"
    CATEGORY = "DIGIT"

    def execute(self, dataset_path, action, caption_preset="",
                system_prompt="", prompt_template="", model="gemini-2.5-flash",
                temperature=0.4, max_tokens=300, overwrite=False,
                caption_ext=".txt", single_image_path="",
                gcp_project_id="", gcp_region=""):
        from .training.captioner import GeminiCaptioner
        from .training.presets_db import PresetsDB

        # Load preset if specified
        if caption_preset:
            db = PresetsDB()
            preset = db.get_caption_preset(caption_preset)
            if preset:
                system_prompt = preset["system_prompt"]
                prompt_template = preset["prompt_template"]
                model = preset["model"]
                temperature = preset["temperature"]
                max_tokens = preset["max_tokens"]

        # Create captioner
        captioner = GeminiCaptioner(
            model=model,
            system_prompt=system_prompt if system_prompt else None,
            prompt_template=prompt_template if prompt_template else None,
            temperature=temperature,
            max_tokens=max_tokens,
            gcp_project_id=gcp_project_id if gcp_project_id else None,
            gcp_region=gcp_region if gcp_region else None,
        )

        if action == "caption_single":
            return self._caption_single(captioner, single_image_path or dataset_path)
        elif action == "preview":
            return self._preview(captioner, dataset_path)
        elif action == "recaption_all":
            return self._caption_batch(captioner, dataset_path, caption_ext, overwrite=True)
        elif action == "caption_all":
            return self._caption_batch(captioner, dataset_path, caption_ext, overwrite=overwrite)
        else:  # caption_uncaptioned
            return self._caption_batch(captioner, dataset_path, caption_ext, overwrite=False)

    def _caption_single(self, captioner, image_path):
        """Caption a single image."""
        if not os.path.exists(image_path):
            return ("Image not found", "", 0)

        try:
            caption = captioner.caption_image(image_path)
            # Save caption file
            caption_path = Path(image_path).with_suffix(".txt")
            caption_path.write_text(caption, encoding="utf-8")

            report = f"Captioned: {os.path.basename(image_path)}\n{caption}"
            return (report, caption, 1)
        except Exception as e:
            return (f"Error: {e}", "", 0)

    def _preview(self, captioner, dataset_path):
        """Caption first 3 images as a preview without saving."""
        from .training.dataset import find_image_caption_pairs, IMAGE_EXTENSIONS

        if not os.path.exists(dataset_path):
            return ("Directory not found", "", 0)

        image_files = sorted(
            p for p in Path(dataset_path).rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )[:3]

        if not image_files:
            return ("No images found", "", 0)

        report_lines = ["Preview (not saved):\n"]
        last_caption = ""

        for img_path in image_files:
            try:
                caption = captioner.caption_image(str(img_path))
                last_caption = caption
                report_lines.append(f"{img_path.name}:")
                report_lines.append(f"  {caption}\n")
            except Exception as e:
                report_lines.append(f"{img_path.name}: ERROR - {e}\n")

        return ("\n".join(report_lines), last_caption, len(image_files))

    def _caption_batch(self, captioner, dataset_path, caption_ext, overwrite):
        """Caption all images in dataset."""
        if not os.path.exists(dataset_path):
            return ("Directory not found", "", 0)

        progress_log = []

        def on_progress(current, total, image_path, caption):
            progress_log.append(f"[{current}/{total}] {os.path.basename(image_path)}: {caption}")

        stats = captioner.caption_dataset(
            dataset_path=dataset_path,
            output_ext=caption_ext,
            overwrite=overwrite,
            progress_callback=on_progress,
        )

        report = (
            f"Captioning complete: {dataset_path}\n"
            f"Total images: {stats['total']}\n"
            f"Captioned: {stats['captioned']}\n"
            f"Skipped: {stats['skipped']}\n"
            f"Errors: {len(stats['errors'])}\n"
        )

        if stats["errors"]:
            report += "\nErrors:\n"
            for err in stats["errors"][:10]:
                report += f"  {err['image']}: {err['error']}\n"

        last_caption = ""
        if progress_log:
            last_entry = progress_log[-1]
            # Extract caption from last log entry
            parts = last_entry.split(": ", 1)
            if len(parts) > 1:
                last_caption = parts[1]

        return (report, last_caption, stats["captioned"])


class DigitCaptionPresetManager:
    """Save and load caption presets."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["save", "load", "list", "delete"], {"default": "save"}),
                "preset_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            },
            "optional": {
                "description": ("STRING", {"default": "", "multiline": False}),
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "prompt_template": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "model": ("STRING", {"default": "gemini-2.5-flash"}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0}),
                "max_tokens": ("INT", {"default": 300, "min": 50, "max": 2000}),
                "example_captions": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "One example caption per line",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = "DIGIT"

    def execute(self, action, preset_name, description="",
                system_prompt="", prompt_template="", model="gemini-2.5-flash",
                temperature=0.4, max_tokens=300, example_captions=""):
        from .training.presets_db import PresetsDB

        db = PresetsDB()

        if action == "save":
            if not preset_name:
                return ("Error: preset_name required",)
            examples = [
                line.strip() for line in example_captions.split("\n")
                if line.strip()
            ]
            db.save_caption_preset(
                name=preset_name,
                system_prompt=system_prompt,
                prompt_template=prompt_template,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                example_captions=examples,
                description=description,
            )
            return (f"Saved caption preset: {preset_name}",)

        elif action == "load":
            preset = db.get_caption_preset(preset_name)
            if not preset:
                return (f"Preset not found: {preset_name}",)
            return (json.dumps(preset, indent=2, default=str),)

        elif action == "list":
            presets = db.list_caption_presets()
            if not presets:
                return ("No caption presets saved.",)
            lines = ["Caption Presets:"]
            for p in presets:
                lines.append(f"  {p['name']} ({p['model']}) — {p['description']}")
            return ("\n".join(lines),)

        elif action == "delete":
            if db.delete_caption_preset(preset_name):
                return (f"Deleted: {preset_name}",)
            return (f"Not found: {preset_name}",)

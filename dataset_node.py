"""DIGIT Dataset Manager — ComfyUI node for managing training datasets."""

import json
import os
import shutil
from pathlib import Path

from .training.dataset import IMAGE_EXTENSIONS, find_image_caption_pairs

# Platform-aware base paths
if os.path.exists("/mnt/lucid"):
    _BASE_PATH = "/mnt/lucid"
elif os.path.exists("/Volumes/saint/goose"):
    _BASE_PATH = "/Volumes/saint/goose"
else:
    _BASE_PATH = os.path.expanduser("~/datasets")


class DigitDatasetManager:
    """Create and manage training datasets from image directories."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["scan", "create", "validate", "stats"], {"default": "scan"}),
                "dataset_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to dataset directory",
                }),
            },
            "optional": {
                "source_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Source images to copy into dataset",
                }),
                "dataset_name": ("STRING", {
                    "default": "my_dataset",
                    "multiline": False,
                }),
                "caption_ext": ("STRING", {"default": ".txt"}),
                "min_resolution": ("INT", {
                    "default": 512, "min": 64, "max": 4096, "step": 64,
                }),
                "copy_images": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT",)
    RETURN_NAMES = ("dataset_path", "report", "image_count",)
    FUNCTION = "execute"
    CATEGORY = "DIGIT"

    def execute(self, action, dataset_path, source_path="", dataset_name="my_dataset",
                caption_ext=".txt", min_resolution=512, copy_images=True):
        if action == "create":
            return self._create_dataset(
                dataset_path, source_path, dataset_name,
                min_resolution, copy_images,
            )
        elif action == "validate":
            return self._validate_dataset(dataset_path, caption_ext)
        elif action == "stats":
            return self._dataset_stats(dataset_path, caption_ext)
        else:  # scan
            return self._scan_dataset(dataset_path, caption_ext)

    def _scan_dataset(self, dataset_path, caption_ext):
        """Scan a directory and report what's there."""
        if not dataset_path or not os.path.exists(dataset_path):
            return (dataset_path, "Directory not found", 0)

        pairs = find_image_caption_pairs(dataset_path, caption_ext)
        captioned = sum(1 for _, cap in pairs if cap is not None)
        uncaptioned = len(pairs) - captioned

        report = (
            f"Dataset: {dataset_path}\n"
            f"Total images: {len(pairs)}\n"
            f"With captions: {captioned}\n"
            f"Without captions: {uncaptioned}\n"
        )

        if pairs:
            # Show first few
            report += "\nFirst 5 images:\n"
            for img, cap in pairs[:5]:
                cap_status = "captioned" if cap else "NO CAPTION"
                report += f"  {os.path.basename(img)} [{cap_status}]\n"

        return (dataset_path, report, len(pairs))

    def _create_dataset(self, dataset_path, source_path, dataset_name,
                        min_resolution, copy_images):
        """Create a new dataset from source images."""
        if not dataset_path:
            dataset_path = os.path.join(_BASE_PATH, "datasets", dataset_name)

        os.makedirs(dataset_path, exist_ok=True)

        copied = 0
        skipped = 0
        errors = []

        if source_path and os.path.exists(source_path):
            from PIL import Image

            source = Path(source_path)
            for img_file in sorted(source.rglob("*")):
                if img_file.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                try:
                    img = Image.open(img_file)
                    w, h = img.size
                    img.close()

                    if w < min_resolution and h < min_resolution:
                        skipped += 1
                        continue

                    dest = os.path.join(dataset_path, img_file.name)
                    if copy_images:
                        shutil.copy2(str(img_file), dest)
                    else:
                        os.symlink(str(img_file), dest)
                    copied += 1

                except Exception as e:
                    errors.append(f"{img_file.name}: {e}")

        report = (
            f"Dataset created: {dataset_path}\n"
            f"Images copied: {copied}\n"
            f"Skipped (too small): {skipped}\n"
        )
        if errors:
            report += f"Errors: {len(errors)}\n"
            for err in errors[:5]:
                report += f"  {err}\n"

        return (dataset_path, report, copied)

    def _validate_dataset(self, dataset_path, caption_ext):
        """Validate dataset integrity."""
        if not os.path.exists(dataset_path):
            return (dataset_path, "Directory not found", 0)

        from PIL import Image

        pairs = find_image_caption_pairs(dataset_path, caption_ext)
        issues = []
        valid = 0

        for img_path, cap_path in pairs:
            try:
                img = Image.open(img_path)
                img.verify()
                valid += 1
            except Exception as e:
                issues.append(f"Corrupt: {os.path.basename(img_path)}: {e}")

            if cap_path:
                caption = Path(cap_path).read_text(encoding="utf-8").strip()
                if not caption:
                    issues.append(f"Empty caption: {os.path.basename(cap_path)}")
                elif len(caption) < 10:
                    issues.append(f"Short caption ({len(caption)} chars): {os.path.basename(cap_path)}")

        report = f"Validation: {dataset_path}\n"
        report += f"Valid images: {valid}/{len(pairs)}\n"
        if issues:
            report += f"Issues ({len(issues)}):\n"
            for issue in issues[:20]:
                report += f"  {issue}\n"
        else:
            report += "No issues found.\n"

        return (dataset_path, report, valid)

    def _dataset_stats(self, dataset_path, caption_ext):
        """Get detailed statistics about a dataset."""
        if not os.path.exists(dataset_path):
            return (dataset_path, "Directory not found", 0)

        from PIL import Image
        import statistics

        pairs = find_image_caption_pairs(dataset_path, caption_ext)
        widths = []
        heights = []
        aspects = []
        caption_lengths = []

        for img_path, cap_path in pairs:
            try:
                img = Image.open(img_path)
                w, h = img.size
                img.close()
                widths.append(w)
                heights.append(h)
                aspects.append(w / h)
            except Exception:
                pass

            if cap_path and os.path.exists(cap_path):
                caption = Path(cap_path).read_text(encoding="utf-8").strip()
                caption_lengths.append(len(caption.split()))

        report = f"Dataset Statistics: {dataset_path}\n"
        report += f"Total images: {len(pairs)}\n\n"

        if widths:
            report += "Resolution:\n"
            report += f"  Width:  min={min(widths)}, max={max(widths)}, avg={statistics.mean(widths):.0f}\n"
            report += f"  Height: min={min(heights)}, max={max(heights)}, avg={statistics.mean(heights):.0f}\n"
            report += f"  Aspect: min={min(aspects):.2f}, max={max(aspects):.2f}, avg={statistics.mean(aspects):.2f}\n\n"

        if caption_lengths:
            report += "Captions:\n"
            report += f"  With captions: {len(caption_lengths)}/{len(pairs)}\n"
            report += f"  Words: min={min(caption_lengths)}, max={max(caption_lengths)}, avg={statistics.mean(caption_lengths):.0f}\n"
        else:
            report += "Captions: None found\n"

        return (dataset_path, report, len(pairs))

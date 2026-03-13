"""DIGIT Dataset Prep — resize and prepare images for LoRA training datasets."""

import logging
import os
import shutil

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

RESIZE_MODES = {
    "fit": "Resize to fit within target resolution (maintains aspect ratio, no cropping)",
    "fill_crop": "Resize to fill target resolution, center crop overflow",
    "stretch": "Stretch to exact target resolution (distorts aspect ratio)",
    "pad": "Resize to fit, pad remaining area with solid color",
}

OUTPUT_FORMATS = ["png", "jpg"]


def _process_image(img, target_size, mode, pad_color):
    """Resize a PIL Image according to the specified mode."""
    tw, th = target_size

    if mode == "stretch":
        return img.resize((tw, th), Image.LANCZOS)

    elif mode == "fit":
        img.thumbnail((tw, th), Image.LANCZOS)
        return img

    elif mode == "fill_crop":
        w, h = img.size
        scale = max(tw / w, th / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - tw) // 2
        top = (new_h - th) // 2
        return img.crop((left, top, left + tw, top + th))

    elif mode == "pad":
        img.thumbnail((tw, th), Image.LANCZOS)
        w, h = img.size
        padded = Image.new("RGB", (tw, th), pad_color)
        padded.paste(img, ((tw - w) // 2, (th - h) // 2))
        return padded

    return img


class DigitDatasetPrep:
    """Resize and prepare a folder of images for LoRA training."""

    CATEGORY = "DIGIT"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("log", "processed_count")
    FUNCTION = "prepare"
    OUTPUT_NODE = True
    DESCRIPTION = "Resize and prepare images in a folder for LoRA training. Outputs to a new folder."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Path to folder containing source images.",
                }),
                "output_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Path to output folder. Created if it doesn't exist.",
                }),
                "resolution": ("INT", {
                    "default": 1024, "min": 256, "max": 4096, "step": 64,
                    "tooltip": "Target resolution (used for both width and height).",
                }),
                "resize_mode": (list(RESIZE_MODES.keys()), {
                    "default": "fit",
                    "tooltip": "How to handle aspect ratio differences.",
                }),
                "output_format": (OUTPUT_FORMATS, {"default": "png"}),
                "quality": ("INT", {
                    "default": 95, "min": 1, "max": 100,
                    "tooltip": "JPEG quality (ignored for PNG).",
                }),
                "overwrite": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Overwrite existing files in output folder.",
                }),
            },
            "optional": {
                "copy_captions": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Copy existing .txt caption files to the output folder.",
                }),
                "pad_color_r": ("INT", {"default": 0, "min": 0, "max": 255, "tooltip": "Pad color red (pad mode only)."}),
                "pad_color_g": ("INT", {"default": 0, "min": 0, "max": 255, "tooltip": "Pad color green (pad mode only)."}),
                "pad_color_b": ("INT", {"default": 0, "min": 0, "max": 255, "tooltip": "Pad color blue (pad mode only)."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def prepare(self, source_folder, output_folder, resolution, resize_mode,
                output_format, quality, overwrite, copy_captions=True,
                pad_color_r=0, pad_color_g=0, pad_color_b=0):

        source_folder = source_folder.strip()
        output_folder = output_folder.strip()

        if not os.path.isdir(source_folder):
            raise ValueError(f"Source folder not found: {source_folder}")

        os.makedirs(output_folder, exist_ok=True)

        target_size = (resolution, resolution)
        pad_color = (pad_color_r, pad_color_g, pad_color_b)

        image_files = sorted([
            f for f in os.listdir(source_folder)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ])

        if not image_files:
            return {"ui": {"log_text": ["No images found."]},
                    "result": ("No images found.", 0)}

        log_lines = []
        processed = 0
        skipped = 0
        errors = 0

        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            out_name = f"{base_name}.{output_format}"
            out_path = os.path.join(output_folder, out_name)

            if os.path.isfile(out_path) and not overwrite:
                skipped += 1
                continue

            try:
                src_path = os.path.join(source_folder, img_file)
                img = Image.open(src_path).convert("RGB")
                orig_size = img.size

                img = _process_image(img, target_size, resize_mode, pad_color)

                save_kwargs = {"format": "PNG"} if output_format == "png" else {"format": "JPEG", "quality": quality}
                img.save(out_path, **save_kwargs)

                # Copy caption file if it exists
                if copy_captions:
                    txt_src = os.path.join(source_folder, base_name + ".txt")
                    txt_dst = os.path.join(output_folder, base_name + ".txt")
                    if os.path.isfile(txt_src):
                        shutil.copy2(txt_src, txt_dst)

                processed += 1
                log_lines.append(f"{img_file} ({orig_size[0]}x{orig_size[1]}) -> {out_name} ({img.size[0]}x{img.size[1]})")

            except Exception as e:
                errors += 1
                log_lines.append(f"{img_file} -> ERROR: {e}")
                logger.error("Dataset prep error: %s -> %s", img_file, e)

        summary = f"Done. Processed: {processed}, Skipped: {skipped}, Errors: {errors}, Total: {len(image_files)}"
        log_lines.append("")
        log_lines.append(summary)
        log_text = "\n".join(log_lines)

        logger.info("DIGIT Dataset Prep: %s", summary)

        return {"ui": {"log_text": [summary]},
                "result": (log_text, processed)}

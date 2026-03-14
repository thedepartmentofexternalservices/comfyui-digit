"""DIGIT Caption Viewer — step through image + caption pairs in a dataset folder."""

import logging
import os

import folder_paths
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


class DigitCaptionViewer:
    """Load and display image + caption pairs from a dataset folder for review."""

    CATEGORY = "DIGIT"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("image", "caption", "filename", "status", "total")
    FUNCTION = "view"
    OUTPUT_NODE = True
    DESCRIPTION = "Step through image + caption pairs in a dataset folder. Use index to navigate."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Path to dataset folder containing image + .txt pairs.",
                }),
                "index": ("INT", {
                    "default": 0, "min": 0, "max": 99999, "step": 1,
                    "tooltip": "Index of the image to view (0-based).",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def view(self, dataset_folder, index):
        dataset_folder = dataset_folder.strip()
        if not os.path.isdir(dataset_folder):
            raise ValueError(f"Dataset folder not found: {dataset_folder}")

        # Find all images that have caption pairs
        image_files = sorted([
            f for f in os.listdir(dataset_folder)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ])

        if not image_files:
            blank = torch.zeros(1, 64, 64, 3)
            return {"ui": {"viewer_text": ["No images found."]},
                    "result": (blank, "(no images)", "", "No images found.", 0)}

        total = len(image_files)
        index = index % total  # wrap around

        img_file = image_files[index]
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(dataset_folder, img_file)
        txt_path = os.path.join(dataset_folder, base_name + ".txt")

        # Load image
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        # Load caption
        if os.path.isfile(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
        else:
            caption = "(no caption file)"

        status = f"[{index + 1}/{total}] {img_file} ({img.size[0]}x{img.size[1]})"
        has_caption = "YES" if os.path.isfile(txt_path) else "NO"
        display = f"{status} | caption: {has_caption}"

        # Save preview to temp dir for on-node display
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        preview_name = f"digit_viewer_{base_name}.png"
        preview_path = os.path.join(temp_dir, preview_name)
        img.save(preview_path, format="PNG")

        return {"ui": {"images": [{"filename": preview_name, "subfolder": "", "type": "temp"}],
                        "viewer_text": [display],
                        "caption_text": [caption],
                        "filename_text": [img_file]},
                "result": (img_tensor, caption, img_file, status, total)}

"""DIGIT Batch Caption — caption a folder of images using Gemini via Vertex AI."""

import base64
import io
import logging
import os
import time

import comfy.utils
import numpy as np
import requests
from PIL import Image

from .llm_node import get_gcp_metadata, get_gcp_access_token

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

CAPTION_STYLES = {
    "descriptive_formal": (
        "Write a detailed, formal description of this image suitable for training an image generation model. "
        "Describe the subject, composition, lighting, camera angle, colors, textures, background, "
        "and overall aesthetic quality. Be thorough and precise."
    ),
    "descriptive_casual": (
        "Describe this image in a detailed but conversational way, as if explaining it to someone who can't see it. "
        "Cover what's in the image, the mood, lighting, colors, and any notable details."
    ),
    "training_detailed": (
        "Write a detailed caption for this image that would be used to train a diffusion model (LoRA/fine-tune). "
        "Include: subject description, camera equipment and settings (aperture, shutter speed, ISO), "
        "lighting conditions, composition techniques (rule of thirds, leading lines, symmetry), "
        "background details, textures, and aesthetic quality assessment. "
        "Write in a professional, descriptive tone."
    ),
    "training_concise": (
        "Write a concise caption for this image suitable for training a diffusion model. "
        "Describe the key subject, setting, lighting, and style in 1-3 sentences. Be specific but brief."
    ),
    "booru_tags": (
        "Generate a comma-separated list of descriptive tags for this image, in the style of image board tags. "
        "Include tags for: subject, action/pose, clothing, hair, setting, lighting, camera angle, "
        "art style, quality, and mood. Output only the comma-separated tags, nothing else."
    ),
    "prompt_style": (
        "Write a text-to-image prompt that would reproduce this image. "
        "Use natural language mixed with descriptive keywords. "
        "Focus on what a diffusion model needs to know to recreate this scene."
    ),
    "custom": "",
}

CAPTION_LENGTHS = {
    "short": "Keep the caption under 75 words.",
    "medium": "Keep the caption between 75-200 words.",
    "long": "Write a detailed caption of 200-500 words.",
    "any": "",
}


def _encode_image_file(filepath, max_dimension=2048):
    """Load and encode an image file to base64 PNG, resizing if needed."""
    img = Image.open(filepath).convert("RGB")
    w, h = img.size
    if max(w, h) > max_dimension:
        scale = max_dimension / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _call_gemini(project, region, model, token, image_b64, system_prompt, prompt,
                 max_tokens, temperature):
    """Send a single image+prompt to Gemini via Vertex AI."""
    parts = [
        {"inlineData": {"mimeType": "image/png", "data": image_b64}},
        {"text": prompt},
    ]
    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
    }
    if system_prompt:
        body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    url = (f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}"
           f"/locations/{region}/publishers/google/models/{model}:generateContent")

    for attempt in range(3):
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=body,
            timeout=120,
        )
        if resp.status_code == 429 or resp.status_code == 503:
            wait = 5 * (2 ** attempt)
            logger.warning("Rate limited (HTTP %d), retrying in %ds...", resp.status_code, wait)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

    resp.raise_for_status()
    return ""


class DigitBatchCaption:
    """Caption a folder of images using Gemini via Vertex AI. Saves .txt files alongside images."""

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
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("log", "captioned_count")
    FUNCTION = "caption_folder"
    OUTPUT_NODE = True
    DESCRIPTION = "Batch caption images in a folder using Gemini via Vertex AI. Saves .txt files next to each image."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Path to folder containing images to caption.",
                }),
                "model": (cls.MODELS, {"default": "gemini-2.5-flash"}),
                "caption_style": (list(CAPTION_STYLES.keys()), {"default": "training_detailed"}),
                "caption_length": (list(CAPTION_LENGTHS.keys()), {"default": "long"}),
                "overwrite": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Overwrite existing .txt caption files. If false, skips images that already have captions.",
                }),
            },
            "optional": {
                "trigger_word": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Trigger word prepended to each caption. Connect from LoRA Loader or type manually. Leave disconnected to skip.",
                }),
                "custom_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Custom prompt used when caption_style is 'custom'. Also appended as extra instructions for other styles.",
                }),
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional system prompt override. If empty, a default captioning system prompt is used.",
                }),
                "gcp_project_id": ("STRING", {
                    "default": "",
                    "tooltip": "GCP project ID. Auto-detected on GCP instances.",
                }),
                "gcp_region": ("STRING", {
                    "default": "",
                    "tooltip": "GCP region. Auto-detected on GCP instances.",
                }),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_dimension": ("INT", {
                    "default": 2048, "min": 512, "max": 4096, "step": 256,
                    "tooltip": "Resize images to this max dimension before sending to API (saves bandwidth).",
                }),
                "delay_seconds": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Delay between API calls to avoid rate limiting.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def caption_folder(self, image_folder, model, caption_style, caption_length,
                       trigger_word="", overwrite=False, custom_prompt="",
                       system_prompt="", gcp_project_id="", gcp_region="",
                       max_tokens=1024, temperature=0.4, max_dimension=2048,
                       delay_seconds=0.5):
        # Validate folder
        image_folder = image_folder.strip()
        if not os.path.isdir(image_folder):
            raise ValueError(f"Image folder not found: {image_folder}")

        # Resolve GCP config
        project = gcp_project_id.strip() or get_gcp_metadata("project/project-id")
        region = gcp_region.strip()
        if not region:
            zone = get_gcp_metadata("instance/zone")
            if zone:
                region = "-".join(zone.split("/")[-1].split("-")[:-1])
        if not project:
            raise ValueError("GCP project ID required. Set in node or run on GCP instance.")
        if not region:
            raise ValueError("GCP region required. Set in node or run on GCP instance.")

        # Build prompt
        style_prompt = CAPTION_STYLES.get(caption_style, "")
        if caption_style == "custom":
            if not custom_prompt.strip():
                raise ValueError("custom_prompt is required when caption_style is 'custom'.")
            style_prompt = custom_prompt.strip()
        elif custom_prompt.strip():
            style_prompt += f"\n\nAdditional instructions: {custom_prompt.strip()}"

        length_instruction = CAPTION_LENGTHS.get(caption_length, "")
        if length_instruction:
            style_prompt += f"\n\n{length_instruction}"

        if not system_prompt.strip():
            system_prompt = (
                "You are an expert image captioning model for AI training datasets. "
                "Produce accurate, detailed captions that faithfully describe what is in the image. "
                "Do not add information that is not visible. Do not include preamble or commentary — "
                "output only the caption text."
            )

        # Find images
        image_files = sorted([
            f for f in os.listdir(image_folder)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ])

        if not image_files:
            return {"ui": {"log_text": ["No images found in folder."]},
                    "result": ("No images found.", 0)}

        # Get token
        token = get_gcp_access_token()

        # Process images
        log_lines = []
        captioned = 0
        skipped = 0
        errors = 0
        total = len(image_files)
        pbar = comfy.utils.ProgressBar(total)

        for idx, img_file in enumerate(image_files):
            base_name = os.path.splitext(img_file)[0]
            txt_path = os.path.join(image_folder, base_name + ".txt")
            img_path = os.path.join(image_folder, img_file)

            # Skip if caption exists and overwrite is off
            if os.path.isfile(txt_path) and not overwrite:
                skipped += 1
                continue

            try:
                image_b64 = _encode_image_file(img_path, max_dimension)

                caption = _call_gemini(
                    project, region, model, token,
                    image_b64, system_prompt, style_prompt,
                    max_tokens, temperature,
                )

                # Clean up — remove markdown fences if Gemini wraps output
                caption = caption.strip()
                if caption.startswith("```"):
                    lines = caption.split("\n")
                    if lines[-1].strip() == "```":
                        lines = lines[1:-1]
                    elif lines[0].startswith("```"):
                        lines = lines[1:]
                    caption = "\n".join(lines).strip()

                # Prepend trigger word
                if trigger_word.strip():
                    caption = f"{trigger_word.strip()}\n\n{caption}"

                # Save caption
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)

                captioned += 1
                status = f"[{idx + 1}/{total}] {img_file} -> OK ({len(caption)} chars)"
                log_lines.append(status)
                logger.info("DIGIT Batch Caption: %s", status)

            except Exception as e:
                errors += 1
                status = f"[{idx + 1}/{total}] {img_file} -> ERROR: {e}"
                log_lines.append(status)
                logger.error("DIGIT Batch Caption: %s", status)

            pbar.update_absolute(idx + 1)

            # Rate limit delay
            if delay_seconds > 0 and idx < total - 1:
                time.sleep(delay_seconds)

        # Summary
        summary = f"Done. Captioned: {captioned}, Skipped: {skipped}, Errors: {errors}, Total: {total}"
        log_lines.append("")
        log_lines.append(summary)
        log_text = "\n".join(log_lines)

        logger.info("DIGIT Batch Caption: %s", summary)

        return {"ui": {"log_text": [summary]},
                "result": (log_text, captioned)}

"""DIGIT Batch Gemini Image — run Gemini image generation across a folder with LLM-driven prompt variation."""

import io
import logging
import os
import random
import time

import comfy.utils
import numpy as np
import torch
from PIL import Image

from .gcp_config import resolve_gcp_config, default_project, default_region

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

DEFAULT_IMAGE_SYSTEM_PROMPT = (
    "You are an expert image-generation engine. You must ALWAYS produce an image.\n"
    "Interpret all user input—regardless of format, intent, or abstraction—as literal visual directives for image composition.\n"
    "If a prompt is conversational or lacks specific visual details, you must creatively invent a concrete visual scenario that depicts the concept.\n"
    "Prioritize generating the visual representation above any text, formatting, or conversational requests."
)

DEFAULT_VARIATION_SYSTEM_PROMPT = (
    "You are a creative prompt engineer for image generation. "
    "Given a base prompt, produce a variation that preserves the core intent and subject "
    "but introduces creative differences in style, mood, lighting, composition, color palette, "
    "or artistic approach. Output ONLY the varied prompt text — no preamble, numbering, or commentary."
)

SAFETY_THRESHOLD_OPTIONS = [
    "BLOCK_NONE",
    "BLOCK_ONLY_HIGH",
    "BLOCK_MEDIUM_AND_ABOVE",
    "BLOCK_LOW_AND_ABOVE",
]

HARM_CATEGORIES = [
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY",
]


def _image_file_to_png_bytes(filepath, max_dimension=2048):
    """Load an image file and return PNG bytes, resizing if needed."""
    img = Image.open(filepath).convert("RGB")
    w, h = img.size
    if max(w, h) > max_dimension:
        scale = max_dimension / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _png_bytes_to_tensor(png_bytes):
    """Convert PNG bytes to a ComfyUI IMAGE tensor (1,H,W,3 float32 0-1)."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).unsqueeze(0)


def _save_image_tensor(tensor, path):
    """Save a single image tensor (H,W,C) to disk as PNG."""
    img_np = tensor.cpu().numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    img.save(path, format="PNG")


class DigitBatchGeminiImage:
    """Batch Gemini image generation: iterate over a folder of source images,
    use an LLM to vary the prompt, and generate multiple outputs per image."""

    IMAGE_MODELS = [
        "gemini-3.1-flash-image-preview",
        "gemini-3-pro-image-preview",
        "gemini-2.5-flash-image",
        "gemini-2.5-flash",
    ]

    LLM_MODELS = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
    ]

    ASPECT_RATIOS = [
        "auto", "1:1", "2:3", "3:2", "3:4", "4:1", "4:3",
        "4:5", "5:4", "8:1", "9:16", "16:9", "21:9",
    ]

    RESOLUTIONS = ["1K", "2K", "4K"]

    THINKING_LEVELS = ["MINIMAL", "HIGH"]

    CATEGORY = "DIGIT"
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "STRING")
    RETURN_NAMES = ("images", "log", "generated_count", "output_folder")
    FUNCTION = "generate_batch"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Batch Gemini image generation. Scans a folder of source images, "
        "uses an LLM to create prompt variations, and generates multiple "
        "Gemini image outputs per source image. Results saved to output subfolder."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Path to folder containing source images.",
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Base prompt for Gemini image generation. The LLM will create variations of this.",
                }),
                "variations_per_image": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 50,
                    "tooltip": "How many prompt variations (and image generations) per source image.",
                }),
                "image_model": (cls.IMAGE_MODELS, {"default": cls.IMAGE_MODELS[0]}),
                "llm_model": (cls.LLM_MODELS, {
                    "default": cls.LLM_MODELS[0],
                    "tooltip": "Gemini model used to generate prompt variations.",
                }),
                "aspect_ratio": (cls.ASPECT_RATIOS, {"default": "16:9"}),
                "resolution": (cls.RESOLUTIONS, {"default": "1K"}),
                "thinking_level": (cls.THINKING_LEVELS, {
                    "default": "MINIMAL",
                    "tooltip": "Thinking level for image generation. HIGH may improve quality.",
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Temperature for image generation.",
                }),
                "variation_temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Temperature for LLM prompt variation. Higher = more creative variations.",
                }),
                "gcp_project_id": ("STRING", {
                    "default": default_project(),
                    "tooltip": "GCP project ID.",
                }),
                "gcp_region": ("STRING", {
                    "default": default_region(),
                    "tooltip": "GCP region.",
                }),
            },
            "optional": {
                "output_subfolder": ("STRING", {
                    "default": "gemini_output",
                    "tooltip": "Subfolder name inside image_folder for outputs. Created automatically.",
                }),
                "system_instruction": ("STRING", {
                    "default": DEFAULT_IMAGE_SYSTEM_PROMPT,
                    "multiline": True,
                    "tooltip": "System instruction for Gemini image generation.",
                }),
                "variation_system_prompt": ("STRING", {
                    "default": DEFAULT_VARIATION_SYSTEM_PROMPT,
                    "multiline": True,
                    "tooltip": "System prompt for the LLM that generates prompt variations.",
                }),
                "variation_instruction": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Extra instructions appended to the variation request. E.g. 'Keep it photorealistic' or 'Vary the lighting dramatically'.",
                }),
"seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                    "tooltip": "Base seed. 0 = random. Each variation gets seed + variation_index.",
                }),
                "max_dimension": ("INT", {
                    "default": 2048,
                    "min": 512,
                    "max": 4096,
                    "step": 256,
                    "tooltip": "Resize source images to this max dimension before sending to API.",
                }),
                "delay_seconds": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.5,
                    "tooltip": "Delay between API calls to avoid rate limiting.",
                }),
                "harassment_threshold": (SAFETY_THRESHOLD_OPTIONS, {"default": "BLOCK_NONE"}),
                "hate_speech_threshold": (SAFETY_THRESHOLD_OPTIONS, {"default": "BLOCK_NONE"}),
                "sexually_explicit_threshold": (SAFETY_THRESHOLD_OPTIONS, {"default": "BLOCK_NONE"}),
                "dangerous_content_threshold": (SAFETY_THRESHOLD_OPTIONS, {"default": "BLOCK_NONE"}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def _build_safety_settings(self, types, harassment, hate_speech, sexually_explicit, dangerous_content):
        threshold_map = {
            "HARM_CATEGORY_HARASSMENT": harassment,
            "HARM_CATEGORY_HATE_SPEECH": hate_speech,
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": sexually_explicit,
            "HARM_CATEGORY_DANGEROUS_CONTENT": dangerous_content,
        }
        settings = []
        for category in HARM_CATEGORIES:
            threshold = threshold_map.get(category, harassment)
            settings.append(types.SafetySetting(
                category=category,
                threshold=threshold,
            ))
        return settings

    def _vary_prompt(self, client, llm_model, base_prompt, variation_index,
                     variation_system_prompt, variation_instruction, variation_temperature,
                     source_filename):
        """Use Gemini LLM to create a prompt variation."""
        from google.genai import types

        request_text = (
            f"Base prompt: {base_prompt}\n\n"
            f"Source image filename: {source_filename}\n"
            f"Variation number: {variation_index + 1}\n\n"
            f"Create a unique variation of this base prompt. "
            f"Make it distinctly different from what variations 1-{variation_index} might produce."
        )
        if variation_instruction.strip():
            request_text += f"\n\nAdditional guidance: {variation_instruction.strip()}"

        config = types.GenerateContentConfig(
            temperature=variation_temperature,
            max_output_tokens=1024,
            system_instruction=variation_system_prompt.strip() or None,
        )

        response = client.models.generate_content(
            model=llm_model,
            contents=request_text,
            config=config,
        )

        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text.strip()
        return base_prompt  # Fallback to original

    def _generate_image(self, client, image_model, parts, config, max_retries=3, base_delay=5.0):
        """Call Gemini image generation with retry on rate limits."""
        last_error = None
        for attempt in range(max_retries):
            try:
                return client.models.generate_content(
                    model=image_model,
                    contents=parts,
                    config=config,
                )
            except Exception as e:
                last_error = e
                error_str = str(e)
                if "429" in error_str or "503" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    delay = base_delay * (2 ** attempt)
                    logger.warning("Rate limited (attempt %d/%d), retrying in %ds: %s",
                                   attempt + 1, max_retries, delay, e)
                    time.sleep(delay)
                else:
                    raise
        raise last_error

    def generate_batch(
        self,
        image_folder,
        prompt,
        variations_per_image,
        image_model,
        llm_model,
        aspect_ratio,
        resolution,
        thinking_level,
        temperature,
        variation_temperature,
        gcp_project_id="",
        gcp_region="",
        output_subfolder="gemini_output",
        system_instruction="",
        variation_system_prompt="",
        variation_instruction="",
        seed=0,
        max_dimension=2048,
        delay_seconds=1.0,
        harassment_threshold="BLOCK_NONE",
        hate_speech_threshold="BLOCK_NONE",
        sexually_explicit_threshold="BLOCK_NONE",
        dangerous_content_threshold="BLOCK_NONE",
    ):
        from google import genai
        from google.genai import types

        logger.warning("=== DIGIT Batch Gemini Image: generate_batch called ===")
        logger.warning("  image_folder=%r, prompt=%r, variations=%d, image_model=%s, llm_model=%s",
                        image_folder, prompt[:80], variations_per_image, image_model, llm_model)
        print(f"\n=== DIGIT Batch Gemini Image START ===")
        print(f"  folder: {image_folder!r}")
        print(f"  prompt: {prompt[:80]!r}")
        print(f"  variations_per_image: {variations_per_image}")

        print("  >> entering try block...")
        import sys
        sys.stdout.flush()
        try:
            result = self._generate_batch_inner(
                genai, types,
                image_folder, prompt, variations_per_image, image_model, llm_model,
                aspect_ratio, resolution, thinking_level, temperature, variation_temperature,
                gcp_project_id, gcp_region, output_subfolder, system_instruction,
                variation_system_prompt, variation_instruction,
                seed, max_dimension, delay_seconds,
                harassment_threshold, hate_speech_threshold,
                sexually_explicit_threshold, dangerous_content_threshold,
            )
            print("  >> _generate_batch_inner returned successfully")
            sys.stdout.flush()
            return result
        except Exception as e:
            print(f"\n!!! DIGIT Batch Gemini Image FATAL ERROR !!!")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            raise

    def _generate_batch_inner(
        self,
        genai, types,
        image_folder, prompt, variations_per_image, image_model, llm_model,
        aspect_ratio, resolution, thinking_level, temperature, variation_temperature,
        gcp_project_id, gcp_region, output_subfolder, system_instruction,
        variation_system_prompt, variation_instruction,
        seed, max_dimension, delay_seconds,
        harassment_threshold, hate_speech_threshold,
        sexually_explicit_threshold, dangerous_content_threshold,
    ):
        print("  >> inside _generate_batch_inner")
        # Validate
        image_folder = image_folder.strip()
        print(f"  >> os.path.isdir({image_folder!r}) = {os.path.isdir(image_folder)}")
        if not os.path.isdir(image_folder):
            raise ValueError(f"Image folder not found: {image_folder}")
        if not prompt.strip():
            raise ValueError("Prompt is required.")

        project, region = resolve_gcp_config(gcp_project_id, gcp_region)

        client = genai.Client(
            vertexai=True,
            project=project,
            location=region,
        )

        # Create output folder
        output_dir = os.path.join(image_folder, output_subfolder.strip() or "gemini_output")
        os.makedirs(output_dir, exist_ok=True)

        # Find source images (recursive walk through subfolders)
        image_files = []
        for root, _dirs, files in os.walk(image_folder):
            # Skip the output subfolder
            if os.path.abspath(root).startswith(os.path.abspath(output_dir)):
                continue
            for f in files:
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                    image_files.append(os.path.join(root, f))
        image_files.sort()
        print(f"  >> Found {len(image_files)} images in {image_folder!r}")

        if not image_files:
            blank = torch.zeros((1, 512, 512, 3))
            return {"ui": {"log_text": ["No images found in folder."]},
                    "result": (blank, "No images found.", 0, output_dir)}

        # Collect generated image tensors for preview output
        output_tensors = []

        total_ops = len(image_files) * variations_per_image
        pbar = comfy.utils.ProgressBar(total_ops)

        safety_settings = self._build_safety_settings(
            types, harassment_threshold, hate_speech_threshold,
            sexually_explicit_threshold, dangerous_content_threshold,
        )

        log_lines = []
        generated = 0
        errors = 0
        op_idx = 0

        for img_idx, img_path in enumerate(image_files):
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]

            # Load source image bytes
            source_png_bytes = None
            try:
                source_png_bytes = _image_file_to_png_bytes(img_path, max_dimension)
            except Exception as e:
                log_lines.append(f"[{img_name}] ERROR loading image: {e}")
                logger.error("Failed to load %s: %s", img_name, e)
                errors += variations_per_image
                op_idx += variations_per_image
                pbar.update_absolute(op_idx)
                continue

            for var_idx in range(variations_per_image):
                op_idx += 1
                try:
                    # Step 1: Generate prompt variation via LLM
                    if variations_per_image == 1:
                        varied_prompt = prompt.strip()
                    else:
                        varied_prompt = self._vary_prompt(
                            client, llm_model, prompt, var_idx,
                            variation_system_prompt, variation_instruction,
                            variation_temperature, img_name,
                        )

                    # Step 2: Build image generation request
                    parts = []
                    if source_png_bytes:
                        parts.append(types.Part.from_bytes(data=source_png_bytes, mime_type="image/png"))
                    parts.append(types.Part.from_text(text=varied_prompt))

                    effective_seed = seed if seed > 0 else random.randint(1, 2147483647)
                    if seed > 0:
                        effective_seed = seed + (img_idx * variations_per_image) + var_idx

                    image_cfg_kwargs = {"image_size": resolution}
                    if aspect_ratio != "auto":
                        image_cfg_kwargs["aspect_ratio"] = aspect_ratio

                    config = types.GenerateContentConfig(
                        temperature=temperature,
                        seed=effective_seed if seed > 0 else None,
                        max_output_tokens=32768,
                        response_modalities=["TEXT", "IMAGE"],
                        image_config=types.ImageConfig(**image_cfg_kwargs),
                        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
                        system_instruction=system_instruction.strip() or None,
                        safety_settings=safety_settings,
                    )

                    # Step 3: Generate
                    response = self._generate_image(client, image_model, parts, config)

                    # Step 4: Extract and save output image
                    saved = False
                    if response.candidates:
                        for candidate in response.candidates:
                            if candidate.content and candidate.content.parts:
                                for part in candidate.content.parts:
                                    if part.inline_data and part.inline_data.mime_type and "image" in part.inline_data.mime_type:
                                        tensor = _png_bytes_to_tensor(part.inline_data.data)
                                        output_tensors.append(tensor)
                                        out_name = f"{base_name}_v{var_idx + 1:03d}.png"
                                        out_path = os.path.join(output_dir, out_name)
                                        _save_image_tensor(tensor[0], out_path)

                                        # Save the prompt used alongside the image
                                        prompt_path = os.path.join(output_dir, f"{base_name}_v{var_idx + 1:03d}.txt")
                                        with open(prompt_path, "w", encoding="utf-8") as f:
                                            f.write(varied_prompt)

                                        generated += 1
                                        saved = True
                                        status = f"[{op_idx}/{total_ops}] {img_name} v{var_idx + 1} -> {out_name}"
                                        log_lines.append(status)
                                        logger.info("Batch Gemini Image: %s", status)
                                        break
                            if saved:
                                break

                    if not saved:
                        errors += 1
                        status = f"[{op_idx}/{total_ops}] {img_name} v{var_idx + 1} -> NO IMAGE returned"
                        log_lines.append(status)
                        logger.warning("Batch Gemini Image: %s", status)

                except Exception as e:
                    errors += 1
                    status = f"[{op_idx}/{total_ops}] {img_name} v{var_idx + 1} -> ERROR: {e}"
                    log_lines.append(status)
                    logger.error("Batch Gemini Image: %s", status)

                pbar.update_absolute(op_idx)

                # Rate limit delay
                if delay_seconds > 0 and op_idx < total_ops:
                    time.sleep(delay_seconds)

        # Summary
        summary = (
            f"Done. Generated: {generated}, Errors: {errors}, "
            f"Source images: {len(image_files)}, Variations each: {variations_per_image}, "
            f"Total attempts: {total_ops}"
        )
        log_lines.append("")
        log_lines.append(summary)
        log_text = "\n".join(log_lines)

        logger.info("Batch Gemini Image: %s", summary)

        # Build image batch tensor — resize all to match first image dimensions
        if output_tensors:
            # All tensors are (1,H,W,3) — resize to common dimensions for batching
            target_h, target_w = output_tensors[0].shape[1], output_tensors[0].shape[2]
            resized = []
            for t in output_tensors:
                if t.shape[1] != target_h or t.shape[2] != target_w:
                    # Resize via PIL
                    img_np = (t[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    img = Image.fromarray(img_np).resize((target_w, target_h), Image.LANCZOS)
                    t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
                resized.append(t)
            image_batch = torch.cat(resized, dim=0)
        else:
            image_batch = torch.zeros((1, 512, 512, 3))

        return {"ui": {"log_text": [summary]},
                "result": (image_batch, log_text, generated, output_dir)}

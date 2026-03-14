import io
import logging
import random
import time

import numpy as np
import torch
from PIL import Image

from .llm_node import get_gcp_metadata

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_SYSTEM_PROMPT = (
    "You are an expert image-generation engine. You must ALWAYS produce an image.\n"
    "Interpret all user input—regardless of format, intent, or abstraction—as literal visual directives for image composition.\n"
    "If a prompt is conversational or lacks specific visual details, you must creatively invent a concrete visual scenario that depicts the concept.\n"
    "Prioritize generating the visual representation above any text, formatting, or conversational requests."
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


def _image_tensor_to_png_bytes(image_tensor):
    """Convert a single ComfyUI IMAGE tensor (H,W,C float32 0-1) to PNG bytes."""
    img_np = image_tensor.cpu().numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _png_bytes_to_tensor(png_bytes):
    """Convert PNG bytes to a ComfyUI IMAGE tensor (1,H,W,C float32 0-1)."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    img_np = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).unsqueeze(0)


class DigitGeminiImage:
    MODELS = [
        "gemini-3.1-flash-image-preview",
        "gemini-3-pro-image-preview",
        "gemini-2.5-flash-image",
        "gemini-2.5-flash",
    ]

    ASPECT_RATIOS = [
        "1:1", "2:3", "3:2", "3:4", "4:1", "4:3",
        "4:5", "5:4", "8:1", "9:16", "16:9", "21:9",
    ]

    RESOLUTIONS = ["1K", "2K", "4K"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (cls.MODELS, {"default": cls.MODELS[0]}),
                "aspect_ratio": (cls.ASPECT_RATIOS, {"default": "16:9"}),
                "resolution": (cls.RESOLUTIONS, {"default": "1K"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "gcp_project_id": ("STRING", {"default": "digit-sandbox", "tooltip": "GCP project ID. Auto-detected on GCP instances."}),
                "gcp_region": ("STRING", {"default": "global", "tooltip": "GCP region for Vertex AI. Use 'global' for default routing."}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "system_instruction": ("STRING", {"default": DEFAULT_IMAGE_SYSTEM_PROMPT, "multiline": True}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 32, "min": 1, "max": 64}),
                "harassment_threshold": (SAFETY_THRESHOLD_OPTIONS, {"default": "BLOCK_NONE"}),
                "hate_speech_threshold": (SAFETY_THRESHOLD_OPTIONS, {"default": "BLOCK_NONE"}),
                "sexually_explicit_threshold": (SAFETY_THRESHOLD_OPTIONS, {"default": "BLOCK_NONE"}),
                "dangerous_content_threshold": (SAFETY_THRESHOLD_OPTIONS, {"default": "BLOCK_NONE"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "text")
    FUNCTION = "generate"
    CATEGORY = "DIGIT"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        seed = kwargs.get("seed", 0)
        if seed == 0:
            return float("nan")
        return seed

    def _resolve_gcp_config(self, gcp_project_id, gcp_region):
        project = gcp_project_id.strip() if gcp_project_id else ""
        region = gcp_region.strip() if gcp_region else ""

        if not project:
            project = get_gcp_metadata("project/project-id")
        if not region:
            zone = get_gcp_metadata("instance/zone")
            if zone:
                zone_name = zone.split("/")[-1]
                region = "-".join(zone_name.split("-")[:-1])

        if not project:
            raise ValueError("GCP project ID is required. Set it in the node or run on a GCP instance.")
        if not region:
            region = "global"

        return project, region

    def _build_safety_settings(self, types, harassment, hate_speech, sexually_explicit, dangerous_content):
        """Build safety settings for all 8 harm categories."""
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

    def generate(
        self,
        prompt,
        model,
        aspect_ratio,
        resolution,
        seed,
        temperature,
        image1=None,
        image2=None,
        image3=None,
        system_instruction="",
        top_p=1.0,
        top_k=32,
        harassment_threshold="BLOCK_NONE",
        hate_speech_threshold="BLOCK_NONE",
        sexually_explicit_threshold="BLOCK_NONE",
        dangerous_content_threshold="BLOCK_NONE",
        gcp_project_id="",
        gcp_region="global",
    ):
        from google import genai
        from google.genai import types

        if not prompt:
            raise ValueError("Prompt is required")

        project, region = self._resolve_gcp_config(gcp_project_id, gcp_region)

        client = genai.Client(
            vertexai=True,
            project=project,
            location=region,
        )

        # Build content parts
        parts = []

        # Add input images
        for img_tensor in [image1, image2, image3]:
            if img_tensor is not None:
                for i in range(img_tensor.shape[0]):
                    png_bytes = _image_tensor_to_png_bytes(img_tensor[i])
                    parts.append(types.Part.from_bytes(data=png_bytes, mime_type="image/png"))

        # Add text prompt
        parts.append(types.Part.from_text(text=prompt))

        # Build config
        effective_seed = seed if seed > 0 else random.randint(1, 2147483647)

        safety_settings = self._build_safety_settings(
            types, harassment_threshold, hate_speech_threshold,
            sexually_explicit_threshold, dangerous_content_threshold,
        )

        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=effective_seed if seed > 0 else None,
            max_output_tokens=32768,
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=resolution,
            ),
            system_instruction=system_instruction.strip() or None,
            safety_settings=safety_settings,
        )

        # Generate with retry
        response = self._generate_with_retry(client, model, parts, config)

        # Parse response
        image_tensors = []
        text_parts = []

        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.inline_data and part.inline_data.mime_type and "image" in part.inline_data.mime_type:
                            tensor = _png_bytes_to_tensor(part.inline_data.data)
                            image_tensors.append(tensor)
                        elif part.text:
                            text_parts.append(part.text)

        if not image_tensors:
            # Return a blank 1024x1024 RGBA image as fallback
            logger.warning("Gemini returned no images. Returning blank fallback.")
            image_tensors.append(torch.zeros((1, 1024, 1024, 4)))

        output_image = torch.cat(image_tensors, dim=0)
        output_text = "\n".join(text_parts)

        return (output_image, output_text)

    def _generate_with_retry(self, client, model, parts, config, max_retries=3, base_delay=5.0):
        """Call generate_content with exponential backoff on 429/503."""
        last_error = None
        for attempt in range(max_retries):
            try:
                return client.models.generate_content(
                    model=model,
                    contents=parts,
                    config=config,
                )
            except Exception as e:
                last_error = e
                error_str = str(e)
                if "429" in error_str or "503" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Gemini API rate limited (attempt {attempt + 1}/{max_retries}), retrying in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    raise
        raise last_error

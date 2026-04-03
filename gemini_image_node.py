import base64
import io
import logging
import random
import time

import numpy as np
import requests as http_requests
import torch
from PIL import Image

from .gcp_config import resolve_gcp_config, get_gcp_access_token, build_vertex_url, default_project, default_region

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

# Image-specific safety categories (used by Nano Banana 2 / Google's nodes)
IMAGE_HARM_CATEGORIES = [
    "HARM_CATEGORY_IMAGE_HATE",
    "HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT",
    "HARM_CATEGORY_IMAGE_HARASSMENT",
    "HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT",
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
    """Convert PNG bytes to a ComfyUI IMAGE tensor (1,H,W,3 float32 0-1)."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
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
        "auto", "1:1", "2:3", "3:2", "3:4", "4:1", "4:3",
        "4:5", "5:4", "8:1", "9:16", "16:9", "21:9",
    ]

    THINKING_LEVELS = ["MINIMAL", "HIGH"]

    RESOLUTIONS = ["1K", "2K", "4K"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (cls.MODELS, {"default": cls.MODELS[0]}),
                "aspect_ratio": (cls.ASPECT_RATIOS, {"default": "16:9"}),
                "resolution": (cls.RESOLUTIONS, {"default": "1K"}),
                "thinking_level": (cls.THINKING_LEVELS, {"default": "MINIMAL", "tooltip": "Thinking level for image generation. HIGH may improve quality."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "gcp_project_id": ("STRING", {"default": default_project(), "tooltip": "GCP project ID. Auto-detected from DIGIT_GCP_PROJECT env var or GCP metadata."}),
                "gcp_region": ("STRING", {"default": default_region(), "tooltip": "GCP region. Auto-detected from DIGIT_GCP_REGION env var or GCP metadata. Defaults to 'global'."}),
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

    def _build_safety_settings(self, harassment, hate_speech, sexually_explicit, dangerous_content):
        """Build safety settings matching Nano Banana 2 format (text + image categories)."""
        settings = [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": hate_speech},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": dangerous_content},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": sexually_explicit},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": harassment},
            {"category": "HARM_CATEGORY_IMAGE_HATE", "threshold": hate_speech},
            {"category": "HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT", "threshold": dangerous_content},
            {"category": "HARM_CATEGORY_IMAGE_HARASSMENT", "threshold": harassment},
            {"category": "HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT", "threshold": sexually_explicit},
        ]
        return settings

    def generate(
        self,
        prompt,
        model,
        aspect_ratio,
        resolution,
        thinking_level,
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
        if not prompt:
            raise ValueError("Prompt is required")

        project, region = resolve_gcp_config(gcp_project_id, gcp_region)
        token = get_gcp_access_token()

        # Build content parts — text first, then images (matching Nano Banana 2 order)
        parts = [{"text": prompt}]

        for img_tensor in [image1, image2, image3]:
            if img_tensor is not None:
                for i in range(img_tensor.shape[0]):
                    png_bytes = _image_tensor_to_png_bytes(img_tensor[i])
                    b64 = base64.b64encode(png_bytes).decode("utf-8")
                    parts.append({"inlineData": {"mimeType": "image/png", "data": b64}})

        # Build imageConfig — matching Nano Banana 2 exactly
        image_config = {"imageSize": resolution}
        if aspect_ratio != "auto":
            image_config["aspectRatio"] = aspect_ratio

        # Build request body — matching Nano Banana 2 structure
        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": image_config,
                "thinkingConfig": {"thinkingLevel": thinking_level},
            },
            "safetySettings": self._build_safety_settings(
                harassment_threshold, hate_speech_threshold,
                sexually_explicit_threshold, dangerous_content_threshold,
            ),
        }

        if system_instruction and system_instruction.strip():
            body["systemInstruction"] = {"parts": [{"text": system_instruction.strip()}]}

        url = build_vertex_url(project, region, model)

        logger.warning("DIGIT Gemini Image config: model=%s, imageSize=%s, thinkingLevel=%s, aspect_ratio=%s",
                        model, resolution, thinking_level, aspect_ratio)
        logger.warning("DIGIT Gemini Image URL: %s", url)

        # Log the full request body (minus image data) for debugging
        import json as _json
        debug_body = _json.loads(_json.dumps(body))
        for content in debug_body.get("contents", []):
            for part in content.get("parts", []):
                if "inlineData" in part:
                    part["inlineData"]["data"] = f"<{len(part['inlineData']['data'])} chars>"
        logger.warning("DIGIT Gemini Image request body: %s", _json.dumps(debug_body, indent=2))

        # Generate with retry
        response_data = self._call_with_retry(url, token, body)

        # Log response metadata (not image data)
        debug_resp = {}
        if "usageMetadata" in response_data:
            debug_resp["usageMetadata"] = response_data["usageMetadata"]
        if "modelVersion" in response_data:
            debug_resp["modelVersion"] = response_data["modelVersion"]
        if "candidates" in response_data:
            debug_resp["candidateCount"] = len(response_data["candidates"])
            for ci, cand in enumerate(response_data["candidates"]):
                parts_summary = []
                for p in cand.get("content", {}).get("parts", []):
                    if "inlineData" in p:
                        parts_summary.append(f"image/{p['inlineData'].get('mimeType', '?')}")
                    elif "text" in p:
                        parts_summary.append(f"text({len(p['text'])} chars)")
                    elif "thought" in p:
                        parts_summary.append(f"thought({len(p.get('thought',''))} chars)")
                debug_resp[f"candidate_{ci}_parts"] = parts_summary
        logger.warning("DIGIT Gemini Image response metadata: %s", _json.dumps(debug_resp, indent=2))

        # Parse response
        image_tensors = []
        text_parts = []

        if "candidates" in response_data:
            for candidate in response_data["candidates"]:
                content = candidate.get("content", {})
                for part in content.get("parts", []):
                    if "inlineData" in part:
                        mime = part["inlineData"].get("mimeType", "")
                        if "image" in mime:
                            img_bytes = base64.b64decode(part["inlineData"]["data"])
                            tensor = _png_bytes_to_tensor(img_bytes)
                            image_tensors.append(tensor)
                    elif "text" in part:
                        text_parts.append(part["text"])

        if not image_tensors:
            logger.warning("Gemini returned no images. Returning blank fallback.")
            image_tensors.append(torch.zeros((1, 1024, 1024, 3)))

        output_image = image_tensors[0]
        logger.warning("DIGIT Gemini Image output: shape=%s (H=%d, W=%d)",
                        output_image.shape, output_image.shape[1], output_image.shape[2])
        output_text = "\n".join(text_parts)

        return (output_image, output_text)

    def _call_with_retry(self, url, token, body, max_retries=3, base_delay=5.0):
        """POST to Vertex AI with exponential backoff on rate limits."""
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = http_requests.post(
                    url,
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json=body,
                    timeout=300,
                )
                if resp.status_code in (429, 503):
                    delay = base_delay * (2 ** attempt)
                    logger.warning("Rate limited (HTTP %d, attempt %d/%d), retrying in %ds...",
                                   resp.status_code, attempt + 1, max_retries, delay)
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
                return resp.json()
            except http_requests.exceptions.HTTPError:
                raise
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

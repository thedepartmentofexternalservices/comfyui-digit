import base64
import io

import numpy as np
import requests
from PIL import Image

from .gcp_config import (
    get_gcp_access_token,
    build_vertex_url,
    resolve_gcp_config,
)


class LLMQueryNode:
    MODELS = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        # Preview models — may require allowlist or specific regions
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (cls.MODELS, {"default": cls.MODELS[0]}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "gcp_project_id": ("STRING", {"default": "", "tooltip": "GCP project ID. Auto-detected from DIGIT_GCP_PROJECT env var or GCP metadata."}),
                "gcp_region": ("STRING", {"default": "", "tooltip": "GCP region. Auto-detected from DIGIT_GCP_REGION env var or GCP metadata. Defaults to 'global'."}),
            },
            "optional": {
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "max_tokens": ("INT", {"default": 8192, "min": 1, "max": 65536}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "query"
    CATEGORY = "DIGIT"

    def query(self, model, prompt, gcp_project_id="", gcp_region="",
              system_prompt="", image=None, max_tokens=1024, temperature=0.7):
        if not prompt:
            raise ValueError("Prompt is required")

        project, region = resolve_gcp_config(gcp_project_id, gcp_region)
        image_b64 = self._encode_image(image) if image is not None else None

        token = get_gcp_access_token()

        parts = []
        if image_b64:
            parts.append({"inlineData": {"mimeType": "image/png", "data": image_b64}})
        parts.append({"text": prompt})

        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
        }
        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        url = build_vertex_url(project, region, model)

        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=body,
            timeout=120,
        )
        resp.raise_for_status()
        return (resp.json()["candidates"][0]["content"]["parts"][0]["text"],)

    def _encode_image(self, image_tensor):
        img_np = image_tensor[0].cpu().numpy()
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

import base64
import io

import numpy as np
import requests
from PIL import Image


def get_gcp_metadata(path):
    """Fetch metadata from GCP metadata service (works on Compute Engine/GKE)."""
    try:
        response = requests.get(
            f"http://metadata.google.internal/computeMetadata/v1/{path}",
            headers={"Metadata-Flavor": "Google"},
            timeout=5,
        )
        response.raise_for_status()
        return response.text.strip()
    except requests.exceptions.RequestException:
        return None


def get_gcp_access_token():
    """Get an access token from Application Default Credentials."""
    import google.auth
    import google.auth.transport.requests

    credentials, _ = google.auth.default()
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token


class LLMQueryNode:
    MODELS = [
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (cls.MODELS, {"default": cls.MODELS[0]}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "gcp_project_id": ("STRING", {"default": "", "tooltip": "GCP project ID. Auto-detected on GCP instances."}),
                "gcp_region": ("STRING", {"default": "", "tooltip": "GCP region (e.g. us-central1). Auto-detected on GCP instances."}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "query"
    CATEGORY = "DIGIT"

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
            raise ValueError("GCP region is required. Set it in the node or run on a GCP instance.")

        return project, region

    def query(self, model, prompt, gcp_project_id="", gcp_region="",
              system_prompt="", image=None, max_tokens=1024, temperature=0.7):
        if not prompt:
            raise ValueError("Prompt is required")

        project, region = self._resolve_gcp_config(gcp_project_id, gcp_region)
        image_b64 = self._encode_image(image) if image is not None else None

        token = get_gcp_access_token()

        parts = []
        if image_b64:
            parts.append({"inlineData": {"mimeType": "image/png", "data": image_b64}})
        parts.append({"text": prompt})

        body = {
            "contents": [{"parts": parts}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
        }
        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/google/models/{model}:generateContent"

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

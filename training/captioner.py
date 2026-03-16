"""Gemini-based auto-captioning for training datasets."""

import base64
import os
import time
from pathlib import Path
from typing import Optional

from .dataset import IMAGE_EXTENSIONS

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


class GeminiCaptioner:
    """Batch caption images using Google Gemini API."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        system_prompt: str = None,
        prompt_template: str = None,
        temperature: float = 0.4,
        max_tokens: int = 300,
        requests_per_minute: int = 30,
        gcp_project_id: str = None,
        gcp_region: str = None,
    ):
        if not HAS_GENAI:
            raise ImportError(
                "google-genai is required for captioning. "
                "Install with: pip install google-genai"
            )

        self.model_name = model
        self.system_prompt = system_prompt or (
            "You are an expert image captioner for AI training datasets. "
            "Describe the image in detail, focusing on subject, composition, "
            "lighting, colors, style, and mood. Be specific and descriptive. "
            "Output only the caption, no preamble."
        )
        self.prompt_template = prompt_template or "Describe this image in detail for AI training:"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.requests_per_minute = requests_per_minute
        self._min_interval = 60.0 / requests_per_minute
        self._last_request_time = 0

        # Initialize client
        self.client = genai.Client(
            vertexai=True,
            project=gcp_project_id or self._get_gcp_project(),
            location=gcp_region or self._get_gcp_region(),
        )

    def _get_gcp_project(self) -> str:
        """Get GCP project from metadata or environment."""
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT_ID")
        if project:
            return project
        try:
            import requests
            resp = requests.get(
                "http://metadata.google.internal/computeMetadata/v1/project/project-id",
                headers={"Metadata-Flavor": "Google"},
                timeout=2,
            )
            if resp.status_code == 200:
                return resp.text
        except Exception:
            pass
        raise ValueError(
            "Could not determine GCP project. Set GOOGLE_CLOUD_PROJECT env var."
        )

    def _get_gcp_region(self) -> str:
        """Get GCP region from metadata or environment."""
        region = os.environ.get("GOOGLE_CLOUD_REGION") or os.environ.get("GCP_REGION")
        if region:
            return region
        try:
            import requests
            resp = requests.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/zone",
                headers={"Metadata-Flavor": "Google"},
                timeout=2,
            )
            if resp.status_code == 200:
                zone = resp.text.split("/")[-1]
                return "-".join(zone.split("-")[:-1])
        except Exception:
            pass
        return "us-central1"

    def _rate_limit(self):
        """Simple rate limiter."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def caption_image(self, image_path: str, custom_prompt: str = None) -> str:
        """Caption a single image using Gemini."""
        self._rate_limit()

        # Read image
        with open(image_path, "rb") as f:
            image_data = f.read()

        # Determine MIME type
        ext = Path(image_path).suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
        }
        mime_type = mime_map.get(ext, "image/jpeg")

        prompt = custom_prompt or self.prompt_template

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=image_data, mime_type=mime_type),
                        types.Part.from_text(text=prompt),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )

        return response.text.strip()

    def caption_dataset(
        self,
        dataset_path: str,
        output_ext: str = ".txt",
        overwrite: bool = False,
        custom_prompt: str = None,
        progress_callback=None,
    ) -> dict:
        """Caption all images in a dataset directory.

        Args:
            dataset_path: Path to directory containing images.
            output_ext: Extension for caption files (default .txt).
            overwrite: Whether to overwrite existing captions.
            custom_prompt: Optional custom prompt to use instead of default.
            progress_callback: Optional callback(current, total, image_path, caption).

        Returns:
            Dict with stats: {total, captioned, skipped, errors}.
        """
        dataset_path = Path(dataset_path)
        image_files = sorted(
            p for p in dataset_path.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

        stats = {"total": len(image_files), "captioned": 0, "skipped": 0, "errors": []}

        for i, img_path in enumerate(image_files):
            caption_path = img_path.with_suffix(output_ext)

            # Skip if caption exists and not overwriting
            if caption_path.exists() and not overwrite:
                stats["skipped"] += 1
                if progress_callback:
                    existing = caption_path.read_text(encoding="utf-8").strip()
                    progress_callback(i + 1, len(image_files), str(img_path), f"[skipped] {existing[:80]}")
                continue

            try:
                caption = self.caption_image(str(img_path), custom_prompt)
                caption_path.write_text(caption, encoding="utf-8")
                stats["captioned"] += 1

                if progress_callback:
                    progress_callback(i + 1, len(image_files), str(img_path), caption[:80])

            except Exception as e:
                stats["errors"].append({"image": str(img_path), "error": str(e)})
                if progress_callback:
                    progress_callback(i + 1, len(image_files), str(img_path), f"[error] {e}")

        return stats

    def recaption_image(
        self,
        image_path: str,
        existing_caption: str,
        instruction: str = "Improve this caption to be more detailed and accurate:",
    ) -> str:
        """Recaption an image using its existing caption as context."""
        self._rate_limit()

        with open(image_path, "rb") as f:
            image_data = f.read()

        ext = Path(image_path).suffix.lower()
        mime_map = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(ext, "image/jpeg")

        prompt = f"{instruction}\n\nExisting caption: {existing_caption}"

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=image_data, mime_type=mime_type),
                        types.Part.from_text(text=prompt),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )

        return response.text.strip()

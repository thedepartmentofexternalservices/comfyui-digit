import json
import logging
import os
import re

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np

logger = logging.getLogger("DigitImageSaver")
from PIL import Image, PngImagePlugin
from aiohttp import web
from server import PromptServer

PROJEKTS_ROOTS = [
    "/Volumes/saint/goose/PROJEKTS",
    "/mnt/lucid/PROJEKTS",
]

PROJECT_RE = re.compile(r"^\d{5}_")
FRAME_RE = re.compile(r"\.(\d+)\.[^.]+$")


def sRGBtoLinear(npArray):
    """Convert sRGB gamma-encoded values to linear light."""
    less = npArray <= 0.04045
    result = np.where(less, npArray / 12.92, ((npArray + 0.055) / 1.055) ** 2.4)
    return result.astype(np.float32)


def scan_projects(projekts_root):
    """Return sorted list of project folders matching 5-digit prefix pattern."""
    if not os.path.isdir(projekts_root):
        return ["(no projects found)"]
    folders = [
        d for d in sorted(os.listdir(projekts_root))
        if os.path.isdir(os.path.join(projekts_root, d)) and PROJECT_RE.match(d)
    ]
    return folders if folders else ["(no projects found)"]


def scan_shots(projekts_root, project):
    """Return sorted list of shot folders inside <project>/shots/."""
    shots_dir = os.path.join(projekts_root, project, "shots")
    if not os.path.isdir(shots_dir):
        return ["(no shots found)"]
    folders = sorted(
        d for d in os.listdir(shots_dir)
        if os.path.isdir(os.path.join(shots_dir, d))
    )
    return folders if folders else ["(no shots found)"]


def next_frame(target_dir, prefix, shot, task, ext, start_frame, frame_pad):
    """Find highest existing frame number in target_dir and return next frame number."""
    # Match pattern: <prefix>_<shot>_<task>.<digits>.<ext>
    pat = re.compile(
        rf"^{re.escape(prefix)}_{re.escape(shot)}_{re.escape(task)}\.(\d+)\.{re.escape(ext)}$"
    )
    max_frame = start_frame - 1
    if os.path.isdir(target_dir):
        for f in os.listdir(target_dir):
            m = pat.match(f)
            if m:
                max_frame = max(max_frame, int(m.group(1)))
    return max_frame + 1


@PromptServer.instance.routes.get("/digit/projects")
async def get_projects(request):
    root = request.rel_url.query.get("root", "")
    projects = scan_projects(root)
    return web.json_response(projects)


@PromptServer.instance.routes.get("/digit/shots")
async def get_shots(request):
    root = request.rel_url.query.get("root", "")
    project = request.rel_url.query.get("project", "")
    shots = scan_shots(root, project)
    return web.json_response(shots)


class DigitImageSaver:
    CATEGORY = "DIGIT"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save_image"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        available_roots = [r for r in PROJEKTS_ROOTS if os.path.isdir(r)]
        if not available_roots:
            available_roots = PROJEKTS_ROOTS

        first_root = available_roots[0]
        projects = scan_projects(first_root)
        first_project = projects[0] if projects else ""
        shots = scan_shots(first_root, first_project)

        return {
            "required": {
                "image": ("IMAGE",),
                "projekts_root": (available_roots,),
                "project": (projects,),
                "shot": (shots,),
                "subfolder": ("STRING", {"default": "comfy"}),
                "task": ("STRING", {"default": "comp"}),
                "format": (["png", "jpg", "exr"],),
                "tonemap": (["linear", "sRGB", "Reinhard"],),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "start_frame": ("INT", {"default": 1001, "min": 0, "max": 99999999, "step": 1}),
                "frame_pad": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
                "show_preview": ("BOOLEAN", {"default": True}),
                "save_workflow": (["ui", "api", "ui + api", "none"],),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute so frame numbering increments each run.
        return float("nan")

    def save_image(self, image, projekts_root, project, shot, subfolder, task,
                   format, tonemap, quality, start_frame, frame_pad, show_preview,
                   save_workflow, prompt=None, extra_pnginfo=None):
        prefix = project[:5]
        target_dir = os.path.join(projekts_root, project, "shots", shot, subfolder, task)
        os.makedirs(target_dir, exist_ok=True)

        frame_num = next_frame(target_dir, prefix, shot, task, format, start_frame, frame_pad)

        metadata = {}
        if prompt is not None:
            metadata["prompt"] = prompt
        if extra_pnginfo is not None:
            for key in extra_pnginfo:
                metadata[key] = extra_pnginfo[key]

        ui_images = []
        last_filepath = ""
        batch_size = image.shape[0]

        for i in range(batch_size):
            current_frame = frame_num + i
            filename = f"{prefix}_{shot}_{task}.{current_frame:0{frame_pad}d}.{format}"
            filepath = os.path.join(target_dir, filename)

            img_np = image[i].cpu().numpy()

            try:
                if format == "png":
                    self._save_png(img_np, filepath, metadata)
                elif format == "jpg":
                    self._save_jpg(img_np, filepath, metadata, quality)
                elif format == "exr":
                    self._save_exr(img_np, filepath, tonemap)
                    # Save sidecar metadata for first frame only
                    if i == 0 and save_workflow != "none":
                        self._save_exr_sidecar(filepath, metadata, save_workflow)
            except Exception as e:
                logger.error(f"[DigitImageSaver] SAVE FAILED: {e}", exc_info=True)
                raise

            ui_images.append({"filename": filename, "subfolder": "", "type": "output"})
            last_filepath = filepath

        if not show_preview:
            ui_images = []

        return {"ui": {"images": ui_images, "filepath_text": [last_filepath]},
                "result": (last_filepath,)}

    def _save_png(self, img_np, filepath, metadata):
        channels = img_np.shape[2] if img_np.ndim == 3 else 1
        img_8bit = np.clip(255.0 * img_np, 0, 255).astype(np.uint8)
        if channels == 4:
            pil_img = Image.fromarray(img_8bit, mode="RGBA")
        else:
            pil_img = Image.fromarray(img_8bit[:, :, :3], mode="RGB")
        pnginfo = PngImagePlugin.PngInfo()
        for key, value in metadata.items():
            pnginfo.add_text(key, json.dumps(value))
        pil_img.save(filepath, format="PNG", pnginfo=pnginfo)

    def _save_jpg(self, img_np, filepath, metadata, quality):
        # JPEG doesn't support alpha — strip to RGB
        img_8bit = np.clip(255.0 * img_np[:, :, :3], 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_8bit, mode="RGB")

        exif_bytes = None
        if metadata:
            try:
                import piexif
                exif_dict = {"Exif": {piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(
                    json.dumps(metadata), encoding="unicode")}}
                exif_bytes = piexif.dump(exif_dict)
            except ImportError:
                pass

        save_kwargs = {"format": "JPEG", "quality": quality}
        if exif_bytes:
            save_kwargs["exif"] = exif_bytes
        pil_img.save(filepath, **save_kwargs)

    def _save_exr(self, img_np, filepath, tonemap):
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python (cv2) is required for EXR saving. "
                              "Install with: pip install opencv-python")

        img_float32 = img_np.astype(np.float32)
        channels = img_float32.shape[2] if img_float32.ndim == 3 else 1

        # Apply tone mapping to RGB channels only
        if tonemap == "sRGB":
            rgb = sRGBtoLinear(img_float32[:, :, :3])
        elif tonemap == "Reinhard":
            rgb = img_float32[:, :, :3]
            rgb = rgb / (1.0 + rgb)
        else:
            # linear — no transform
            rgb = img_float32[:, :, :3]

        if channels == 4:
            # Invert alpha per HQ convention
            alpha = 1.0 - img_float32[:, :, 3:4]
            rgba = np.concatenate([rgb, alpha], axis=2)
            img_bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            out = img_bgra
        else:
            out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        success = cv2.imwrite(filepath, out,
                              [int(cv2.IMWRITE_EXR_TYPE), int(cv2.IMWRITE_EXR_TYPE_FLOAT)])
        if not success:
            raise RuntimeError(f"cv2.imwrite failed to write EXR: {filepath}")

    def _save_exr_sidecar(self, filepath, metadata, save_workflow):
        """Save EXR metadata as JSON sidecar file(s)."""
        base = os.path.splitext(filepath)[0]

        prompt_data = metadata.get("prompt")
        workflow_data = metadata.get("workflow")

        if save_workflow in ("api", "ui + api") and prompt_data is not None:
            api_path = base + "_api.json"
            with open(api_path, "w") as f:
                json.dump(prompt_data, f, indent=2)

        if save_workflow in ("ui", "ui + api") and workflow_data is not None:
            ui_path = base + "_ui.json"
            with open(ui_path, "w") as f:
                json.dump(workflow_data, f, indent=2)

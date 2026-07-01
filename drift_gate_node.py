"""DIGIT Drift Gate — compare reference vs generated frames and reject on drift threshold."""

import json
import logging
import os

import folder_paths
import numpy as np
import torch
from PIL import Image

from .drift_compare import (
    build_qc_sheet_bgr,
    build_qc_sidecar,
    compare_image_tensors,
    resolve_qc_basename,
    save_qc_artifacts,
    _bgr_to_tensor,
)

logger = logging.getLogger(__name__)


def _load_path_to_tensor(filepath: str):
    if not filepath or not filepath.strip():
        raise ValueError("filepath is empty")
    filepath = filepath.strip()
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Image not found: {filepath}")

    img = Image.open(filepath).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).unsqueeze(0)


class DigitDriftGate:
    """Compare a reference plate to a generated image and gate on drift confidence.

    Designed for VFX / automotive QC: edge-sensitive SSIM catches badge, logo,
    and typography drift that global pixel metrics can miss.
    """

    CATEGORY = "DIGIT"
    RETURN_TYPES = (
        "BOOLEAN", "FLOAT", "STRING", "STRING", "STRING",
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
    )
    RETURN_NAMES = (
        "passed",
        "confidence",
        "drift_report",
        "qc_json",
        "qc_filepath",
        "gated_image",
        "qc_sheet",
        "diff_heatmap",
        "edge_diff",
        "reference_aligned",
        "generated_aligned",
    )
    FUNCTION = "compare"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Resize-align a reference and generated image, score drift confidence "
        "(pixel + edge/detail SSIM), and reject when below threshold."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference": ("IMAGE",),
                "generated": ("IMAGE",),
                "resize_mode": (
                    ["fit_reference", "fit_generated", "fit_largest", "custom"],
                    {"default": "fit_reference"},
                ),
                "fit_method": (
                    ["stretch", "letterbox", "crop_center"],
                    {"default": "stretch"},
                ),
                "compare_width": (
                    "INT",
                    {"default": 0, "min": 0, "max": 16384, "step": 8,
                     "tooltip": "Used when resize_mode is custom. 0 = ignored."},
                ),
                "compare_height": (
                    "INT",
                    {"default": 0, "min": 0, "max": 16384, "step": 8,
                     "tooltip": "Used when resize_mode is custom. 0 = ignored."},
                ),
                "grid_size": (
                    "INT",
                    {"default": 6, "min": 2, "max": 24, "step": 1,
                     "tooltip": "Grid resolution for hotspot drift report."},
                ),
                "detail_weight": (
                    "FLOAT",
                    {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05,
                     "tooltip": "How much edge/detail SSIM affects confidence (badge/logo sensitivity)."},
                ),
                "confidence_threshold": (
                    "FLOAT",
                    {"default": 75.0, "min": 0.0, "max": 100.0, "step": 0.5,
                     "tooltip": "Pass when confidence >= threshold."},
                ),
                "diff_threshold": (
                    "INT",
                    {"default": 30, "min": 1, "max": 255, "step": 1,
                     "tooltip": "Pixel delta treated as visible drift."},
                ),
                "reject_mode": (
                    ["blackout", "passthrough"],
                    {"default": "passthrough",
                     "tooltip": "blackout = gated_image goes black on reject. passthrough = gated_image always forwards generated. Use the passed output for branching."},
                ),
                "save_qc": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Write burn-in QC PNG + JSON sidecar to qc_output_dir."},
                ),
                "qc_output_dir": (
                    "STRING",
                    {"default": "",
                     "tooltip": "Folder for QC artifacts. Example: .../shots/dev_0010/qc"},
                ),
                "qc_filename": (
                    "STRING",
                    {"default": "",
                     "tooltip": "Optional base name (no extension, no _qc). Auto-derived from generated_path. Output is always <name>_qc.png/.json."},
                ),
                "show_qc_preview": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Show the burn-in QC sheet in ComfyUI's preview panel after run."},
                ),
            },
            "optional": {
                "reference_path": (
                    "STRING",
                    {"default": "", "tooltip": "Optional filesystem path instead of reference IMAGE input."},
                ),
                "generated_path": (
                    "STRING",
                    {"default": "", "tooltip": "Optional filesystem path instead of generated IMAGE input."},
                ),
            },
        }

    def compare(
        self,
        reference,
        generated,
        resize_mode,
        fit_method,
        compare_width,
        compare_height,
        grid_size,
        detail_weight,
        confidence_threshold,
        diff_threshold,
        reject_mode,
        save_qc,
        qc_output_dir,
        qc_filename,
        show_qc_preview,
        reference_path="",
        generated_path="",
    ):
        ref_path = reference_path.strip() if reference_path else ""
        gen_path = generated_path.strip() if generated_path else ""
        ref_tensor = _load_path_to_tensor(ref_path) if ref_path else reference
        gen_tensor = _load_path_to_tensor(gen_path) if gen_path else generated

        out = compare_image_tensors(
            ref_tensor,
            gen_tensor,
            resize_mode=resize_mode,
            fit_method=fit_method,
            compare_width=compare_width,
            compare_height=compare_height,
            grid_size=grid_size,
            detail_weight=detail_weight,
            confidence_threshold=confidence_threshold,
            diff_threshold=diff_threshold,
        )
        result = out["result"]

        gated = out["generated_aligned"]
        if not result.passed and reject_mode == "blackout":
            gated = torch.zeros_like(gated)

        qc_sheet_bgr = build_qc_sheet_bgr(
            result,
            out["reference_aligned_bgr"],
            out["generated_aligned_bgr"],
            out["diff_heatmap_bgr"],
            out["edge_diff_bgr"],
            reference_label="REFERENCE",
            generated_label="GENERATED",
        )
        sidecar = build_qc_sidecar(
            result,
            reference_path=ref_path,
            generated_path=gen_path,
            reject_mode=reject_mode,
        )
        qc_json = json.dumps(sidecar, indent=2)
        qc_filepath = ""

        qc_basename = resolve_qc_basename(gen_path, qc_filename)

        if save_qc:
            output_dir = qc_output_dir.strip()
            if not output_dir:
                raise ValueError("qc_output_dir is required when save_qc is enabled.")
            png_path, json_path = save_qc_artifacts(
                qc_sheet_bgr, sidecar, output_dir, qc_basename
            )
            qc_filepath = png_path
            sidecar["qc_sheet_path"] = png_path
            sidecar["qc_json_path"] = json_path
            sidecar["qc_basename"] = qc_basename
            qc_json = json.dumps(sidecar, indent=2)

        ui_images = []
        if show_qc_preview:
            temp_dir = folder_paths.get_temp_directory()
            os.makedirs(temp_dir, exist_ok=True)
            preview_name = f"digit_drift_{qc_basename}.png"
            preview_path = os.path.join(temp_dir, preview_name)
            rgb = qc_sheet_bgr[:, :, ::-1]
            Image.fromarray(rgb).save(preview_path, format="PNG")
            ui_images.append({"filename": preview_name, "subfolder": "", "type": "temp"})

        ui_payload = {
            "drift_report_text": [result.report],
            "confidence_text": [f"{result.confidence:.1f}%"],
            "verdict_text": ["PASS" if result.passed else "REJECT"],
            "qc_filepath_text": [qc_filepath or "(not saved — enable save_qc)"],
        }
        if ui_images:
            ui_payload["images"] = ui_images

        return {
            "ui": ui_payload,
            "result": (
                result.passed,
                result.confidence,
                result.report,
                qc_json,
                qc_filepath,
                gated,
                _bgr_to_tensor(qc_sheet_bgr),
                out["diff_heatmap"],
                out["edge_diff"],
                out["reference_aligned"],
                out["generated_aligned"],
            ),
        }


class DigitDriftGateFromPaths(DigitDriftGate):
    """Load reference + generated from filesystem paths (handy for QC outside the graph)."""

    DESCRIPTION = "Compare two filesystem image paths and gate on drift confidence."

    @classmethod
    def INPUT_TYPES(cls):
        base = DigitDriftGate.INPUT_TYPES()
        required = dict(base["required"])
        required.pop("reference")
        required.pop("generated")
        required["reference_path"] = (
            "STRING",
            {"default": "", "tooltip": "Absolute path to reference / plate image."},
        )
        required["generated_path"] = (
            "STRING",
            {"default": "", "tooltip": "Absolute path to generated image."},
        )
        return {"required": required}

    RETURN_TYPES = DigitDriftGate.RETURN_TYPES
    RETURN_NAMES = DigitDriftGate.RETURN_NAMES
    FUNCTION = "compare_paths"

    def compare_paths(self, reference_path, generated_path, **kwargs):
        blank = torch.zeros(1, 64, 64, 3)
        return self.compare(
            blank,
            blank,
            reference_path=reference_path,
            generated_path=generated_path,
            **kwargs,
        )

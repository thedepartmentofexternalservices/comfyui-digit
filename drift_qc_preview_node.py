"""DIGIT Drift QC Preview — interactive before/after review with hotspot annotations."""

import json
import logging
import os
import uuid

import folder_paths
import torch
from PIL import Image

from .drift_compare import (
    _bgr_to_tensor,
    _tensor_to_bgr,
    annotate_failure_hotspots,
    parse_hotspots_from_json,
)

logger = logging.getLogger(__name__)

LAYER_DEFS = [
    ("reference", "Reference"),
    ("generated", "Generated"),
    ("diff", "Pixel Diff"),
    ("edge", "Edge Diff"),
    ("annotated", "Annotated"),
]


def _save_temp_layer(tensor, layer_id: str, run_id: str) -> dict:
    temp_dir = folder_paths.get_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)
    filename = f"digit_qc_{run_id}_{layer_id}.png"

    img_np = tensor[0].cpu().numpy()
    img_np = (img_np * 255.0).clip(0, 255).astype("uint8")
    if img_np.shape[2] >= 3:
        img_np = img_np[:, :, :3]
    Image.fromarray(img_np, mode="RGB").save(os.path.join(temp_dir, filename), format="PNG")
    return {"id": layer_id, "filename": filename, "subfolder": "", "type": "temp"}


class DigitDriftQCPreview:
    """Interactive QC preview for drift gate outputs.

    Wire after DIGIT Drift Gate. Shows reference vs generated with A/B blink,
    layer cycling, and red marker circles on failed hotspot regions.
    """

    CATEGORY = "DIGIT"
    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING")
    RETURN_NAMES = ("annotated_image", "passed", "layer_manifest")
    FUNCTION = "preview"
    OUTPUT_NODE = True
    DESCRIPTION = "Interactive QC preview — cycle layers or A/B blink ref vs gen; circles drift on fail."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_aligned": ("IMAGE",),
                "generated_aligned": ("IMAGE",),
                "diff_heatmap": ("IMAGE",),
                "edge_diff": ("IMAGE",),
                "passed": ("BOOLEAN",),
                "qc_json": ("STRING", {"default": "", "forceInput": True}),
            },
            "optional": {
                "confidence": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
        }

    def preview(
        self,
        reference_aligned,
        generated_aligned,
        diff_heatmap,
        edge_diff,
        passed,
        qc_json,
        confidence=0.0,
    ):
        hotspots, _, _, grid_size = parse_hotspots_from_json(qc_json)
        gen_bgr = _tensor_to_bgr(generated_aligned)
        annotated_bgr = annotate_failure_hotspots(
            gen_bgr,
            hotspots,
            grid_size=grid_size,
            passed=bool(passed),
        )
        annotated_tensor = _bgr_to_tensor(annotated_bgr)

        run_id = uuid.uuid4().hex[:10]
        layers = {
            "reference": reference_aligned,
            "generated": generated_aligned,
            "diff": diff_heatmap,
            "edge": edge_diff,
            "annotated": annotated_tensor,
        }

        layer_manifest = []
        ui_layers = []
        for layer_id, _label in LAYER_DEFS:
            info = _save_temp_layer(layers[layer_id], layer_id, run_id)
            layer_manifest.append(info)
            ui_layers.append({**info, "label": _label})

        manifest_json = json.dumps({
            "layers": ui_layers,
            "passed": bool(passed),
            "confidence": confidence,
            "verdict": "PASS" if passed else "REJECT",
        })

        verdict = "PASS" if passed else "REJECT"

        ui_images = [
            {"filename": info["filename"], "subfolder": "", "type": "temp"}
            for info in layer_manifest
        ]

        return {
            "ui": {
                "qc_layers": [manifest_json],
                "layer_manifest_text": [manifest_json],
                "images": ui_images,
                "verdict_text": [verdict],
                "confidence_text": [f"{confidence:.1f}%"],
            },
            "result": (annotated_tensor, bool(passed), manifest_json),
        }

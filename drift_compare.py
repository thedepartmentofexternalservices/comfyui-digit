"""Pixel and edge-drift comparison utilities for DIGIT Drift Gate."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class HotspotCell:
    row: int
    col: int
    mean_diff: float
    pixel_ssim: float
    edge_ssim: float
    severity: str


@dataclass
class DriftResult:
    compare_width: int
    compare_height: int
    resize_mode: str
    fit_method: str
    pixel_ssim: float
    edge_ssim: float
    mean_abs_diff: float
    max_abs_diff: float
    drift_pixel_pct: float
    hotspot_mean_diff: float
    confidence: float
    passed: bool
    threshold: float
    grid_size: int
    hotspots: List[HotspotCell]
    annotation_hotspots: List[HotspotCell]
    report: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _tensor_to_bgr(image_tensor) -> np.ndarray:
    img_np = image_tensor[0].cpu().numpy()
    img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
    if img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)


def _bgr_to_tensor(img_bgr: np.ndarray):
    import torch

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).unsqueeze(0)


def _fit_image(img: np.ndarray, width: int, height: int, method: str) -> np.ndarray:
    h, w = img.shape[:2]
    if w == width and h == height:
        return img

    if method == "stretch":
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    scale = min(width / w, height / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if method == "letterbox":
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        y0 = (height - new_h) // 2
        x0 = (width - new_w) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas

    # crop_center: scale to cover, then center-crop
    scale = max(width / w, height / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = max(0, (new_w - width) // 2)
    y0 = max(0, (new_h - height) // 2)
    return resized[y0:y0 + height, x0:x0 + width]


def _resolve_compare_size(ref_shape, gen_shape, resize_mode: str, custom_w: int, custom_h: int) -> Tuple[int, int]:
    ref_h, ref_w = ref_shape[:2]
    gen_h, gen_w = gen_shape[:2]

    if resize_mode == "custom" and custom_w > 0 and custom_h > 0:
        return custom_w, custom_h
    if resize_mode == "fit_generated":
        return gen_w, gen_h
    if resize_mode == "fit_largest":
        return max(ref_w, gen_w), max(ref_h, gen_h)
    return ref_w, ref_h


def _ssim_map(img1: np.ndarray, img2: np.ndarray) -> Tuple[float, np.ndarray]:
    c1, c2 = 6.5025, 58.5225
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu12
    num = (2 * mu12 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
    ssim_map = num / den
    return float(ssim_map.mean()), ssim_map


def _edge_map(gray: np.ndarray) -> np.ndarray:
    return np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))


def _severity(mean_diff: float, pixel_ssim: float, edge_ssim: float) -> str:
    if mean_diff >= 100 or edge_ssim < 0.55 or pixel_ssim < 0.65:
        return "CRITICAL"
    if mean_diff >= 60 or edge_ssim < 0.70 or pixel_ssim < 0.78:
        return "HIGH"
    if mean_diff >= 35 or edge_ssim < 0.82 or pixel_ssim < 0.88:
        return "MODERATE"
    return "LOW"


def _grid_hotspots(
    diff: np.ndarray,
    ref_gray: np.ndarray,
    gen_gray: np.ndarray,
    grid_size: int,
    top_n: int = 8,
) -> List[HotspotCell]:
    h, w = diff.shape
    rows = cols = max(2, grid_size)
    cell_h = max(1, h // rows)
    cell_w = max(1, w // cols)
    cells: List[HotspotCell] = []

    for row in range(rows):
        for col in range(cols):
            y1, y2 = row * cell_h, min(h, (row + 1) * cell_h)
            x1, x2 = col * cell_w, min(w, (col + 1) * cell_w)
            d = diff[y1:y2, x1:x2]
            rg = ref_gray[y1:y2, x1:x2]
            gg = gen_gray[y1:y2, x1:x2]
            mean_diff = float(d.mean())
            pixel_ssim, _ = _ssim_map(rg, gg)
            edge_ssim, _ = _ssim_map(_edge_map(rg), _edge_map(gg))
            cells.append(
                HotspotCell(
                    row=row,
                    col=col,
                    mean_diff=round(mean_diff, 2),
                    pixel_ssim=round(pixel_ssim, 4),
                    edge_ssim=round(edge_ssim, 4),
                    severity=_severity(mean_diff, pixel_ssim, edge_ssim),
                )
            )

    cells.sort(key=lambda c: (c.mean_diff, -c.edge_ssim), reverse=True)
    return cells[:top_n], cells


def _confidence_score(
    pixel_ssim: float,
    edge_ssim: float,
    drift_pixel_pct: float,
    hotspot_mean_diff: float,
    detail_weight: float,
) -> float:
    detail_weight = float(np.clip(detail_weight, 0.0, 1.0))
    base = (1.0 - detail_weight) * pixel_ssim + detail_weight * edge_ssim
    drift_penalty = min(0.35, (drift_pixel_pct / 100.0) * 0.25)
    hotspot_penalty = min(0.25, hotspot_mean_diff / 255.0 * 0.35)
    score = max(0.0, base - drift_penalty - hotspot_penalty)
    return round(score * 100.0, 2)


def _build_report(result: DriftResult) -> str:
    lines = [
        "DIGIT DRIFT REPORT",
        "==================",
        f"Compare size: {result.compare_width}x{result.compare_height} ({result.resize_mode}, {result.fit_method})",
        "",
        "GLOBAL METRICS",
        f"  Pixel SSIM:      {result.pixel_ssim:.4f}  (structure / lighting)",
        f"  Edge SSIM:       {result.edge_ssim:.4f}  (badge, text, fine detail)",
        f"  Mean abs diff:   {result.mean_abs_diff:.1f} / 255",
        f"  Max abs diff:    {result.max_abs_diff:.1f} / 255",
        f"  Drift area:      {result.drift_pixel_pct:.1f}% pixels >30 delta",
        f"  Hotspot mean:    {result.hotspot_mean_diff:.1f} / 255 (worst grid cells)",
        "",
        f"CONFIDENCE: {result.confidence:.1f}%   THRESHOLD: {result.threshold:.1f}%",
        f"VERDICT: {'PASS' if result.passed else 'REJECT'}",
        "",
        "HOTSPOTS (worst regions first — check badges/logos here)",
    ]

    for idx, cell in enumerate(result.hotspots, start=1):
        lines.append(
            f"  {idx}. grid[{cell.row},{cell.col}]  "
            f"diff={cell.mean_diff:.1f}  edge_ssim={cell.edge_ssim:.3f}  "
            f"pixel_ssim={cell.pixel_ssim:.3f}  {cell.severity}"
        )

    if result.edge_ssim < 0.75:
        lines.extend([
            "",
            "DETAIL ALERT: Edge SSIM is low — likely badge/logo/text drift even if overall tone looks close.",
        ])

    return "\n".join(lines)


def compare_image_tensors(
    reference_tensor,
    generated_tensor,
    *,
    resize_mode: str = "fit_reference",
    fit_method: str = "stretch",
    compare_width: int = 0,
    compare_height: int = 0,
    grid_size: int = 6,
    detail_weight: float = 0.65,
    confidence_threshold: float = 75.0,
    diff_threshold: int = 30,
):
    ref_bgr = _tensor_to_bgr(reference_tensor)
    gen_bgr = _tensor_to_bgr(generated_tensor)

    width, height = _resolve_compare_size(
        ref_bgr.shape, gen_bgr.shape, resize_mode, compare_width, compare_height
    )
    ref_aligned = _fit_image(ref_bgr, width, height, fit_method)
    gen_aligned = _fit_image(gen_bgr, width, height, fit_method)

    ref_gray = cv2.cvtColor(ref_aligned, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gen_gray = cv2.cvtColor(gen_aligned, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = np.abs(ref_gray - gen_gray)

    pixel_ssim, _ = _ssim_map(ref_gray, gen_gray)
    edge_ssim, _ = _ssim_map(_edge_map(ref_gray), _edge_map(gen_gray))

    mean_abs_diff = float(diff.mean())
    max_abs_diff = float(diff.max())
    drift_pixel_pct = float((diff > diff_threshold).mean() * 100.0)

    hotspots, all_cells = _grid_hotspots(diff, ref_gray, gen_gray, grid_size)
    hotspot_mean_diff = float(np.mean([c.mean_diff for c in hotspots[:3]])) if hotspots else 0.0
    annotation_hotspots = sorted(all_cells, key=lambda c: c.mean_diff, reverse=True)[:12]

    confidence = _confidence_score(
        pixel_ssim, edge_ssim, drift_pixel_pct, hotspot_mean_diff, detail_weight
    )
    passed = confidence >= confidence_threshold

    result = DriftResult(
        compare_width=width,
        compare_height=height,
        resize_mode=resize_mode,
        fit_method=fit_method,
        pixel_ssim=round(pixel_ssim, 4),
        edge_ssim=round(edge_ssim, 4),
        mean_abs_diff=round(mean_abs_diff, 2),
        max_abs_diff=round(max_abs_diff, 2),
        drift_pixel_pct=round(drift_pixel_pct, 2),
        hotspot_mean_diff=round(hotspot_mean_diff, 2),
        confidence=confidence,
        passed=passed,
        threshold=confidence_threshold,
        grid_size=grid_size,
        hotspots=hotspots,
        annotation_hotspots=annotation_hotspots,
        report="",
    )
    result.report = _build_report(result)

    diff_norm = np.clip(diff / max(1.0, diff.max()) * 255.0, 0, 255).astype(np.uint8)
    diff_heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_TURBO)

    edge_diff = np.abs(_edge_map(ref_gray) - _edge_map(gen_gray))
    edge_norm = np.clip(edge_diff / max(1.0, edge_diff.max()) * 255.0, 0, 255).astype(np.uint8)
    edge_heatmap = cv2.applyColorMap(edge_norm, cv2.COLORMAP_MAGMA)

    overlay = gen_aligned.copy()
    mask = (diff > diff_threshold).astype(np.uint8)
    overlay[mask > 0] = (overlay[mask > 0] * 0.45 + diff_heatmap[mask > 0] * 0.55).astype(np.uint8)

    return {
        "result": result,
        "reference_aligned": _bgr_to_tensor(ref_aligned),
        "generated_aligned": _bgr_to_tensor(gen_aligned),
        "diff_heatmap": _bgr_to_tensor(diff_heatmap),
        "edge_diff": _bgr_to_tensor(edge_heatmap),
        "drift_overlay": _bgr_to_tensor(overlay),
        "reference_aligned_bgr": ref_aligned,
        "generated_aligned_bgr": gen_aligned,
        "diff_heatmap_bgr": diff_heatmap,
        "edge_diff_bgr": edge_heatmap,
    }


def _label_panel(img_bgr: np.ndarray, label: str, panel_h: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = panel_h / max(1, h)
    panel_w = max(1, int(round(w * scale)))
    resized = cv2.resize(img_bgr, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    banner = np.zeros((28, panel_w, 3), dtype=np.uint8)
    pil = Image.fromarray(cv2.cvtColor(banner, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text((8, 6), label, fill=(220, 220, 220))
    banner = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return np.vstack([banner, resized])


def build_qc_sheet_bgr(
    result: DriftResult,
    ref_bgr: np.ndarray,
    gen_bgr: np.ndarray,
    diff_bgr: np.ndarray,
    edge_bgr: np.ndarray,
    *,
    reference_label: str = "REFERENCE",
    generated_label: str = "GENERATED",
    panel_height: int = 360,
) -> np.ndarray:
    """Build a review sheet with side-by-side panels and burn-in QC metadata."""
    panels = [
        _label_panel(ref_bgr, reference_label, panel_height),
        _label_panel(gen_bgr, generated_label, panel_height),
        _label_panel(diff_bgr, "PIXEL DIFF", panel_height),
        _label_panel(edge_bgr, "EDGE DIFF", panel_height),
    ]
    panel_row = np.hstack(panels)
    sheet_w = panel_row.shape[1]

    verdict = "PASS" if result.passed else "REJECT"
    verdict_color = (70, 180, 90) if result.passed else (220, 70, 70)

    header_h = 72
    header = Image.new("RGB", (sheet_w, header_h), (18, 18, 22))
    draw = ImageDraw.Draw(header)
    draw.rectangle((0, 0, sheet_w, header_h), fill=(18, 18, 22))
    draw.text((20, 14), f"DIGIT DRIFT QC — {verdict}", fill=verdict_color)
    draw.text(
        (20, 40),
        f"Confidence {result.confidence:.1f}%  |  Threshold {result.threshold:.1f}%  |  "
        f"Pixel SSIM {result.pixel_ssim:.3f}  |  Edge SSIM {result.edge_ssim:.3f}",
        fill=(210, 210, 210),
    )

    footer_lines = [
        f"Compare: {result.compare_width}x{result.compare_height} ({result.resize_mode}, {result.fit_method})",
        f"Mean diff {result.mean_abs_diff:.1f}/255  |  Drift area {result.drift_pixel_pct:.1f}%  |  "
        f"Hotspot mean {result.hotspot_mean_diff:.1f}/255",
        "Top hotspots: " + ", ".join(
            f"[{c.row},{c.col}] {c.severity}" for c in result.hotspots[:4]
        ),
        datetime.now(timezone.utc).strftime("UTC %Y-%m-%d %H:%M:%S"),
    ]
    footer_h = 28 + len(footer_lines) * 22
    footer = Image.new("RGB", (sheet_w, footer_h), (12, 12, 16))
    fdraw = ImageDraw.Draw(footer)
    y = 10
    for line in footer_lines:
        fdraw.text((20, y), line, fill=(185, 185, 195))
        y += 22

    header_bgr = cv2.cvtColor(np.array(header), cv2.COLOR_RGB2BGR)
    footer_bgr = cv2.cvtColor(np.array(footer), cv2.COLOR_RGB2BGR)
    return np.vstack([header_bgr, panel_row, footer_bgr])


def build_qc_sidecar(
    result: DriftResult,
    *,
    reference_path: str = "",
    generated_path: str = "",
    reject_mode: str = "",
) -> dict:
    payload = asdict(result)
    payload["verdict"] = "PASS" if result.passed else "REJECT"
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    payload["reference_path"] = reference_path
    payload["generated_path"] = generated_path
    payload["reject_mode"] = reject_mode
    return payload


def _hotspot_cell_rect(row: int, col: int, grid_size: int, width: int, height: int) -> Tuple[int, int, int, int]:
    rows = cols = max(2, grid_size)
    cell_h = max(1, height // rows)
    cell_w = max(1, width // cols)
    x1 = col * cell_w
    y1 = row * cell_h
    x2 = min(width, (col + 1) * cell_w)
    y2 = min(height, (row + 1) * cell_h)
    return x1, y1, x2, y2


def annotate_failure_hotspots(
    image_bgr: np.ndarray,
    hotspots: List[HotspotCell],
    *,
    grid_size: int,
    passed: bool,
    min_severity: str = "MODERATE",
) -> np.ndarray:
    """Draw red marker circles on drift hotspots when QC fails."""
    if passed or not hotspots:
        return image_bgr.copy()

    severity_rank = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}
    min_rank = severity_rank.get(min_severity, 1)
    h, w = image_bgr.shape[:2]
    out = image_bgr.copy()
    pad = 6

    for idx, cell in enumerate(hotspots, start=1):
        if severity_rank.get(cell.severity, 0) < min_rank:
            continue
        x1, y1, x2, y2 = _hotspot_cell_rect(cell.row, cell.col, grid_size, w, h)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        rx = max(12, (x2 - x1) // 2 + pad)
        ry = max(12, (y2 - y1) // 2 + pad)

        # Marker-style double stroke
        cv2.ellipse(out, (cx, cy), (rx, ry), 0, 0, 360, (0, 0, 220), 4, cv2.LINE_AA)
        cv2.ellipse(out, (cx, cy), (rx - 3, ry - 3), 0, 0, 360, (0, 0, 255), 2, cv2.LINE_AA)

        label = str(idx)
        cv2.putText(
            out, label, (cx - 8, cy + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA,
        )

    return out


def _hotspots_from_payload(items: list) -> List[HotspotCell]:
    return [
        HotspotCell(
            row=h["row"],
            col=h["col"],
            mean_diff=h["mean_diff"],
            pixel_ssim=h["pixel_ssim"],
            edge_ssim=h["edge_ssim"],
            severity=h["severity"],
        )
        for h in (items or [])
    ]


def parse_hotspots_from_json(qc_json: str) -> Tuple[List[HotspotCell], int, int, int]:
    """Parse annotation hotspots and compare dimensions from drift gate qc_json."""
    if not qc_json or not qc_json.strip():
        return [], 0, 0, 6

    data = json.loads(qc_json)
    hotspots = _hotspots_from_payload(
        data.get("annotation_hotspots") or data.get("hotspots", [])
    )
    width = int(data.get("compare_width", 0) or 0)
    height = int(data.get("compare_height", 0) or 0)
    grid_size = int(data.get("grid_size", 6) or 6)
    return hotspots, width, height, grid_size


def resolve_qc_basename(generated_path: str = "", qc_filename: str = "") -> str:
    """Return filename stem ending in _qc (suffix style, not prefix)."""
    if qc_filename and qc_filename.strip():
        stem = qc_filename.strip()
        if stem.lower().endswith("_qc"):
            return stem
        return f"{stem}_qc"

    if generated_path and generated_path.strip():
        base = os.path.splitext(os.path.basename(generated_path.strip()))[0]
        if base.lower().endswith("_qc"):
            return base
        return f"{base}_qc"

    return "drift_qc"


def save_qc_artifacts(
    qc_sheet_bgr: np.ndarray,
    sidecar: dict,
    output_dir: str,
    filename_stem: str,
) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    if not filename_stem.lower().endswith("_qc"):
        filename_stem = f"{filename_stem}_qc"
    png_path = os.path.join(output_dir, f"{filename_stem}.png")
    json_path = os.path.join(output_dir, f"{filename_stem}.json")

    rgb = cv2.cvtColor(qc_sheet_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(png_path, format="PNG")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    return png_path, json_path

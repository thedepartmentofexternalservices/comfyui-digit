# Test Coverage Analysis — comfyui-digit

## Current State

**Test coverage: 0%.** The codebase has no test files, no test configuration, and no CI step that runs tests. The publish workflow (`publish.yml`) triggers on `pyproject.toml` changes and pushes directly to the ComfyUI Registry with zero validation.

---

## Proposed Test Plan

The recommendations below are ordered by **impact vs. effort** — the first items give you the most safety for the least work.

### Priority 1 — Pure utility functions (no external dependencies)

These functions have zero side effects and can be tested immediately with no mocking.

| Function | File | What to test |
|---|---|---|
| `sRGBtoLinear()` | `image_saver_node.py:24` | Known sRGB→linear conversion values (0.0→0.0, 1.0→1.0, 0.5→0.214), array shapes preserved, float32 output dtype |
| `_seconds_to_srt_time()` | `srt_maker_node.py:136` | Boundary values (0, 59.999, 3600, 86399.999), correct `HH:MM:SS,mmm` formatting |
| `_image_tensor_to_png_bytes()` | `gemini_image_node.py:37` | Round-trip: tensor → PNG bytes → tensor preserves shape, values clamp to 0–255, returns `bytes` |
| `_png_bytes_to_tensor()` | `gemini_image_node.py:47` | Output shape is `(1,H,W,4)` (RGBA), dtype float32, values in 0–1 range |
| `_encode_image()` | `llm_node.py:115` | Returns valid base64 string, decoded bytes are a valid PNG |
| `_HTMLTextExtractor` | `srt_maker_node.py:47` | Strips `<script>`/`<style>`, converts block tags to newlines, preserves plain text |
| `DigitCropInfo.interpret()` | `drag_crop_node.py:183` | Valid JSON, empty string, malformed JSON, missing keys fallback to defaults, CSV/pretty format correctness |
| `next_frame()` | `image_saver_node.py:54` | Empty directory → returns `start_frame`, existing frames → returns max+1, non-matching files ignored |
| `scan_projects()` | `image_saver_node.py:31` | Matches `12345_*` folders, ignores non-matching, returns fallback on missing dir |
| `scan_shots()` | `image_saver_node.py:42` | Returns sorted subdirectories, returns fallback on missing dir |

**Estimated effort:** ~2 hours. **Value:** Covers the core data-transformation logic that silently corrupts output if broken.

---

### Priority 2 — Node logic with mockable I/O

These test the core business logic of each node by mocking external services (GCP APIs, filesystem).

| Area | File | What to test |
|---|---|---|
| `_resolve_gcp_config()` | `llm_node.py:63`, `gemini_image_node.py:107`, `veo_video_node.py:82` | Explicit values used as-is; fallback to metadata service; raises `ValueError` when both are empty and metadata returns `None`; zone → region parsing (`us-central1-a` → `us-central1`) |
| `DigitDragCrop.crop()` | `drag_crop_node.py:38` | Resolution change resets crop; out-of-bounds crop resets to full image; mask dimension broadcasting (2D/3D/4D input); batch cropping shape correctness; `CROP_JSON` payload matches expected schema |
| `DigitImageSaver._save_png/jpg/exr()` | `image_saver_node.py:183–247` | PNG: metadata roundtrip via `PngInfo`; JPG: EXIF UserComment present; EXR: correct `cv2.imwrite` calls, alpha inversion, tone mapping (sRGB/Reinhard/linear produce different outputs) |
| `DigitImageSaver.save_image()` | `image_saver_node.py:133` | Directory creation, batch frame numbering (3-image batch starts at correct frame and increments), `show_preview=False` returns empty `ui_images` |
| `DigitImageLoader._find_latest()` | `image_loader_node.py:109` | Finds highest frame number, ignores non-matching files, returns `(None, 0)` for empty dir |
| `DigitImageLoader._load_image()` | `image_loader_node.py:130` | RGBA preserved as 4-channel tensor, RGB images get 3 channels, output shape `(1,H,W,C)` |
| `DigitVideoSaver.save_video()` | `video_saver_node.py:103` | `video_paths` list mode copies files correctly, single `VIDEO` mode works, raises on no input, sidecar files written when `save_workflow != "none"` |
| `_generate_with_retry()` | `gemini_image_node.py:241`, `veo_video_node.py:248` | Retries on 429/503/RESOURCE_EXHAUSTED, raises immediately on other errors, respects `max_retries`, exponential delay |
| SRT markdown fence stripping | `srt_maker_node.py:316–323` | Strips `` ```srt `` and trailing `` ``` ``, leaves clean SRT intact |

**Estimated effort:** ~4 hours. **Value:** Catches regressions in file I/O, crop math, and retry logic — the parts most likely to break across ComfyUI version updates.

---

### Priority 3 — Integration-style tests

These require heavier fixtures or optional external credentials.

| Area | What to test |
|---|---|
| **PNG/JPG/EXR round-trip** | Save an image tensor → load it back → pixel values match within tolerance. Verifies the Saver↔Loader pipeline end-to-end. |
| **VFX path construction** | Given project `12345_test`, shot `sh010`, task `comp`, verify the full output path `PROJEKTS/12345_test/shots/sh010/comfy/comp/12345_sh010_comp.1001.png` |
| **Batch video save** | Create 3 temp MP4 files → `save_video()` with `video_paths` → verify 3 sequential files created with correct naming |
| **EXR alpha inversion round-trip** | Save RGBA EXR (alpha inverted on write) → load (alpha un-inverted on read) → alpha values match original |
| **Drag crop + CropInfo pipeline** | Crop an image → pass `CROP_JSON` to `DigitCropInfo.interpret()` → verify extracted values match |

**Estimated effort:** ~3 hours. **Value:** Validates the end-to-end VFX pipeline that is the product's core differentiator.

---

### Priority 4 — JavaScript frontend tests

The `web/drag_crop/` module has well-isolated utility files that are testable with a lightweight runner (Vitest or Jest):

| File | What to test |
|---|---|
| `utils/mathUtils.js` | Clamping, rounding, interpolation |
| `utils/geometryUtils.js` | Intersection, distance calculations |
| `utils/colorUtils.js` | Color format conversions |
| `core/cropModel.js` | `normalizeCropBox()`, `clampCropValues()`, `resetCrop()` |
| `core/aspectSnap.js` | Parsing ratio strings ("16:9", "2.35", "0.5"), applying constraints |

**Estimated effort:** ~2 hours. **Value:** The crop UI has the most complex client-side logic and is the most likely place for subtle math bugs.

---

## Recommended Test Infrastructure

```
tests/
├── conftest.py              # Shared fixtures (mock tensors, temp directories, GCP mock)
├── test_utils.py            # Priority 1: pure utility functions
├── test_image_saver.py      # Priority 2: image saver logic
├── test_image_loader.py     # Priority 2: image loader logic
├── test_drag_crop.py        # Priority 2: crop math and mask handling
├── test_crop_info.py        # Priority 1: CropInfo parsing
├── test_srt_maker.py        # Priority 2: SRT utilities and fence stripping
├── test_video_saver.py      # Priority 2: video save logic
├── test_gcp_config.py       # Priority 2: GCP config resolution
├── test_retry.py            # Priority 2: retry with backoff
└── test_integration.py      # Priority 3: end-to-end pipelines
```

### `pyproject.toml` additions

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[project.optional-dependencies]
test = ["pytest", "pytest-cov"]
```

### CI addition (`.github/workflows/test.yml`)

A test workflow that runs on every PR and push, gating merges on passing tests. This prevents regressions from reaching the ComfyUI Registry.

---

## Key Risks Without Tests

1. **Silent image corruption** — `sRGBtoLinear`, alpha inversion, and tensor↔PNG conversions have no validation. A numpy dtype change or off-by-one in clipping would ship undetected.
2. **Frame numbering collisions** — `next_frame()` scans the filesystem. If the regex pattern drifts from the save pattern, frames could overwrite each other.
3. **GCP credential fallback** — The metadata-service fallback path and zone→region parsing have multiple string-splitting edge cases that are untested.
4. **Retry logic** — Both `_generate_with_retry()` implementations use string matching (`"429" in error_str`) which is fragile. Tests would lock down the expected behavior.
5. **Mask dimension handling** — `DigitDragCrop.crop()` handles 2D, 3D, and 4D mask tensors with complex broadcasting. This is the highest-risk code path for shape bugs.

---

## Summary

| Priority | Scope | Effort | Risk mitigated |
|---|---|---|---|
| **P1** | Pure utility functions | ~2h | Data corruption, formatting bugs |
| **P2** | Node logic with mocks | ~4h | I/O regressions, crop bugs, retry failures |
| **P3** | Integration pipelines | ~3h | End-to-end VFX pipeline correctness |
| **P4** | JavaScript frontend | ~2h | Crop UI math bugs |
| **Infra** | pytest config + CI | ~1h | Prevents regressions from shipping |

**Recommended starting point:** P1 + test infrastructure (~3 hours total) gives immediate coverage of the most dangerous pure-logic code paths and establishes the test framework for everything else.

# DIGIT Nodes for ComfyUI

AI image generation, video generation, and LLM queries powered by Google Vertex AI — directly from ComfyUI. No proxy, no comfy.org API credits. All usage bills to your own GCP account.

- **Publisher:** DIGIT
- **GitHub:** https://github.com/thedepartmentofexternalservices/comfyui-digit
- **Registry:** https://registry.comfy.org/publishers/digit/nodes/comfyui-digit
- **Version:** 2.1.4
- **License:** MIT

---

## ✨ What Is This?

**DIGIT Nodes** is a collection of ComfyUI custom nodes that connect directly to Google Cloud Vertex AI using the `google-genai` SDK. Every API call goes straight to your GCP project — no middleman, no markup, full control over your billing and quotas.

Use it for:
* AI image generation with Gemini models (Nano Banana 2, Nano Banana Pro)
* AI video generation with Veo models (3.1, 3.0, 2.0) including audio
* LLM text generation with Gemini Pro/Flash models
* VFX-pipeline-style file saving with project/shot/task folder structure

---

## 🎨 Nodes

### DIGIT Gemini Image

Unified image generation node supporting all Gemini image models on Vertex AI. One node handles text-to-image, image-to-image editing, and multi-image composition.

**Models:**
| Model ID | Codename | Notes |
|---|---|---|
| `gemini-3.1-flash-image` | Nano Banana 2 | Newest, fastest. Default. |
| `gemini-3-pro-image-preview` | Nano Banana Pro | Highest fidelity, slower. |
| `gemini-2.5-flash-image` | Nano Banana | Original flash image model. |

**Required inputs:**
* `prompt` — Text prompt describing the image. Multiline.
* `model` — Which Gemini image model to use.
* `aspect_ratio` — Output aspect ratio: `1:1`, `2:3`, `3:2`, `3:4`, `4:1`, `4:3`, `4:5`, `5:4`, `8:1`, `9:16`, `16:9`, `21:9`.
* `resolution` — Output resolution: `1K`, `2K`, or `4K`.
* `seed` — Seed for deterministic output. `0` = random each run. Range: 0–2,147,483,647.
* `temperature` — Controls randomness (0.0–2.0). Default: `1.0`.

**Optional inputs:**
* `image1`, `image2`, `image3` — Input images for editing/reference. Each supports batched tensors.
* `system_instruction` — System prompt. Defaults to an image-generation-optimized instruction.
* `top_p` — Nucleus sampling (0.0–1.0). Default: `1.0`.
* `top_k` — Top-K sampling (1–64). Default: `32`.
* `harassment_threshold`, `hate_speech_threshold`, `sexually_explicit_threshold`, `dangerous_content_threshold` — Per-category safety filters. Options: `BLOCK_NONE`, `BLOCK_ONLY_HIGH`, `BLOCK_MEDIUM_AND_ABOVE`, `BLOCK_LOW_AND_ABOVE`. Default: `BLOCK_NONE`.
* `gcp_project_id` — Your GCP project ID. Auto-detected on GCP instances.
* `gcp_region` — Vertex AI region. Default: `global`.

**Outputs:**
* `image` (IMAGE) — Generated image as RGBA tensor. Multiple images batched on dim 0.
* `text` (STRING) — Any text the model returns alongside the image.

**How it works:**
* **Text-to-image:** Just provide a prompt, no images connected.
* **Image-to-image:** Connect 1–3 images and describe what to change.
* **Seed `0`** means random every run. Set seed > 0 for reproducible results.
* Automatic retry on rate limits (429) and server errors (503) with exponential backoff.

---

### DIGIT Veo Video

Unified video generation node supporting all Veo models on Vertex AI. Auto-detects generation mode based on which inputs are connected: text-to-video, image-to-video, frame interpolation, or reference-guided.

**Models:**
| Model ID | Type | Notes |
|---|---|---|
| `veo-3.1-generate-preview` | Standard | Newest, highest quality. Default. |
| `veo-3.1-fast-generate-preview` | Fast | Lower latency, slightly lower quality. |
| `veo-3.0-generate-001` | Standard | Previous generation. |
| `veo-3.0-fast-generate-001` | Fast | Previous generation, fast. |
| `veo-2.0-generate-001` | Standard | Legacy. No audio, 720p only. |

**Required inputs:**
* `prompt` — Text description of the video. English only. Multiline.
* `model` — Which Veo model to use.
* `aspect_ratio` — `16:9` (landscape) or `9:16` (portrait).
* `resolution` — `720p` or `1080p`.
* `duration_seconds` — Video duration: 4, 6, or 8 seconds.
* `generate_audio` — Generate synchronized audio. Veo 3.0+ only. Default: `True`.
* `seed` — Seed for deterministic output. `0` = random. Range: 0–2,147,483,647.

**Optional inputs:**
* `first_frame` (IMAGE) — Start frame. Triggers image-to-video mode.
* `last_frame` (IMAGE) — End frame. Combined with `first_frame` for interpolation. Veo 3.1 only.
* `reference1`, `reference2`, `reference3` (IMAGE) — Reference images for style/asset guidance. Triggers reference mode.
* `negative_prompt` — Elements to exclude from the video. Multiline.
* `person_generation` — `allow_adult` or `dont_allow`. Default: `allow_adult`.
* `sample_count` — Number of videos per call (1–4). All generate in parallel. Default: `1`.
* `compression_quality` — `optimized` (default) or `lossless` (requires GCS bucket).
* `output_gcs_uri` — GCS bucket URI for lossless output (e.g., `gs://my-bucket/veo-output/`).
* `enhance_prompt` — Let the API expand your prompt for better results. Default: `True`.
* `gcp_project_id` — Your GCP project ID. Auto-detected on GCP instances.
* `gcp_region` — Vertex AI region. Default: `us-central1`.

**Outputs:**
* `video` (VIDEO) — First generated video. Compatible with ComfyUI's built-in SaveVideo node.
* `video_paths` (VEO_PATHS) — List of all video file paths. Connect to DIGIT Video Saver for batch saving.
* `status` (STRING) — Status message with model, mode, resolution, duration, and all file paths.

**Generation modes (auto-detected):**
| Mode | Trigger | Description |
|---|---|---|
| Text-to-video | No images connected | Video from prompt only. |
| Image-to-video | `first_frame` connected | Video starts from that frame. |
| Interpolation | `first_frame` + `last_frame` | Smooth transition between two frames. Veo 3.1 only. |
| Reference | Any `reference` connected | Up to 3 images guide style/content. |

**How it works:**
* Video generation is a long-running operation. The node polls every 20 seconds until complete. Expect 1–3 minutes.
* Set `sample_count` to 2–4 for batch generation. All videos generate in parallel and return together.
* For highest quality: set `compression_quality` to `lossless` with a `output_gcs_uri`.

---

### DIGIT LLM Query

Text generation using Gemini LLM models via Vertex AI. Supports optional image input for vision/multimodal queries.

**Models:**
| Model ID | Notes |
|---|---|
| `gemini-3.1-pro-preview` | Latest, most capable. Default. |
| `gemini-2.5-pro` | Previous generation pro. |
| `gemini-2.5-flash` | Fast and cost-effective. |
| `gemini-2.5-flash-lite` | Fastest, lowest cost. |

**Required inputs:**
* `model` — Which Gemini LLM to use.
* `prompt` — Your question or instruction. Multiline.

**Optional inputs:**
* `gcp_project_id`, `gcp_region` — GCP config. Auto-detected on GCP instances.
* `system_prompt` — System instruction. Multiline.
* `image` (IMAGE) — Image for vision queries (describe, analyze, OCR).
* `max_tokens` — Max response length (1–8,192). Default: `1024`.
* `temperature` — Randomness (0.0–2.0). Default: `0.7`.

**Outputs:**
* `response` (STRING) — The model's text response.

---

### DIGIT Image Saver

Saves images to a VFX-pipeline folder structure with auto-incrementing frame numbers. Supports PNG, JPEG, and EXR.

**Key features:**
* Project/shot/task folder hierarchy with dynamic dropdowns
* Auto-incrementing frame numbers from the highest existing frame
* Batch support — each image in a batch gets its own frame number
* Workflow metadata embedding (PNG chunks, JPEG EXIF, or JSON sidecars for EXR)
* EXR with 32-bit float, RGBA with inverted alpha (VFX convention), tone mapping options

**Inputs:**
* `image` (IMAGE) — Image tensor to save. Supports batch.
* `projekts_root` — Root projects folder (auto-detected).
* `project` — Project folder (5-digit prefix pattern). Dynamic dropdown.
* `shot` — Shot folder. Dynamic dropdown — refreshes when project changes.
* `subfolder` — Subfolder name. Default: `comfy`.
* `task` — Task name. Default: `comp`.
* `format` — `png`, `jpg`, or `exr`.
* `tonemap` — EXR tone mapping: `linear`, `sRGB`, or `Reinhard`.
* `quality` — JPEG quality (1–100). Default: `95`.
* `start_frame` — Starting frame number. Default: `1001`.
* `frame_pad` — Frame padding (1–8). Default: `4`.
* `show_preview` — Show preview in UI. Default: `True`.
* `save_workflow` — Sidecar metadata: `ui`, `api`, `ui + api`, or `none`.

**Output path format:**
```
PROJEKTS_ROOT/PROJECT/shots/SHOT/SUBFOLDER/TASK/PREFIX_SHOT_TASK.FRAME.EXT
```

Example: `25999_comfy_corner/shots/sh010/comfy/comp/25999_sh010_comp.1001.png`

---

### DIGIT Video Saver

Saves videos to the same project/shot folder structure. Supports single VIDEO or batch VEO_PATHS for saving multiple videos from one Veo generation.

**Key features:**
* Same project/shot/task hierarchy as the image saver
* Batch support — connect `video_paths` to save all 4 videos from a `sample_count=4` run
* Auto-incrementing frame numbers
* Workflow metadata sidecars

**Inputs:**
* `video` (VIDEO) — Single video input from DIGIT Veo Video.
* `video_paths` (VEO_PATHS) — Batch video paths from DIGIT Veo Video. Each gets its own frame number.
* Same project/shot/task/frame inputs as the image saver.

Example with `sample_count=4`:
```
25999_sh010_comp.1001.mp4
25999_sh010_comp.1002.mp4
25999_sh010_comp.1003.mp4
25999_sh010_comp.1004.mp4
```

---

### DIGIT Image Loader

Loads the latest rendered frame from a shot/task directory. Pairs with DIGIT Image Saver for round-tripping.

**Key features:**
* Scans the target directory and loads the highest-numbered frame
* Supports PNG, JPEG, and EXR (32-bit float with alpha)
* Optional direct `filepath` input — chain from the image saver output
* Dynamic project/shot dropdowns

**Outputs:**
* `image` (IMAGE) — Loaded image tensor.
* `filepath` (STRING) — Full path to the loaded file.
* `frame` (INT) — Frame number from the filename.

---

## 📦 Installation

### From ComfyUI Manager

Search for **"DIGIT"** in ComfyUI Manager and install.

### Manual

```bash
git clone https://github.com/thedepartmentofexternalservices/comfyui-digit.git
```

Your folder should look like:

```bash
ComfyUI/
└── custom_nodes/
    └── comfyui-digit/
        ├── __init__.py
        ├── gemini_image_node.py
        ├── veo_video_node.py
        ├── llm_node.py
        ├── image_saver_node.py
        ├── image_loader_node.py
        ├── video_saver_node.py
        ├── requirements.txt
        └── web/
            └── digit_image_saver.js
```

Then install dependencies:

```bash
cd ComfyUI/custom_nodes/comfyui-digit
pip install -r requirements.txt
```

Restart ComfyUI to load the new nodes.

---

## 🔧 GCP Setup

All generation nodes require a Google Cloud project with the Vertex AI API enabled.

**1. Install the Google Cloud SDK:**

https://cloud.google.com/sdk/docs/install

**2. Authenticate (local machine):**

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

On GCP instances (Compute Engine, GKE), authentication is automatic.

**3. Enable Vertex AI:**

```bash
gcloud services enable aiplatform.googleapis.com
```

**4. Configure nodes:**

Set `gcp_project_id` on any DIGIT node, or leave blank to auto-detect on GCP instances.

---

## 📁 Project Folder Structure

The saver and loader nodes use a VFX-pipeline folder convention:

```
PROJEKTS_ROOT/
└── NNNNN_project_name/
    └── shots/
        └── shot_name/
            └── subfolder/
                └── task/
                    ├── NNNNN_shot_task.1001.png
                    ├── NNNNN_shot_task.1002.png
                    └── ...
```

Projects must have a 5-digit numeric prefix (e.g., `25999_comfy_corner`). The first 5 characters become the filename prefix.

Supported PROJEKTS_ROOT locations:
* `/Volumes/saint/goose/PROJEKTS` (macOS)
* `/mnt/lucid/PROJEKTS` (Linux)

The project and shot dropdowns are **dynamic** — changing the root refreshes projects, changing the project refreshes shots. No restart needed.

---

## 📦 Dependencies

| Package | Used By |
|---|---|
| `google-genai` | Gemini Image, Veo Video |
| `google-auth` | All GCP nodes |
| `google-cloud-storage` | Veo Video (lossless GCS download) |
| `requests` | LLM Query |
| `piexif` | Image Saver (JPEG EXIF metadata) |
| `opencv-python` | Image Saver/Loader (EXR support) |

---

## 📋 Version History

- **2.1.4** — Published to ComfyUI Registry. Auto-publish GitHub Action.
- **2.1.0** — Added DIGIT Veo Video node, DIGIT Video Saver node. Gemini Image: added Nano Banana 2 (`gemini-3.1-flash-image`), widened aspect ratios and temperature range, added default system prompt.
- **2.0.0** — Added DIGIT Gemini Image node with Vertex AI direct integration.
- **1.0.0** — Initial release with LLM Query, Image Saver, Image Loader.

---

## ⚠️ Known Limitations

- **Veo video generation takes 1–3 minutes.** This is an API limitation — the node polls every 20 seconds until complete.
- **Gemini image models may refuse certain prompts** due to safety filtering, even with thresholds set to `BLOCK_NONE`. This is an API-side restriction.
- **Lossless video output requires a GCS bucket.** The API writes directly to Cloud Storage — there's no way to get lossless output without one.
- **Project/shot dropdowns require at least one run** to populate dynamically when switching roots or projects.
- **The `enhance_prompt` rewrite is invisible.** The Veo API doesn't return the enhanced version of your prompt.

---

## 💬 Notes

This extension is under active development. Functionality may change as Google updates their Vertex AI APIs and model offerings. New models are added as they become available.

Feedback, bug reports, and feature requests welcome via GitHub Issues.

---

## License

MIT License — use it however you want.

---

## Author

Created by [The Department of External Services](https://github.com/thedepartmentofexternalservices)

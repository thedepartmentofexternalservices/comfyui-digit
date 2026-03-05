# ComfyUI DIGIT Nodes

Custom nodes for ComfyUI that connect directly to Google Cloud Vertex AI. No proxy, no comfy.org API credits — all usage bills straight to your own GCP account.

Includes AI image generation (Gemini / Nano Banana), AI video generation (Veo), LLM text generation, and a VFX-pipeline-style file saver with project/shot/task folder structure and auto-incrementing frame numbers.

---

## Nodes

- [DIGIT Gemini Image](#digit-gemini-image) — AI image generation
- [DIGIT Veo Video](#digit-veo-video) — AI video generation
- [DIGIT LLM Query](#digit-llm-query) — LLM text generation
- [DIGIT Image Saver](#digit-image-saver) — Project/shot image saving
- [DIGIT Video Saver](#digit-video-saver) — Project/shot video saving
- [DIGIT Image Loader](#digit-image-loader) — Project/shot image loading

---

## DIGIT Gemini Image

Unified image generation node supporting all Gemini image models on Vertex AI. Handles text-to-image, image-to-image editing, and multi-image inputs in a single node.

### Models

| Model ID | Codename | Notes |
|---|---|---|
| `gemini-3.1-flash-image` | Nano Banana 2 | Newest, fastest. Default. |
| `gemini-3-pro-image-preview` | Nano Banana Pro | Highest fidelity, slower. |
| `gemini-2.5-flash-image` | Nano Banana | Original flash image model. |

### Required Inputs

| Input | Type | Default | Description |
|---|---|---|---|
| `prompt` | STRING | — | Text prompt describing the image to generate. Multiline. |
| `model` | Combo | `gemini-3.1-flash-image` | Which Gemini image model to use. |
| `aspect_ratio` | Combo | `16:9` | Output aspect ratio. Options: `1:1`, `2:3`, `3:2`, `3:4`, `4:1`, `4:3`, `4:5`, `5:4`, `8:1`, `9:16`, `16:9`, `21:9`. |
| `resolution` | Combo | `1K` | Output resolution tier: `1K`, `2K`, or `4K`. |
| `seed` | INT | `0` | Seed for deterministic output. `0` = random each run. Range: 0–2,147,483,647. |
| `temperature` | FLOAT | `1.0` | Controls randomness. Lower = more deterministic, higher = more creative. Range: 0.0–2.0. |

### Optional Inputs

| Input | Type | Default | Description |
|---|---|---|---|
| `image1` | IMAGE | — | Input image for editing/reference. Supports batched tensors (multiple images along dim 0). |
| `image2` | IMAGE | — | Second input image. |
| `image3` | IMAGE | — | Third input image. |
| `system_instruction` | STRING | *(see below)* | System prompt that guides the model's behavior. Defaults to an image-generation-focused instruction. |
| `top_p` | FLOAT | `1.0` | Nucleus sampling threshold. Range: 0.0–1.0. |
| `top_k` | INT | `32` | Top-K sampling. Range: 1–64. |
| `harassment_threshold` | Combo | `BLOCK_NONE` | Safety filter level for harassment content. |
| `hate_speech_threshold` | Combo | `BLOCK_NONE` | Safety filter level for hate speech. |
| `sexually_explicit_threshold` | Combo | `BLOCK_NONE` | Safety filter level for sexually explicit content. |
| `dangerous_content_threshold` | Combo | `BLOCK_NONE` | Safety filter level for dangerous content. |
| `gcp_project_id` | STRING | — | Your GCP project ID. Auto-detected on GCP instances if left blank. |
| `gcp_region` | STRING | `global` | Vertex AI region. Use `global` for default routing. |

Safety threshold options: `BLOCK_NONE`, `BLOCK_ONLY_HIGH`, `BLOCK_MEDIUM_AND_ABOVE`, `BLOCK_LOW_AND_ABOVE`.

### Default System Prompt

The node ships with a default system instruction optimized for image generation:

> You are an expert image-generation engine. You must ALWAYS produce an image. Interpret all user input — regardless of format, intent, or abstraction — as literal visual directives for image composition. If a prompt is conversational or lacks specific visual details, you must creatively invent a concrete visual scenario that depicts the concept. Prioritize generating the visual representation above any text, formatting, or conversational requests.

You can override or clear this in the `system_instruction` field.

### Outputs

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | Generated image as an RGBA tensor. If the model returns multiple images, they are batched along dim 0. |
| `text` | STRING | Any text the model returns alongside the image (commentary, reasoning, etc.). |

### How It Works

- **Text-to-image**: Just provide a prompt, no images connected.
- **Image-to-image**: Connect 1–3 images and describe what to change. Each image slot supports batched tensors, so you can send many images through a single slot.
- **Seed behavior**: Seed `0` means random every run (the node always re-executes). Set a seed > 0 for reproducible results — running twice with the same seed and prompt produces the same image.
- **Retry logic**: Automatically retries on rate limits (429) and server errors (503) with exponential backoff — 3 attempts with 5/10/20 second delays.

---

## DIGIT Veo Video

Unified video generation node supporting all Veo models on Vertex AI. Handles text-to-video, image-to-video, first/last frame interpolation, and reference-guided generation in a single node. Automatically detects which mode to use based on connected inputs.

### Models

| Model ID | Type | Notes |
|---|---|---|
| `veo-3.1-generate-preview` | Standard | Newest, highest quality. Default. |
| `veo-3.1-fast-generate-preview` | Fast | Lower latency, slightly lower quality. |
| `veo-3.0-generate-001` | Standard | Previous generation. |
| `veo-3.0-fast-generate-001` | Fast | Previous generation, fast variant. |
| `veo-2.0-generate-001` | Standard | Legacy model. No audio, 720p only. |

### Required Inputs

| Input | Type | Default | Description |
|---|---|---|---|
| `prompt` | STRING | — | Text description of the video to generate. English only. Multiline. |
| `model` | Combo | `veo-3.1-generate-preview` | Which Veo model to use. |
| `aspect_ratio` | Combo | `16:9` | Output aspect ratio: `16:9` (landscape) or `9:16` (portrait). |
| `resolution` | Combo | `720p` | Output resolution: `720p` or `1080p`. |
| `duration_seconds` | INT | `8` | Video duration in seconds: 4, 6, or 8. Step: 2. |
| `generate_audio` | BOOLEAN | `True` | Generate synchronized audio track. Veo 3.0+ only. |
| `seed` | INT | `0` | Seed for deterministic output. `0` = random. Range: 0–2,147,483,647. |

### Optional Inputs

| Input | Type | Default | Description |
|---|---|---|---|
| `first_frame` | IMAGE | — | Start frame for image-to-video mode. Connect an image and the video will begin from that frame. |
| `last_frame` | IMAGE | — | End frame for interpolation. Used with `first_frame` to create a video transitioning between two frames. Veo 3.1 only. |
| `reference1` | IMAGE | — | Reference image 1 for style/asset guidance. Triggers reference mode. |
| `reference2` | IMAGE | — | Reference image 2. |
| `reference3` | IMAGE | — | Reference image 3. |
| `negative_prompt` | STRING | — | Elements to exclude from the video. Multiline. |
| `person_generation` | Combo | `allow_adult` | Person generation control: `allow_adult` or `dont_allow`. |
| `sample_count` | INT | `1` | Number of videos to generate in one call. Range: 1–4. All generate in parallel. |
| `compression_quality` | Combo | `optimized` | `optimized` returns compressed video bytes directly. `lossless` writes higher-bitrate output to a GCS bucket (requires `output_gcs_uri`). |
| `output_gcs_uri` | STRING | — | GCS bucket URI for lossless output. Example: `gs://my-bucket/veo-output/`. Required when `compression_quality` is `lossless`. |
| `enhance_prompt` | BOOLEAN | `True` | Let the API rewrite your prompt with more detail for better results. Turn off for precise prompt control. |
| `gcp_project_id` | STRING | — | Your GCP project ID. Auto-detected on GCP instances. |
| `gcp_region` | STRING | `us-central1` | Vertex AI region. |

### Outputs

| Output | Type | Description |
|---|---|---|
| `video` | VIDEO | First generated video. Compatible with ComfyUI's built-in `SaveVideo` and `GetVideoComponents` nodes. |
| `video_paths` | VEO_PATHS | List of all generated video file paths (for batch saving with DIGIT Video Saver). |
| `status` | STRING | Status message with model, mode, resolution, duration, and file paths. |

### Generation Modes

The node auto-detects the mode based on which inputs are connected:

| Mode | Trigger | Description |
|---|---|---|
| **Text-to-video** | No images connected | Generates video from prompt only. |
| **Image-to-video** | `first_frame` connected | Video starts from the provided frame. |
| **Interpolation** | `first_frame` + `last_frame` connected | Video transitions from first frame to last frame. Veo 3.1 only. |
| **Reference** | Any `reference1`/`2`/`3` connected | Up to 3 reference images guide the style and content. |

You cannot use reference mode and image-to-video mode simultaneously.

### How It Works

- Video generation is a **long-running operation**. The node submits the request, then polls every 20 seconds until the video is ready. Expect 1–3 minutes depending on duration and model.
- **Batch generation**: Set `sample_count` to 2–4 to generate multiple videos in a single API call. All videos generate in parallel on Google's side and return together.
- **Lossless output**: Set `compression_quality` to `lossless` and provide a `output_gcs_uri`. The API writes higher-bitrate video directly to your GCS bucket. The node downloads it locally so it still flows through ComfyUI.
- **Retry logic**: Retries on 429/503 with exponential backoff for the initial API call. The polling loop handles its own lifecycle.

---

## DIGIT LLM Query

Text generation node using Gemini LLM models via Vertex AI. Supports optional image input for vision/multimodal queries.

### Models

| Model ID | Notes |
|---|---|
| `gemini-3.1-pro-preview` | Latest, most capable. Default. |
| `gemini-2.5-pro` | Previous generation pro model. |
| `gemini-2.5-flash` | Fast and cost-effective. |
| `gemini-2.5-flash-lite` | Fastest, lowest cost. |

### Required Inputs

| Input | Type | Default | Description |
|---|---|---|---|
| `model` | Combo | `gemini-3.1-pro-preview` | Which Gemini LLM model to use. |
| `prompt` | STRING | — | Your question or instruction. Multiline. |

### Optional Inputs

| Input | Type | Default | Description |
|---|---|---|---|
| `gcp_project_id` | STRING | — | GCP project ID. Auto-detected on GCP instances. |
| `gcp_region` | STRING | — | GCP region. Auto-detected on GCP instances. |
| `system_prompt` | STRING | — | System instruction to guide model behavior. Multiline. |
| `image` | IMAGE | — | Image input for vision queries (describe, analyze, OCR, etc.). |
| `max_tokens` | INT | `1024` | Maximum response length. Range: 1–8,192. |
| `temperature` | FLOAT | `0.7` | Controls randomness. Range: 0.0–2.0. |

### Outputs

| Output | Type | Description |
|---|---|---|
| `response` | STRING | The model's text response. |

---

## DIGIT Image Saver

Saves images to a VFX-pipeline-style folder structure with auto-incrementing frame numbers. Supports PNG, JPEG, and EXR formats with workflow metadata embedding.

### Required Inputs

| Input | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | Image tensor to save. Supports batch — each image in the batch gets its own frame number. |
| `projekts_root` | Combo | *(auto-detected)* | Root folder for projects. Scans `/Volumes/saint/goose/PROJEKTS` and `/mnt/lucid/PROJEKTS`. |
| `project` | Combo | *(first found)* | Project folder (must match `NNNNN_` prefix pattern). Dynamic dropdown — updates when you change root. |
| `shot` | Combo | *(first found)* | Shot folder inside `<project>/shots/`. Dynamic dropdown — updates when you change project. |
| `subfolder` | STRING | `comfy` | Subfolder inside the shot directory. |
| `task` | STRING | `comp` | Task name (e.g., `comp`, `beauty`, `diffuse`). |
| `format` | Combo | `png` | Output format: `png`, `jpg`, or `exr`. |
| `tonemap` | Combo | `linear` | Tone mapping for EXR output: `linear` (no transform), `sRGB` (gamma encode to linear), `Reinhard` (HDR compression). |
| `quality` | INT | `95` | JPEG quality. Range: 1–100. Only used for `jpg` format. |
| `start_frame` | INT | `1001` | Starting frame number. The node auto-increments from the highest existing frame in the directory. |
| `frame_pad` | INT | `4` | Frame number padding (e.g., 4 = `1001`, 8 = `00001001`). |
| `show_preview` | BOOLEAN | `True` | Show image preview in the ComfyUI UI. |
| `save_workflow` | Combo | `ui` | Save workflow metadata as sidecar JSON files: `ui` (visual workflow), `api` (API prompt), `ui + api` (both), `none`. |

### Output Path Format

```
<projekts_root>/<project>/shots/<shot>/<subfolder>/<task>/<prefix>_<shot>_<task>.<frame>.<ext>
```

Example:
```
/Volumes/saint/goose/PROJEKTS/25999_comfy_corner/shots/sh010/comfy/comp/25999_sh010_comp.1001.png
```

### Outputs

| Output | Type | Description |
|---|---|---|
| `filepath` | STRING | Full path to the last saved file. Chain into DIGIT Image Loader for round-tripping. |

### Format Details

- **PNG**: 8-bit RGBA/RGB. Workflow metadata embedded in PNG text chunks.
- **JPEG**: 8-bit RGB (alpha stripped). Workflow metadata embedded in EXIF UserComment via piexif.
- **EXR**: 32-bit float. Supports RGBA with inverted alpha (VFX convention). Workflow metadata saved as JSON sidecar files (`_api.json`, `_ui.json`).

---

## DIGIT Video Saver

Saves videos to the same project/shot folder structure as the image saver. Supports single VIDEO input or batch VEO_PATHS for saving multiple videos from a single Veo generation.

### Required Inputs

| Input | Type | Default | Description |
|---|---|---|---|
| `projekts_root` | Combo | *(auto-detected)* | Root folder for projects. |
| `project` | Combo | *(first found)* | Project folder. Dynamic dropdown. |
| `shot` | Combo | *(first found)* | Shot folder. Dynamic dropdown. |
| `subfolder` | STRING | `comfy` | Subfolder inside the shot directory. |
| `task` | STRING | `comp` | Task name. |
| `start_frame` | INT | `1001` | Starting frame number. Auto-increments from highest existing. |
| `frame_pad` | INT | `4` | Frame number padding. |
| `save_workflow` | Combo | `api` | Workflow metadata sidecar: `api`, `ui`, `ui + api`, `none`. |

### Optional Inputs

| Input | Type | Description |
|---|---|---|
| `video` | VIDEO | Single video input. Connect from the `video` output of DIGIT Veo Video. |
| `video_paths` | VEO_PATHS | Batch video paths. Connect from the `video_paths` output of DIGIT Veo Video. Each video gets its own frame number. |

If `video_paths` is connected, all videos in the batch are saved with incrementing frame numbers. If only `video` is connected, a single video is saved.

### Output Path Format

```
<projekts_root>/<project>/shots/<shot>/<subfolder>/<task>/<prefix>_<shot>_<task>.<frame>.mp4
```

Example with `sample_count=4`:
```
25999_sh010_comp.1001.mp4
25999_sh010_comp.1002.mp4
25999_sh010_comp.1003.mp4
25999_sh010_comp.1004.mp4
```

### Outputs

| Output | Type | Description |
|---|---|---|
| `filepaths` | STRING | Newline-separated list of all saved file paths. |

---

## DIGIT Image Loader

Loads the latest rendered frame from a shot/task directory. Pairs with DIGIT Image Saver — point both at the same shot and task to always have the most recent output available.

### Required Inputs

| Input | Type | Default | Description |
|---|---|---|---|
| `projekts_root` | Combo | *(auto-detected)* | Root folder for projects. |
| `project` | Combo | *(first found)* | Project folder. Dynamic dropdown. |
| `shot` | Combo | *(first found)* | Shot folder. Dynamic dropdown. |
| `subfolder` | STRING | `comfy` | Subfolder inside the shot directory. |
| `task` | STRING | `comp` | Task name. |
| `format` | Combo | `png` | File format to look for: `png`, `jpg`, or `exr`. |

### Optional Inputs

| Input | Type | Description |
|---|---|---|
| `filepath` | STRING | Direct file path input. If connected (e.g., from DIGIT Image Saver output), loads that file instead of scanning the directory. |

### Outputs

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | Loaded image as a float32 tensor. |
| `filepath` | STRING | Full path to the loaded file. |
| `frame` | INT | Frame number extracted from the filename. |

---

## Installation

### From ComfyUI Registry / Manager

Search for **"DIGIT"** in ComfyUI Manager.

### Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/thedepartmentofexternalservices/comfyui-digit.git
cd comfyui-digit
pip install -r requirements.txt
```

### Dependencies

| Package | Used By |
|---|---|
| `google-genai` | Gemini Image, Veo Video |
| `google-auth` | All GCP nodes |
| `google-cloud-storage` | Veo Video (lossless GCS download) |
| `requests` | LLM Query |
| `piexif` | Image Saver (JPEG EXIF metadata) |
| `opencv-python` | Image Saver/Loader (EXR support) |

---

## GCP Setup

All generation nodes require a Google Cloud project with the Vertex AI API enabled.

### 1. Install the Google Cloud SDK

https://cloud.google.com/sdk/docs/install

### 2. Authenticate

**Local machine** (your laptop/desktop):
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

**GCP instance** (Compute Engine, GKE): Authentication is automatic via the metadata service. The nodes auto-detect your project ID and region.

### 3. Enable Vertex AI API

```bash
gcloud services enable aiplatform.googleapis.com
```

### 4. Configure the Nodes

Set `gcp_project_id` on any DIGIT node, or leave blank to auto-detect on GCP instances. The `gcp_region` defaults to `global` (Gemini Image) or `us-central1` (Veo Video).

---

## Project Folder Structure

The image saver, video saver, and image loader nodes use a VFX-pipeline folder convention:

```
PROJEKTS_ROOT/
  NNNNN_project_name/
    shots/
      shot_name/
        subfolder/
          task/
            NNNNN_shot_task.FRAME.EXT
```

The `PROJEKTS_ROOT` is auto-detected from:
- `/Volumes/saint/goose/PROJEKTS` (macOS)
- `/mnt/lucid/PROJEKTS` (Linux)

Projects must have a 5-digit numeric prefix (e.g., `25999_comfy_corner`). The first 5 characters become the filename prefix.

The project and shot dropdowns are **dynamic** — changing the root refreshes the project list, changing the project refreshes the shot list. No restart required.

---

## License

MIT

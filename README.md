# DIGIT Nodes for ComfyUI

**Production-grade AI nodes that connect directly to Google Cloud Vertex AI.** No middleman, no proxy, no rate limits beyond your own GCP quota. Your usage bills directly to your GCP account at Google's API pricing — no markup.

DIGIT Nodes give you raw, unfiltered access to Google's most powerful generative AI models from inside ComfyUI. Every node auto-detects your GCP credentials, so once you're authenticated, everything just works.

---

## Why DIGIT Nodes?

Most ComfyUI nodes that talk to Google's models go through third-party proxies or require API keys from wrapper services. DIGIT Nodes skip all of that. You authenticate once with `gcloud`, and every node talks directly to Vertex AI using the official `google-genai` SDK.

This means:

- **No API key management** — uses your existing GCP credentials
- **No rate limit surprises** — you control your own quotas
- **No data routing through third parties** — your prompts and outputs stay between you and Google
- **Lossless video output** — the only way to get uncompressed Veo output is through the API with a GCS bucket, and DIGIT Nodes support this natively
- **Auto-detection** — on GCP instances (Compute Engine, GKE), project ID and region are detected automatically from the metadata service. On local machines, it uses your `gcloud` login.

---

## The Nodes

### DIGIT Gemini Image

Generate and edit images using Google's Gemini image models directly through Vertex AI.

This is a unified node — it handles text-to-image, image editing, and multi-image composition all in one place. Feed it a prompt and it generates an image. Feed it a prompt plus up to 3 input images and it edits or combines them.

**Supported Models:**

| Model | Internal Name | What It Is |
|-------|--------------|------------|
| Gemini 3.1 Flash Image | `gemini-3.1-flash-image` | Nano Banana 2 — Google's newest and fastest image model. Default choice. |
| Gemini 3 Pro Image | `gemini-3-pro-image-preview` | Nano Banana Pro — higher quality, slower. |
| Gemini 2.5 Flash Image | `gemini-2.5-flash-image` | Previous generation. Still solid. |

**Inputs:**

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| prompt | STRING | — | Your image generation prompt. Required. |
| model | COMBO | gemini-3.1-flash-image | Which Gemini image model to use. |
| aspect_ratio | COMBO | 16:9 | Output aspect ratio. 12 options: 1:1, 2:3, 3:2, 3:4, 4:1, 4:3, 4:5, 5:4, 8:1, 9:16, 16:9, 21:9. |
| resolution | COMBO | 1K | Output resolution: 1K, 2K, or 4K. |
| seed | INT | 0 | Reproducibility seed. 0 = random each run. Max 2,147,483,647. |
| temperature | FLOAT | 1.0 | Creativity control. Range 0.0–2.0. Higher = more creative/varied. |
| image1, image2, image3 | IMAGE | — | Optional input images for editing, style transfer, or composition. Batched images are iterated automatically. |
| system_instruction | STRING | (built-in) | System prompt that tells the model to always produce images. Customizable. |
| top_p, top_k | FLOAT/INT | 1.0 / 32 | Nucleus and top-k sampling parameters for fine-tuning output diversity. |
| harassment_threshold | COMBO | BLOCK_NONE | Safety filter for harassment content. Options: BLOCK_NONE, BLOCK_ONLY_HIGH, BLOCK_MEDIUM_AND_ABOVE, BLOCK_LOW_AND_ABOVE. |
| hate_speech_threshold | COMBO | BLOCK_NONE | Safety filter for hate speech. |
| sexually_explicit_threshold | COMBO | BLOCK_NONE | Safety filter for sexually explicit content. |
| dangerous_content_threshold | COMBO | BLOCK_NONE | Safety filter for dangerous content. |
| gcp_project_id | STRING | (auto) | Your GCP project ID. Leave blank to auto-detect. |
| gcp_region | STRING | global | Vertex AI region. "global" uses Google's default routing. |

**Outputs:**

| Output | Type | Description |
|--------|------|-------------|
| image | IMAGE | Generated RGBA image tensor. |
| text | STRING | Any text the model returned alongside the image. |

**Built-in resilience:** Automatic retry with exponential backoff on 429 (rate limit) and 503 (service unavailable) errors. Up to 3 retries.

---

### DIGIT Veo Video

Generate videos using Google's Veo models directly through Vertex AI. Supports text-to-video, image-to-video, frame interpolation, and reference-based generation — all in one unified node.

The node auto-detects which mode to use based on what you connect:
- **Nothing connected** → text-to-video
- **first_frame connected** → image-to-video (animates from your image)
- **first_frame + last_frame** → interpolation (generates video between two frames)
- **reference images connected** → reference-based (generates video maintaining visual consistency with reference images)

**Supported Models:**

| Model | Internal Name | Description |
|-------|--------------|-------------|
| Veo 3.1 | `veo-3.1-generate-preview` | Latest and most capable. Default choice. |
| Veo 3.1 Fast | `veo-3.1-fast-generate-preview` | Faster generation, slightly lower quality. |
| Veo 3.0 | `veo-3.0-generate-001` | Previous generation, very capable. |
| Veo 3.0 Fast | `veo-3.0-fast-generate-001` | Fast version of Veo 3.0. |
| Veo 2.0 | `veo-2.0-generate-001` | Older model, still available. |

**Inputs:**

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| prompt | STRING | — | Video generation prompt. Required. |
| model | COMBO | veo-3.1-generate-preview | Which Veo model to use. |
| aspect_ratio | COMBO | 16:9 | 16:9 (landscape) or 9:16 (portrait). |
| resolution | COMBO | 720p | 720p or 1080p. |
| duration_seconds | INT | 8 | Video length: 4, 6, or 8 seconds. |
| generate_audio | BOOLEAN | true | Whether Veo generates synchronized audio. |
| seed | INT | 0 | Reproducibility seed. 0 = random. |
| first_frame | IMAGE | — | Starting frame for image-to-video mode. |
| last_frame | IMAGE | — | Ending frame for interpolation mode (requires first_frame). |
| reference1, reference2, reference3 | IMAGE | — | Reference images for style/asset consistency. Cannot be used with first_frame. |
| negative_prompt | STRING | — | What you don't want in the video. |
| person_generation | COMBO | allow_adult | "allow_adult" or "dont_allow". |
| sample_count | INT | 1 | Generate 1–4 videos per run. |
| compression_quality | COMBO | optimized | "optimized" = compressed MP4 in API response. "lossless" = full-quality MP4 written to your GCS bucket. |
| output_gcs_uri | STRING | — | GCS bucket path for lossless output, e.g. `gs://my-bucket/output/`. Required when using lossless compression. |
| enhance_prompt | BOOLEAN | true | Let Veo enhance your prompt for better results. |
| gcp_project_id | STRING | (auto) | Your GCP project ID. |
| gcp_region | STRING | us-central1 | Vertex AI region. |

**Outputs:**

| Output | Type | Description |
|--------|------|-------------|
| video | VIDEO | First generated video as a ComfyUI VIDEO type. |
| video_paths | VEO_PATHS | List of all generated video file paths (for batch saving). |
| status | STRING | Generation details: model, mode, duration, resolution, file paths. |

**About lossless vs. optimized:**

Every Veo video generation — whether through Google's AI Studio, the web console, Freepik, Weavy, or any other third-party tool — returns the **optimized** (compressed) version. The **only** way to get lossless output is through the API with a GCS bucket URI. This node is one of the only tools that gives you that option. You can be on your local Mac, a Linux workstation, or a cloud VM — it doesn't matter where you run ComfyUI. As long as the API call includes `output_gcs_uri`, the lossless file goes to your bucket.

**Built-in resilience:** Automatic retry with exponential backoff. 20-second polling interval for long-running operations. Multiple response parsing fallback paths for SDK version compatibility.

---

### DIGIT LLM Query

Send text (and optionally images) to Gemini LLM models and get text responses. Useful for prompt engineering, image analysis, script writing, or any text generation task within a ComfyUI workflow.

**Supported Models:**

| Model | Internal Name | Description |
|-------|--------------|-------------|
| Gemini 3.1 Pro | `gemini-3.1-pro-preview` | Latest and most capable text model. Default. |
| Gemini 2.5 Pro | `gemini-2.5-pro` | Very strong, slightly older. |
| Gemini 2.5 Flash | `gemini-2.5-flash` | Fast and cost-effective. |
| Gemini 2.5 Flash Lite | `gemini-2.5-flash-lite` | Fastest and cheapest. |

**Inputs:**

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| model | COMBO | gemini-3.1-pro-preview | Which Gemini text model to use. |
| prompt | STRING | — | Your text prompt. Required. |
| system_prompt | STRING | — | Optional system instructions to guide the model's behavior. |
| image | IMAGE | — | Optional image input for vision/multimodal queries. |
| max_tokens | INT | 1024 | Maximum response length. Range 1–8192. |
| temperature | FLOAT | 0.7 | Creativity control. Range 0.0–2.0. |
| gcp_project_id | STRING | (auto) | Your GCP project ID. |
| gcp_region | STRING | (auto) | GCP region. |

**Outputs:**

| Output | Type | Description |
|--------|------|-------------|
| response | STRING | The model's text response. |

---

### DIGIT SRT Maker

Automatically generate SRT subtitle files from scripts. Paste a Google Doc URL (private or public) or raw script text, and Gemini 3.1 Pro analyzes it to extract only the spoken dialogue — stripping out stage directions, scene headings, camera instructions, and action lines — then generates a properly timed SRT file.

**How it works:**

1. Fetches the script from a Google Doc URL (using your GCP credentials for private docs) or accepts pasted text
2. Sends the full script to Gemini 3.1 Pro with instructions to identify only spoken dialogue
3. Gemini generates timed SRT subtitles based on natural speaking pace
4. Saves the `.srt` file to your project's `assets/auto_srt/` folder

**Inputs:**

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| script_url | STRING | — | Google Doc URL or any web URL containing the script. Private Google Docs are supported via your `gcloud auth login` credentials. |
| extra_instructions | STRING | (built-in) | Instructions for Gemini about what to include/exclude. Default tells it to only extract spoken dialogue. Customize to filter by character, scene, etc. |
| words_per_second | FLOAT | 2.5 | Speaking rate for subtitle timing. 2.5 wps is natural conversational pace. Lower = slower reading, longer subtitles. |
| script_text | STRING | — | Paste script text directly instead of using a URL. Overrides the URL if both are provided. |
| projekts_root | COMBO | (auto) | PROJEKTS volume root. |
| project | COMBO | (auto) | Project folder (dynamic dropdown). |
| filename | STRING | dialogue | Output filename (without .srt extension). |
| gcp_project_id | STRING | (auto) | Your GCP project ID. |
| gcp_region | STRING | global | Vertex AI region. |

**Outputs:**

| Output | Type | Description |
|--------|------|-------------|
| srt_filepath | STRING | Full path to the saved .srt file. |
| srt_text | STRING | Raw SRT content as text. |

**Output path:** `PROJEKTS/project/assets/auto_srt/filename.srt`

**Google Docs authentication:** For private docs, the node uses `gcloud auth print-access-token` from your `gcloud auth login --enable-gdrive-access` session to authenticate with the Google Drive API. If authenticated access fails, it falls back to public export.

---

### DIGIT Image Saver

Save images to a VFX-pipeline folder structure with auto-incrementing frame numbers. Designed for production workflows where files need to follow a strict naming and directory convention.

**Inputs:**

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| image | IMAGE | — | Image to save. Batch images save as sequential frames. |
| projekts_root | COMBO | (auto) | PROJEKTS volume root. Auto-detects available mount points. |
| project | COMBO | (auto) | Project folder (dynamic dropdown, scans for `#####_` prefix pattern). |
| shot | COMBO | (auto) | Shot folder (dynamic dropdown, scans `project/shots/`). |
| subfolder | STRING | comfy | Subfolder within the shot (e.g. "comfy", "renders", "plates"). |
| task | STRING | comp | Task name (e.g. "comp", "paint", "roto"). |
| format | COMBO | png | Output format: PNG, JPEG, or EXR. |
| tonemap | COMBO | linear | EXR tone mapping: linear, sRGB, or Reinhard. Only applies to EXR format. |
| quality | INT | 95 | JPEG quality (1–100). Only applies to JPEG format. |
| start_frame | INT | 1001 | Starting frame number if no existing frames are found. |
| frame_pad | INT | 4 | Frame number padding (e.g. 4 = `0001`, 8 = `00000001`). |
| show_preview | BOOLEAN | true | Show saved image in ComfyUI's preview panel. |
| save_workflow | COMBO | ui | Save workflow metadata as JSON sidecar: "ui", "api", "ui + api", or "none". |

**Output path:** `PROJEKTS/project/shots/shot/subfolder/task/PREFIX_SHOT_TASK.FRAME.EXT`

**Example:** `/mnt/lucid/PROJEKTS/25999_comfy_corner/shots/sh010/comfy/comp/25999_sh010_comp.1001.png`

**EXR support:** Full 32-bit float EXR with OpenCV. Supports RGBA with inverted alpha (VFX convention). Tone mapping options let you convert from sRGB gamma space to linear on save.

**Batch support:** If a batched IMAGE tensor is connected (e.g. from a batch generation), each image in the batch is saved as a sequential frame.

---

### DIGIT Video Saver

Save videos to the same VFX-pipeline folder structure as the Image Saver. Accepts either a single VIDEO or a batch of video file paths from the Veo node.

**Inputs:**

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| projekts_root | COMBO | (auto) | PROJEKTS volume root. |
| project | COMBO | (auto) | Project folder (dynamic dropdown). |
| shot | COMBO | (auto) | Shot folder (dynamic dropdown). |
| subfolder | STRING | comfy | Subfolder within the shot. |
| task | STRING | comp | Task name. |
| start_frame | INT | 1001 | Starting frame number. |
| frame_pad | INT | 4 | Frame number padding. |
| save_workflow | COMBO | api | Save workflow metadata as JSON sidecar. |
| video | VIDEO | — | Single video input (from Veo node's VIDEO output). |
| video_paths | VEO_PATHS | — | Batch video paths (from Veo node's VEO_PATHS output). Saves all videos with incrementing frame numbers. |

**Output path:** `PROJEKTS/project/shots/shot/subfolder/task/PREFIX_SHOT_TASK.FRAME.mp4`

**Batch support:** Connect the `video_paths` output from the Veo node and all generated videos (up to 4) are saved with sequential frame numbers.

---

### DIGIT Image Loader

Load the latest rendered frame from a shot/task directory. Pairs with the Image Saver — point both at the same shot and task to always have the most recent output available as an IMAGE tensor.

**Inputs:**

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| projekts_root | COMBO | (auto) | PROJEKTS volume root. |
| project | COMBO | (auto) | Project folder (dynamic dropdown). |
| shot | COMBO | (auto) | Shot folder (dynamic dropdown). |
| subfolder | STRING | comfy | Subfolder within the shot. |
| task | STRING | comp | Task name. |
| format | COMBO | png | File format to scan for: PNG, JPEG, or EXR. |
| filepath | STRING | — | Optional direct filepath input. If connected from a Saver node, loads that specific file instead of scanning. |

**Outputs:**

| Output | Type | Description |
|--------|------|-------------|
| image | IMAGE | Loaded image as a ComfyUI tensor. |
| filepath | STRING | Full path to the loaded file. |
| frame | INT | Frame number of the loaded file. |

**Smart loading:** Automatically finds the highest-numbered frame in the target directory. If a `filepath` is connected (e.g. from the Image Saver's output), it loads that exact file instead.

**EXR support:** Full 32-bit float EXR loading with OpenCV. BGRA to RGBA conversion and alpha un-inversion handled automatically.

---

### DIGIT Drag Crop

Interactive image cropping with a drag-and-drop crop box directly on the node's image preview. No more guessing pixel coordinates — drag to select, resize with handles, and the cropped region updates in real time.

**Features:**

- **Drag to crop** — Click and drag anywhere on the preview to create a new crop region
- **Resize handles** — Corner and edge handles for precise resizing
- **Move** — Drag inside the crop box to reposition it
- **Aspect ratio locking** — Enter values like `16:9`, `2.35`, or `0.5` and toggle the lock to constrain the crop box
- **Pixel snapping** — Snap crop dimensions to grids of 2, 4, 8, 16, 32, or 64 pixels
- **Box color** — 11 preset colors (Lime, Grey, White, Black, Red, Green, Blue, Yellow, Magenta, Cyan, Hot Pink)
- **Info overlay** — Shows crop dimensions in pixels and percentage on the crop box. Toggle on/off.
- **Resolution tracking** — Automatically resets crop when the input image resolution changes
- **Mask pass-through** — Optional mask input is cropped to match the image crop region

**Inputs:**

| Input | Type | Description |
|-------|------|-------------|
| image | IMAGE | Image to crop. Run the graph once to load the preview. |
| crop_left/right/top/bottom | INT | Numeric crop offsets (also adjustable via the interactive UI). |
| mask | MASK | Optional mask that gets cropped to match. |

**Outputs:**

| Output | Type | Description |
|--------|------|-------------|
| IMAGE | IMAGE | Cropped image. |
| MASK | MASK | Cropped mask (or zero mask if none connected). |
| CROP_JSON | STRING | JSON with all crop coordinates and dimensions. |

**Note:** You must run the graph once before the interactive preview appears. This is a technical limitation of ComfyUI's widget system — the node needs to receive image data from upstream before it can display anything.

---

### DIGIT Crop Info

Companion node for DIGIT Drag Crop. Takes the CROP_JSON string output and breaks it into individual integer values for use in other nodes.

**Inputs:**

| Input | Type | Description |
|-------|------|-------------|
| crop_json | STRING | CROP_JSON output from the Drag Crop node. |

**Outputs:**

| Output | Type | Description |
|--------|------|-------------|
| left | INT | Left crop offset in pixels. |
| top | INT | Top crop offset in pixels. |
| right | INT | Right crop offset in pixels. |
| bottom | INT | Bottom crop offset in pixels. |
| width | INT | Cropped region width in pixels. |
| height | INT | Cropped region height in pixels. |
| csv | STRING | All values as comma-separated string. |
| pretty | STRING | Human-readable formatted string. |

---

## Installation

### From ComfyUI Registry / Manager
Search for **"DIGIT"** in the ComfyUI Manager node list.

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/thedepartmentofexternalservices/comfyui-digit.git
cd comfyui-digit
pip install -r requirements.txt
```

---

## GCP Setup

DIGIT Nodes require a Google Cloud project with Vertex AI enabled. Setup takes about 2 minutes.

### 1. Install the Google Cloud SDK
https://cloud.google.com/sdk/docs/install

### 2. Authenticate
```bash
# Log in to your Google account
gcloud auth login --enable-gdrive-access

# Set up application default credentials (for Vertex AI)
gcloud auth application-default login

# Set your default project
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

### 3. Enable APIs
```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable drive.googleapis.com
```

### 4. (Optional) Create a GCS bucket for lossless Veo output
```bash
gcloud storage buckets create gs://your-bucket-name --location=us-central1
```

### 5. Use in ComfyUI
Set `gcp_project_id` on any DIGIT node, or leave it blank if running on a GCP instance (auto-detected via metadata service).

---

## Project Folder Structure

The Image Saver, Video Saver, and Image Loader nodes use a VFX-pipeline folder convention:

```
PROJEKTS_ROOT/
  PROJECT_NAME/           (e.g. 25999_comfy_corner)
    shots/
      SHOT_NAME/          (e.g. sh010)
        SUBFOLDER/        (e.g. comfy)
          TASK/           (e.g. comp)
            PREFIX_SHOT_TASK.FRAME.EXT
```

**Example paths:**
```
/mnt/lucid/PROJEKTS/25999_comfy_corner/shots/sh010/comfy/comp/25999_sh010_comp.1001.png
/mnt/lucid/PROJEKTS/25999_comfy_corner/shots/sh010/comfy/comp/25999_sh010_comp.1001.mp4
/mnt/lucid/PROJEKTS/25999_comfy_corner/assets/auto_srt/dialogue.srt
```

The SRT Maker saves to `PROJECT/assets/auto_srt/` instead of the shots hierarchy.

**Supported PROJEKTS roots:**
- `/Volumes/saint/goose/PROJEKTS` (macOS)
- `/mnt/lucid/PROJEKTS` (Linux)

Project folders must follow the `#####_name` pattern (5-digit prefix) to appear in the dropdown menus.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `google-genai` | Official Google GenAI SDK for Gemini and Veo via Vertex AI |
| `google-auth` | GCP authentication and credential management |
| `google-cloud-storage` | GCS bucket access for lossless Veo output |
| `piexif` | EXIF metadata embedding in JPEG files |
| `opencv-python` | EXR file reading and writing |
| `requests` | HTTP requests for LLM Query node |

---

## License

MIT

---

**Built by [DIGIT](https://github.com/thedepartmentofexternalservices/comfyui-digit)**

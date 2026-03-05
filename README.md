# ComfyUI DIGIT Nodes

Custom nodes for ComfyUI that connect directly to Google Cloud Vertex AI for image generation, video generation, and LLM queries. No proxy — your usage bills directly to your GCP account.

## Nodes

### DIGIT Gemini Image
Unified image generation using Gemini models via Vertex AI.

- **Models**: Gemini 3.1 Flash Image (Nano Banana 2), Gemini 3 Pro Image (Nano Banana Pro), Gemini 2.5 Flash Image
- **Features**: Text-to-image, image-to-image editing, up to 3 batched image inputs, configurable aspect ratios, resolution (1K/2K/4K), seed control, safety thresholds, system instructions
- **Output**: IMAGE + TEXT

### DIGIT Veo Video
Unified video generation using Veo models via Vertex AI.

- **Models**: Veo 3.1 (standard + fast), Veo 3.0 (standard + fast), Veo 2.0
- **Modes**: Text-to-video, image-to-video (first frame), first+last frame interpolation, reference images (up to 3)
- **Features**: Audio generation, 720p/1080p resolution, 4-8 second duration, negative prompts, person generation control, batch generation (1-4 samples), optimized or lossless compression with GCS output
- **Output**: VIDEO + VEO_PATHS + STRING

### DIGIT LLM Query
Text generation using Gemini LLM models via Vertex AI.

- **Models**: Gemini 3.1 Pro, Gemini 2.5 Pro/Flash/Flash-Lite
- **Features**: Optional image input for vision queries, system prompts, temperature control
- **Output**: STRING

### DIGIT Image Saver
Save images to a project/shot folder structure with auto-incrementing frame numbers.

- **Formats**: PNG, JPEG, EXR
- **Features**: Project/shot/task folder hierarchy, frame numbering, workflow metadata sidecars, EXR tone mapping (linear/sRGB/Reinhard)

### DIGIT Video Saver
Save videos to the same project/shot folder structure.

- **Features**: Accepts single VIDEO or batch VEO_PATHS input, auto-incrementing frame numbers, workflow metadata sidecars

### DIGIT Image Loader
Load images from the project/shot folder structure.

## Installation

### From ComfyUI Manager
Search for "DIGIT" in the ComfyUI Manager node list.

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/thedepartmentofexternalservices/comfyui-digit.git
cd comfyui-digit
pip install -r requirements.txt
```

## Requirements

- Google Cloud project with Vertex AI API enabled
- Authentication via `gcloud auth application-default login` (local) or GCP metadata service (Compute Engine/GKE)
- Python packages: `google-genai`, `google-auth`, `google-cloud-storage`, `piexif`, `opencv-python`

## Setup

1. Install the Google Cloud SDK: https://cloud.google.com/sdk/docs/install
2. Authenticate:
   ```bash
   gcloud auth application-default login
   gcloud config set project YOUR_PROJECT_ID
   gcloud auth application-default set-quota-project YOUR_PROJECT_ID
   ```
3. In ComfyUI, set `gcp_project_id` on any DIGIT node (or leave blank to auto-detect on GCP instances)

## Project Folder Structure

The image and video saver nodes save to a structured path:

```
PROJEKTS_ROOT/
  PROJECT_NAME/
    shots/
      SHOT_NAME/
        SUBFOLDER/
          TASK/
            PREFIX_SHOT_TASK.FRAME.EXT
```

Example: `25999_comfy_corner/shots/sh010/comfy/comp/25999_sh010_comp.1001.mp4`

## License

MIT

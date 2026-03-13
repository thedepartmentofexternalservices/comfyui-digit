"""DIGIT LoRA Loader — loads LoRAs with trigger word and metadata extraction.

All metadata is read from the safetensors header (no external files required).
Supports AI Toolkit, Kohya/sd-scripts, SimpleTuner, and modelspec formats.
"""

import json
import logging
import os
import struct

import comfy.sd
import comfy.utils
import folder_paths

logger = logging.getLogger(__name__)


def _read_safetensors_metadata(filepath):
    """Read only the metadata header from a safetensors file (no weight loading)."""
    try:
        with open(filepath, "rb") as f:
            header_bytes = f.read(8)
            if len(header_bytes) < 8:
                return {}
            header_size = struct.unpack("<Q", header_bytes)[0]
            if header_size > 100_000_000:
                return {}
            header_json = f.read(header_size)
            header = json.loads(header_json.decode("utf-8"))
            return header.get("__metadata__", {})
    except Exception as e:
        logger.warning("Failed to read safetensors metadata: %s", e)
        return {}


def _extract_trigger_words(metadata):
    """Extract trigger words from safetensors metadata.

    Priority order:
      1. modelspec.trigger_phrase (SimpleTuner, some AI Toolkit versions)
      2. ss_tag_frequency — AI Toolkit stores {"1_TRIGGER": {"TRIGGER": 1}}
      3. ss_training_comment (if it mentions "trigger")
    """
    # 1. modelspec.trigger_phrase
    phrase = metadata.get("modelspec.trigger_phrase", "")
    if phrase:
        return phrase.strip()

    # 2. ss_tag_frequency — the most reliable source for AI Toolkit LoRAs
    #    Format: {"dataset_prefix": {"tag": count, ...}}
    #    AI Toolkit: {"1_TRIGGERNAME": {"TRIGGERNAME": 1}}
    #    Kohya: {"dataset": {"tag1": 50, "tag2": 30, ...}}
    tag_freq_raw = metadata.get("ss_tag_frequency", "")
    if tag_freq_raw:
        try:
            tag_freq = json.loads(tag_freq_raw) if isinstance(tag_freq_raw, str) else tag_freq_raw
            if isinstance(tag_freq, dict):
                for _dataset_key, tags in tag_freq.items():
                    if isinstance(tags, dict) and tags:
                        # Return the highest-frequency tag (the trigger word)
                        top_tag = max(tags, key=tags.get)
                        return top_tag.strip()
        except (json.JSONDecodeError, TypeError):
            pass

    # 3. ss_training_comment
    comment = metadata.get("ss_training_comment", "")
    if comment and "trigger" in comment.lower():
        return comment.strip()

    return ""


def _parse_json_field(metadata, key):
    """Safely parse a JSON string field from metadata, returning the parsed object or None."""
    raw = metadata.get(key, "")
    if not raw:
        return None
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return None


def _format_metadata_summary(metadata):
    """Build a human-readable metadata summary from safetensors header metadata."""
    lines = []

    # --- Name / identity ---
    name = metadata.get("name", "") or metadata.get("ss_output_name", "")
    if name:
        lines.append(f"Name: {name}")

    # --- Base model ---
    base = (metadata.get("ss_base_model_version", "")
            or metadata.get("ss_sd_model_name", "")
            or metadata.get("modelspec.architecture", ""))
    if base:
        lines.append(f"Base Model: {base}")

    # --- Network / rank ---
    dim = metadata.get("ss_network_dim", "")
    alpha = metadata.get("ss_network_alpha", "")
    module = metadata.get("ss_network_module", "")
    if dim:
        rank_str = f"{dim}"
        if alpha:
            rank_str += f" (alpha={alpha})"
        if module:
            rank_str += f" [{module}]"
        lines.append(f"Rank: {rank_str}")

    # --- Training params ---
    field_map = [
        ("ss_learning_rate", "Learning Rate"),
        ("ss_unet_lr", "UNet LR"),
        ("ss_text_encoder_lr", "Text Encoder LR"),
        ("ss_optimizer", "Optimizer"),
        ("ss_lr_scheduler", "LR Scheduler"),
        ("ss_max_train_steps", "Max Steps"),
        ("ss_steps", "Steps"),
        ("ss_num_epochs", "Epochs"),
        ("ss_epoch", "Epoch"),
        ("ss_num_train_images", "Train Images"),
        ("ss_mixed_precision", "Mixed Precision"),
        ("ss_resolution", "Resolution"),
        ("ss_clip_skip", "CLIP Skip"),
        ("ss_seed", "Seed"),
        ("ss_training_comment", "Comment"),
        ("modelspec.title", "Title"),
        ("modelspec.description", "Description"),
        ("modelspec.resolution", "Resolution (modelspec)"),
    ]
    for key, label in field_map:
        val = metadata.get(key, "")
        if val:
            lines.append(f"{label}: {val}")

    # --- AI Toolkit training_info (step/epoch from checkpoint) ---
    training_info = _parse_json_field(metadata, "training_info")
    if training_info:
        step = training_info.get("step", "")
        epoch = training_info.get("epoch", "")
        if step:
            checkpoint_str = str(step)
            if epoch:
                checkpoint_str += f" (epoch {epoch})"
            lines.append(f"Checkpoint Step: {checkpoint_str}")

    # --- Software ---
    software = _parse_json_field(metadata, "software")
    if software and isinstance(software, dict):
        sw_name = software.get("name", "")
        sw_ver = software.get("version", "")
        if sw_name:
            lines.append(f"Software: {sw_name} {sw_ver}".strip())
    elif metadata.get("software"):
        lines.append(f"Software: {metadata['software']}")

    # --- Hashes ---
    model_hash = metadata.get("sshs_model_hash", "")
    if model_hash:
        lines.append(f"Hash: {model_hash[:16]}...")

    # --- Fallback: dump all keys if nothing matched above ---
    if not lines:
        for k, v in sorted(metadata.items()):
            if k.startswith("__") or len(str(v)) > 500:
                continue
            lines.append(f"{k}: {v}")

    return "\n".join(lines) if lines else "(no training metadata found)"


def _build_info_line(metadata, trigger_words, strength_model, strength_clip, lora_name):
    """Build a compact one-line summary."""
    parts = [os.path.basename(lora_name)]

    base = metadata.get("ss_base_model_version", "") or metadata.get("ss_sd_model_name", "")
    if base:
        for prefix in ("black-forest-labs/", "stabilityai/", "runwayml/"):
            if base.startswith(prefix):
                base = base[len(prefix):]
        parts.append(f"base={base}")

    rank = metadata.get("ss_network_dim", "")
    if rank:
        parts.append(f"rank={rank}")

    if trigger_words:
        parts.append(f'trigger="{trigger_words}"')

    parts.append(f"str={strength_model:.2f}/{strength_clip:.2f}")
    return " | ".join(parts)


class DigitLoraLoader:
    """Load a LoRA with automatic trigger word and metadata extraction."""

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The LoRA file to load."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model."}),
            },
            "optional": {
                "trigger_words_override": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Manually specify trigger words. Overrides auto-detected triggers when set.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "trigger_words", "metadata", "lora_info")
    OUTPUT_TOOLTIPS = (
        "The modified diffusion model.",
        "The modified CLIP model.",
        "Auto-detected or manually specified trigger words. Wire to Prompt Combine.",
        "Training metadata summary from safetensors header.",
        "Compact one-line summary: name, base model, rank, trigger.",
    )
    FUNCTION = "load_lora"
    CATEGORY = "DIGIT"
    DESCRIPTION = "Load a LoRA and extract trigger words and training metadata from the safetensors header."
    OUTPUT_NODE = True

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip,
                  trigger_words_override=""):
        if strength_model == 0 and strength_clip == 0:
            return {"ui": {"trigger_text": ["(disabled)"], "info_text": [""]},
                    "result": (model, clip, "", "", "")}

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)

        # --- Read safetensors metadata (header only, no weight loading) ---
        metadata = {}
        if lora_path.lower().endswith(".safetensors"):
            metadata = _read_safetensors_metadata(lora_path)

        # --- Trigger words ---
        if trigger_words_override.strip():
            trigger_words = trigger_words_override.strip()
        else:
            trigger_words = _extract_trigger_words(metadata)

        # --- Metadata summary ---
        meta_summary = _format_metadata_summary(metadata)

        # --- Compact info line ---
        lora_info = _build_info_line(metadata, trigger_words,
                                     strength_model, strength_clip, lora_name)

        # --- Load and apply LoRA ---
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)

        logger.info("DIGIT LoRA Loader: %s (strength %.2f/%.2f, trigger: %s)",
                     lora_name, strength_model, strength_clip, trigger_words or "(none)")

        return {"ui": {"trigger_text": [trigger_words or "(none)"],
                       "info_text": [lora_info]},
                "result": (model_lora, clip_lora, trigger_words, meta_summary, lora_info)}


class DigitLoraLoaderModelOnly(DigitLoraLoader):
    """Load a LoRA (model only, no CLIP patching) with trigger word and metadata extraction."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The LoRA file to load."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model."}),
            },
            "optional": {
                "trigger_words_override": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Manually specify trigger words. Overrides auto-detected triggers when set.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("model", "trigger_words", "metadata", "lora_info")
    OUTPUT_TOOLTIPS = (
        "The modified diffusion model.",
        "Auto-detected or manually specified trigger words.",
        "Training metadata summary from safetensors header.",
        "Compact one-line summary.",
    )
    DESCRIPTION = "Load a LoRA (model only) and extract trigger words and training metadata."

    def load_lora_model_only(self, model, lora_name, strength_model,
                             trigger_words_override=""):
        result = self.load_lora(model, None, lora_name, strength_model, 0,
                                trigger_words_override)
        ui = result["ui"]
        r = result["result"]
        # r = (model, clip, trigger_words, metadata, lora_info) — drop clip
        return {"ui": ui, "result": (r[0], r[2], r[3], r[4])}

    FUNCTION = "load_lora_model_only"

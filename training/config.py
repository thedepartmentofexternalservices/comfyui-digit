"""YAML-based training configuration with validation."""

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ModelConfig:
    name: str = "flux1-dev"
    base_model_path: str = "black-forest-labs/FLUX.1-dev"
    revision: Optional[str] = None
    dtype: str = "bfloat16"
    # Model-specific overrides
    text_encoder_dtype: str = "bfloat16"
    vae_dtype: str = "float32"


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    # Target modules vary by model architecture
    target_modules: list = field(default_factory=lambda: [
        "to_q", "to_k", "to_v", "to_out.0",
        "proj_in", "proj_out",
        "ff.net.0.proj", "ff.net.2",
    ])
    # Advanced
    use_dora: bool = False
    use_rslora: bool = False


@dataclass
class DatasetConfig:
    path: str = ""
    resolution: int = 1024
    center_crop: bool = True
    random_flip: bool = False
    caption_ext: str = ".txt"
    # Bucketing for variable aspect ratios
    use_bucketing: bool = True
    bucket_step: int = 64
    min_bucket_resolution: int = 512
    max_bucket_resolution: int = 2048


@dataclass
class OptimizerConfig:
    name: str = "adamw8bit"  # adamw, adamw8bit, prodigy, adafactor
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: list = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    # Prodigy-specific
    prodigy_d_coef: float = 1.0
    prodigy_growth_rate: float = float("inf")


@dataclass
class SchedulerConfig:
    name: str = "cosine"  # constant, cosine, cosine_with_restarts, linear
    warmup_steps: int = 100
    num_cycles: int = 1


@dataclass
class TrainingConfig:
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_train_steps: int = 1000
    max_train_epochs: Optional[int] = None
    seed: int = 42
    mixed_precision: str = "bf16"  # no, fp16, bf16
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    # EMA
    use_ema: bool = False
    ema_decay: float = 0.9999
    # Noise
    noise_offset: float = 0.0
    # Min SNR gamma (0 = disabled)
    min_snr_gamma: float = 0.0


@dataclass
class SamplingConfig:
    enabled: bool = True
    every_n_steps: int = 100
    prompts: list = field(default_factory=lambda: [
        "a photo of [trigger]",
    ])
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    width: int = 1024
    height: int = 1024
    seed: int = 42


@dataclass
class LoggingConfig:
    use_wandb: bool = False
    use_tensorboard: bool = True
    wandb_project: str = "digit-lora"
    wandb_entity: Optional[str] = None
    log_dir: str = "./logs"
    log_every_n_steps: int = 10


@dataclass
class CaptionConfig:
    """Config for Gemini-based auto-captioning."""
    model: str = "gemini-2.5-flash"
    system_prompt: str = (
        "You are an expert image captioner for AI training datasets. "
        "Describe the image in detail, focusing on subject, composition, "
        "lighting, colors, style, and mood. Be specific and descriptive. "
        "Output only the caption, no preamble."
    )
    prompt_template: str = "Describe this image in detail for AI training:"
    max_tokens: int = 300
    temperature: float = 0.4
    batch_size: int = 5
    # Rate limiting
    requests_per_minute: int = 30
    # Caption file extension
    output_ext: str = ".txt"


@dataclass
class NamingConfig:
    """Naming convention templates with variable substitution.

    Available variables:
        {name}       — run name
        {model}      — model type (e.g. flux1-dev)
        {trigger}    — trigger word
        {rank}       — LoRA rank
        {alpha}      — LoRA alpha
        {steps}      — max training steps
        {res}        — resolution
        {date}       — YYYY-MM-DD
        {datetime}   — YYYY-MM-DD_HH-MM-SS
        {seed}       — training seed
        {optimizer}  — optimizer name
        {lr}         — learning rate (formatted)
    """
    # Template for the output directory name
    output_dir_template: str = "{name}_{model}_r{rank}_{date}"
    # Template for the LoRA weight file/folder name
    lora_name_template: str = "{name}_{model}_r{rank}_s{steps}"
    # Template for checkpoint directory names
    checkpoint_template: str = "step_{step:06d}"
    # Template for sample image filenames
    sample_template: str = "sample_{prompt_index:02d}_{step:06d}"


@dataclass
class TriggerConfig:
    """Trigger word configuration with class/instance support."""
    # Primary trigger word (e.g. "ohwx")
    trigger_word: str = ""
    # Class noun for the subject (e.g. "person", "man", "woman", "dog", "style")
    trigger_class: str = ""
    # Combined form used in captions (auto-generated if empty: "{trigger_word} {trigger_class}")
    trigger_phrase: str = ""


@dataclass
class OutputConfig:
    output_dir: str = "./output"
    save_every_n_steps: int = 500
    save_final: bool = True
    lora_name: str = "digit_lora"


@dataclass
class DigitTrainingConfig:
    """Top-level training configuration."""
    name: str = "untitled"
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    caption: CaptionConfig = field(default_factory=CaptionConfig)
    naming: NamingConfig = field(default_factory=NamingConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def get_naming_vars(self) -> dict:
        """Return all available variables for naming template substitution."""
        from datetime import datetime
        now = datetime.now()
        lr = self.optimizer.learning_rate
        # Format LR nicely: 1e-4 -> "1e-4", 0.0004 -> "4e-4"
        lr_str = f"{lr:.0e}".replace("+", "")
        return {
            "name": self.name,
            "model": self.model.name,
            "trigger": self.trigger.trigger_word or "notrigger",
            "trigger_class": self.trigger.trigger_class or "subject",
            "rank": self.lora.rank,
            "alpha": int(self.lora.alpha),
            "steps": self.training.max_train_steps,
            "res": self.dataset.resolution,
            "date": now.strftime("%Y-%m-%d"),
            "datetime": now.strftime("%Y-%m-%d_%H-%M-%S"),
            "seed": self.training.seed,
            "optimizer": self.optimizer.name,
            "lr": lr_str,
        }

    def resolve_naming(self) -> dict:
        """Resolve all naming templates to concrete strings."""
        v = self.get_naming_vars()
        return {
            "output_dir": self.naming.output_dir_template.format(**v),
            "lora_name": self.naming.lora_name_template.format(**v),
        }

    def inject_trigger(self, text: str) -> str:
        """Replace trigger placeholders in a string.

        Supported placeholders:
            [trigger]       — trigger_word
            [trigger_class] — trigger_class (e.g. "person")
            [trigger_phrase] — "trigger_word trigger_class" combined
        """
        tw = self.trigger.trigger_word
        tc = self.trigger.trigger_class
        tp = self.trigger.trigger_phrase
        if not tp and tw:
            tp = f"{tw} {tc}".strip() if tc else tw

        text = text.replace("[trigger]", tw)
        text = text.replace("[trigger_class]", tc)
        text = text.replace("[trigger_phrase]", tp)
        return text


def _merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_config(data: dict) -> DigitTrainingConfig:
    """Convert a flat/nested dict to DigitTrainingConfig."""
    cfg = DigitTrainingConfig()
    sub_configs = {
        "model": (ModelConfig, cfg.model),
        "lora": (LoRAConfig, cfg.lora),
        "dataset": (DatasetConfig, cfg.dataset),
        "optimizer": (OptimizerConfig, cfg.optimizer),
        "scheduler": (SchedulerConfig, cfg.scheduler),
        "training": (TrainingConfig, cfg.training),
        "sampling": (SamplingConfig, cfg.sampling),
        "logging": (LoggingConfig, cfg.logging),
        "caption": (CaptionConfig, cfg.caption),
        "naming": (NamingConfig, cfg.naming),
        "trigger": (TriggerConfig, cfg.trigger),
        "output": (OutputConfig, cfg.output),
    }

    if "name" in data:
        cfg.name = data["name"]

    for section_name, (cls, default_instance) in sub_configs.items():
        if section_name in data and isinstance(data[section_name], dict):
            for k, v in data[section_name].items():
                if hasattr(default_instance, k):
                    setattr(default_instance, k, v)

    return cfg


# Predefined model presets
MODEL_PRESETS = {
    "flux1-dev": {
        "model": {
            "name": "flux1-dev",
            "base_model_path": "black-forest-labs/FLUX.1-dev",
            "dtype": "bfloat16",
        },
        "lora": {
            "target_modules": [
                "to_q", "to_k", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "ff.net.0.proj", "ff.net.2",
            ],
        },
        "training": {
            "mixed_precision": "bf16",
        },
        "sampling": {
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
        },
    },
    "flux1-schnell": {
        "model": {
            "name": "flux1-schnell",
            "base_model_path": "black-forest-labs/FLUX.1-schnell",
            "dtype": "bfloat16",
        },
        "lora": {
            "target_modules": [
                "to_q", "to_k", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "ff.net.0.proj", "ff.net.2",
            ],
        },
        "training": {
            "mixed_precision": "bf16",
        },
        "sampling": {
            "num_inference_steps": 4,
            "guidance_scale": 0.0,
        },
    },
    "flux2": {
        "model": {
            "name": "flux2",
            "base_model_path": "black-forest-labs/FLUX.2-dev",
            "dtype": "bfloat16",
        },
        "lora": {
            "target_modules": [
                "to_q", "to_k", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "ff.net.0.proj", "ff.net.2",
            ],
        },
        "training": {
            "mixed_precision": "bf16",
        },
        "sampling": {
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
        },
    },
    "flux2-klein": {
        "model": {
            "name": "flux2-klein",
            "base_model_path": "black-forest-labs/FLUX.2-klein",
            "dtype": "bfloat16",
        },
        "lora": {
            "rank": 8,
            "target_modules": [
                "to_q", "to_k", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "ff.net.0.proj", "ff.net.2",
            ],
        },
        "training": {
            "mixed_precision": "bf16",
        },
        "sampling": {
            "num_inference_steps": 20,
            "guidance_scale": 3.0,
        },
    },
    "qwen": {
        "model": {
            "name": "qwen",
            "base_model_path": "Qwen/Qwen2.5-VL-7B",
            "dtype": "bfloat16",
        },
        "lora": {
            "rank": 16,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        },
        "training": {
            "mixed_precision": "bf16",
        },
    },
}


def load_config(path: str, model_preset: str = None) -> DigitTrainingConfig:
    """Load a training config from YAML, optionally applying a model preset first."""
    base = {}

    # Apply model preset as base if specified
    if model_preset and model_preset in MODEL_PRESETS:
        base = MODEL_PRESETS[model_preset].copy()

    # Load YAML overrides
    if os.path.exists(path):
        with open(path, "r") as f:
            yaml_data = yaml.safe_load(f) or {}
        base = _merge_dicts(base, yaml_data)

    return _dict_to_config(base)


def save_config(config: DigitTrainingConfig, path: str):
    """Save a training config to YAML."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)

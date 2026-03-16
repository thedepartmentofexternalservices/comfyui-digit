"""DIGIT LoRA Trainer — ComfyUI node for training LoRA adapters."""

import json
import os
import threading
import time

# Track active training threads
_active_training = {}
_training_lock = threading.Lock()


class DigitLoRATrainer:
    """Train LoRA adapters for Flux and Qwen models."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": ([
                    "train",
                    "stop",
                    "status",
                    "load_preset",
                    "save_preset",
                    "list_presets",
                    "list_runs",
                ], {"default": "train"}),
                "dataset_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to captioned dataset",
                }),
            },
            "optional": {
                "run_name": ("STRING", {
                    "default": "my_lora",
                    "multiline": False,
                }),
                "config_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to YAML config (optional)",
                }),
                "preset_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Name of saved preset to use/save",
                }),
                "model_type": ([
                    "flux1-dev",
                    "flux1-schnell",
                    "flux2",
                    "flux2-klein",
                    "qwen",
                ], {"default": "flux1-dev"}),
                "base_model_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "HuggingFace ID or local path (uses default if empty)",
                }),
                "output_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Output directory for LoRA weights",
                }),
                "trigger_word": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Trigger word (e.g., 'ohwx')",
                }),
                "trigger_class": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Class noun (e.g., 'person', 'man', 'woman', 'style')",
                }),
                "trigger_preset": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Load trigger from saved preset",
                }),
                "naming_preset": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Load naming convention from saved preset",
                }),
                "sample_prompt_preset": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Load sample prompts from saved preset",
                }),
                "lora_rank": ("INT", {
                    "default": 16, "min": 1, "max": 256, "step": 1,
                }),
                "lora_alpha": ("FLOAT", {
                    "default": 16.0, "min": 0.1, "max": 256.0, "step": 0.1,
                }),
                "learning_rate": ("FLOAT", {
                    "default": 1e-4,
                    "min": 1e-7, "max": 1e-1,
                    "step": 1e-5,
                }),
                "max_train_steps": ("INT", {
                    "default": 1000, "min": 10, "max": 100000, "step": 10,
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 32, "step": 1,
                }),
                "resolution": ("INT", {
                    "default": 1024, "min": 256, "max": 2048, "step": 64,
                }),
                "gradient_accumulation": ("INT", {
                    "default": 1, "min": 1, "max": 128, "step": 1,
                }),
                "optimizer": ([
                    "adamw8bit", "adamw", "prodigy", "adafactor",
                ], {"default": "adamw8bit"}),
                "lr_scheduler": ([
                    "cosine", "constant", "linear", "cosine_with_restarts",
                ], {"default": "cosine"}),
                "warmup_steps": ("INT", {
                    "default": 100, "min": 0, "max": 10000, "step": 10,
                }),
                "sample_every_n_steps": ("INT", {
                    "default": 100, "min": 0, "max": 10000, "step": 10,
                }),
                "sample_prompts": ("STRING", {
                    "default": "a photo of [trigger]",
                    "multiline": True,
                    "placeholder": "One prompt per line for sample generation",
                }),
                "save_every_n_steps": ("INT", {
                    "default": 500, "min": 0, "max": 50000, "step": 100,
                }),
                "use_wandb": ("BOOLEAN", {"default": False}),
                "wandb_project": ("STRING", {"default": "digit-lora"}),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 2**32 - 1,
                }),
                "noise_offset": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01,
                }),
                "use_bucketing": ("BOOLEAN", {"default": True}),
                "random_flip": ("BOOLEAN", {"default": False}),
                "gradient_checkpointing": ("BOOLEAN", {"default": True}),
                "use_dora": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("status", "output_path",)
    FUNCTION = "execute"
    CATEGORY = "DIGIT"
    OUTPUT_NODE = True

    def execute(self, action, dataset_path, run_name="my_lora",
                config_path="", preset_name="", model_type="flux1-dev",
                base_model_path="", output_dir="", trigger_word="",
                trigger_class="", trigger_preset="",
                naming_preset="", sample_prompt_preset="",
                lora_rank=16, lora_alpha=16.0, learning_rate=1e-4,
                max_train_steps=1000, batch_size=1, resolution=1024,
                gradient_accumulation=1, optimizer="adamw8bit",
                lr_scheduler="cosine", warmup_steps=100,
                sample_every_n_steps=100, sample_prompts="a photo of [trigger]",
                save_every_n_steps=500, use_wandb=False,
                wandb_project="digit-lora", seed=42, noise_offset=0.0,
                use_bucketing=True, random_flip=False,
                gradient_checkpointing=True, use_dora=False):

        if action == "status":
            return self._get_status(run_name)
        elif action == "stop":
            return self._stop_training(run_name)
        elif action == "list_runs":
            return self._list_runs()
        elif action == "list_presets":
            return self._list_presets(model_type)
        elif action == "save_preset":
            return self._save_preset(
                preset_name or run_name, model_type,
                locals(),
            )
        elif action == "load_preset":
            return self._load_preset(preset_name)
        elif action == "train":
            return self._start_training(
                run_name=run_name,
                dataset_path=dataset_path,
                config_path=config_path,
                preset_name=preset_name,
                model_type=model_type,
                base_model_path=base_model_path,
                output_dir=output_dir,
                trigger_word=trigger_word,
                trigger_class=trigger_class,
                trigger_preset=trigger_preset,
                naming_preset=naming_preset,
                sample_prompt_preset=sample_prompt_preset,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                learning_rate=learning_rate,
                max_train_steps=max_train_steps,
                batch_size=batch_size,
                resolution=resolution,
                gradient_accumulation=gradient_accumulation,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                warmup_steps=warmup_steps,
                sample_every_n_steps=sample_every_n_steps,
                sample_prompts=sample_prompts,
                save_every_n_steps=save_every_n_steps,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                seed=seed,
                noise_offset=noise_offset,
                use_bucketing=use_bucketing,
                random_flip=random_flip,
                gradient_checkpointing=gradient_checkpointing,
                use_dora=use_dora,
            )

        return ("Unknown action", "")

    def _build_config(self, **kwargs):
        """Build a DigitTrainingConfig from node parameters."""
        from .training.config import DigitTrainingConfig, load_config, MODEL_PRESETS
        from .training.presets_db import PresetsDB

        db = PresetsDB()
        config_path = kwargs.get("config_path", "")
        preset_name = kwargs.get("preset_name", "")
        model_type = kwargs.get("model_type", "flux1-dev")

        # Start from YAML config or model preset
        if config_path and os.path.exists(config_path):
            config = load_config(config_path, model_preset=model_type)
        elif preset_name:
            preset = db.get_training_preset(preset_name)
            if preset:
                from .training.config import _dict_to_config
                config = _dict_to_config(preset["config"])
            else:
                config = load_config("", model_preset=model_type)
        else:
            config = load_config("", model_preset=model_type)

        # Override with node parameters
        config.name = kwargs.get("run_name", config.name)

        if kwargs.get("base_model_path"):
            config.model.base_model_path = kwargs["base_model_path"]

        config.dataset.path = kwargs.get("dataset_path", config.dataset.path)
        config.dataset.resolution = kwargs.get("resolution", config.dataset.resolution)
        config.dataset.use_bucketing = kwargs.get("use_bucketing", config.dataset.use_bucketing)
        config.dataset.random_flip = kwargs.get("random_flip", config.dataset.random_flip)

        config.lora.rank = kwargs.get("lora_rank", config.lora.rank)
        config.lora.alpha = kwargs.get("lora_alpha", config.lora.alpha)
        config.lora.use_dora = kwargs.get("use_dora", config.lora.use_dora)

        config.optimizer.name = kwargs.get("optimizer", config.optimizer.name)
        config.optimizer.learning_rate = kwargs.get("learning_rate", config.optimizer.learning_rate)

        config.scheduler.name = kwargs.get("lr_scheduler", config.scheduler.name)
        config.scheduler.warmup_steps = kwargs.get("warmup_steps", config.scheduler.warmup_steps)

        config.training.max_train_steps = kwargs.get("max_train_steps", config.training.max_train_steps)
        config.training.batch_size = kwargs.get("batch_size", config.training.batch_size)
        config.training.gradient_accumulation_steps = kwargs.get("gradient_accumulation", config.training.gradient_accumulation_steps)
        config.training.seed = kwargs.get("seed", config.training.seed)
        config.training.noise_offset = kwargs.get("noise_offset", config.training.noise_offset)
        config.training.gradient_checkpointing = kwargs.get("gradient_checkpointing", config.training.gradient_checkpointing)

        # --- Trigger word resolution ---
        # Priority: trigger_preset > node params > existing config
        trigger_preset_name = kwargs.get("trigger_preset", "")
        if trigger_preset_name:
            tp = db.get_trigger_preset(trigger_preset_name)
            if tp:
                config.trigger.trigger_word = tp["trigger_word"]
                config.trigger.trigger_class = tp.get("trigger_class", "")
                config.trigger.trigger_phrase = tp.get("trigger_phrase", "")
        else:
            tw = kwargs.get("trigger_word", "")
            tc = kwargs.get("trigger_class", "")
            if tw:
                config.trigger.trigger_word = tw
            if tc:
                config.trigger.trigger_class = tc
            # Auto-build phrase if not explicitly set
            if config.trigger.trigger_word and not config.trigger.trigger_phrase:
                if config.trigger.trigger_class:
                    config.trigger.trigger_phrase = f"{config.trigger.trigger_word} {config.trigger.trigger_class}"
                else:
                    config.trigger.trigger_phrase = config.trigger.trigger_word

        # --- Naming convention resolution ---
        naming_preset_name = kwargs.get("naming_preset", "")
        if naming_preset_name:
            np = db.get_naming_preset(naming_preset_name)
            if np:
                config.naming.output_dir_template = np["output_dir_template"]
                config.naming.lora_name_template = np["lora_name_template"]
                config.naming.checkpoint_template = np.get("checkpoint_template", config.naming.checkpoint_template)
                config.naming.sample_template = np.get("sample_template", config.naming.sample_template)

        # --- Sample prompts resolution ---
        # Priority: sample_prompt_preset > node sample_prompts > existing config
        sample_preset_name = kwargs.get("sample_prompt_preset", "")
        if sample_preset_name:
            sp = db.get_sample_prompt_preset(sample_preset_name)
            if sp:
                config.sampling.prompts = sp["prompts"]
        else:
            prompts_text = kwargs.get("sample_prompts", "")
            if prompts_text:
                config.sampling.prompts = [p.strip() for p in prompts_text.split("\n") if p.strip()]

        # Inject trigger placeholders into all sample prompts
        config.sampling.prompts = [
            config.inject_trigger(p) for p in config.sampling.prompts
        ]

        config.sampling.every_n_steps = kwargs.get("sample_every_n_steps", config.sampling.every_n_steps)
        config.sampling.enabled = config.sampling.every_n_steps > 0

        config.logging.use_wandb = kwargs.get("use_wandb", config.logging.use_wandb)
        config.logging.wandb_project = kwargs.get("wandb_project", config.logging.wandb_project)

        # --- Output directory resolution ---
        # Resolve naming template → output dir and lora name
        resolved = config.resolve_naming()
        output_dir = kwargs.get("output_dir", "")
        if not output_dir:
            base_output = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "output",
            )
            output_dir = os.path.join(base_output, resolved["output_dir"])

        config.output.output_dir = output_dir
        config.output.lora_name = resolved["lora_name"]
        config.output.save_every_n_steps = kwargs.get("save_every_n_steps", config.output.save_every_n_steps)

        config.logging.log_dir = os.path.join(output_dir, "logs")

        return config

    def _start_training(self, **kwargs):
        """Start training in a background thread."""
        run_name = kwargs.get("run_name", "my_lora")

        with _training_lock:
            if run_name in _active_training:
                info = _active_training[run_name]
                if info.get("thread") and info["thread"].is_alive():
                    return (f"Training '{run_name}' is already running. Use 'stop' to cancel.", "")

        # Build config
        config = self._build_config(**kwargs)

        # Save config to output dir
        from .training.config import save_config
        os.makedirs(config.output.output_dir, exist_ok=True)
        config_save_path = os.path.join(config.output.output_dir, "training_config.yaml")
        save_config(config, config_save_path)

        # Start training in background
        training_state = {
            "status": "starting",
            "step": 0,
            "total_steps": config.training.max_train_steps,
            "loss": 0.0,
            "lr": config.optimizer.learning_rate,
            "trainer": None,
            "thread": None,
            "error": None,
            "output_dir": config.output.output_dir,
        }

        def _train_thread():
            try:
                from .training.lora_trainer import LoRATrainer
                from .training.presets_db import PresetsDB

                db = PresetsDB()
                trainer = LoRATrainer(config, db=db)
                training_state["trainer"] = trainer
                training_state["status"] = "loading_model"

                trainer.setup()
                training_state["status"] = "training"

                def on_progress(step, total, loss, lr):
                    training_state["step"] = step
                    training_state["total_steps"] = total
                    training_state["loss"] = loss
                    training_state["lr"] = lr

                result = trainer.train(progress_callback=on_progress)
                training_state["status"] = "completed"
                training_state["result"] = result

            except Exception as e:
                training_state["status"] = "error"
                training_state["error"] = str(e)
                import traceback
                traceback.print_exc()
            finally:
                if training_state.get("trainer"):
                    training_state["trainer"].cleanup()

        thread = threading.Thread(target=_train_thread, daemon=True)
        training_state["thread"] = thread

        with _training_lock:
            _active_training[run_name] = training_state

        thread.start()

        return (
            f"Training '{run_name}' started.\n"
            f"Model: {config.model.name}\n"
            f"Dataset: {config.dataset.path}\n"
            f"Steps: {config.training.max_train_steps}\n"
            f"Output: {config.output.output_dir}\n"
            f"Use action='status' to monitor progress.",
            config.output.output_dir,
        )

    def _get_status(self, run_name):
        """Get training status."""
        with _training_lock:
            if run_name not in _active_training:
                # Check DB for historical runs
                from .training.presets_db import PresetsDB
                db = PresetsDB()
                runs = db.list_runs()
                matching = [r for r in runs if r["name"] == run_name]
                if matching:
                    r = matching[0]
                    status = (
                        f"Run: {r['name']} (historical)\n"
                        f"Status: {r['status']}\n"
                        f"Steps: {r['current_step']}/{r['total_steps']}\n"
                        f"Loss: {r['loss']:.6f}" if r['loss'] else "N/A"
                    )
                    return (status, "")
                return (f"No training found for '{run_name}'", "")

            state = _active_training[run_name]

        pct = (state["step"] / max(state["total_steps"], 1)) * 100
        status = (
            f"Run: {run_name}\n"
            f"Status: {state['status']}\n"
            f"Step: {state['step']}/{state['total_steps']} ({pct:.1f}%)\n"
            f"Loss: {state['loss']:.6f}\n"
            f"LR: {state['lr']:.2e}\n"
        )
        if state.get("error"):
            status += f"Error: {state['error']}\n"

        return (status, state.get("output_dir", ""))

    def _stop_training(self, run_name):
        """Stop an active training run."""
        with _training_lock:
            if run_name not in _active_training:
                return (f"No active training '{run_name}'", "")
            state = _active_training[run_name]

        trainer = state.get("trainer")
        if trainer:
            trainer.request_stop()
            return (f"Stop requested for '{run_name}'. Will save checkpoint and exit.", "")
        return (f"No trainer found for '{run_name}'", "")

    def _list_runs(self):
        """List recent training runs."""
        from .training.presets_db import PresetsDB
        db = PresetsDB()
        runs = db.list_runs()

        # Also include active runs
        with _training_lock:
            active_names = set(_active_training.keys())

        if not runs and not active_names:
            return ("No training runs found.", "")

        lines = ["Training Runs:"]

        # Active runs first
        with _training_lock:
            for name, state in _active_training.items():
                pct = (state["step"] / max(state["total_steps"], 1)) * 100
                lines.append(
                    f"  [ACTIVE] {name}: {state['status']} "
                    f"step {state['step']}/{state['total_steps']} ({pct:.1f}%) "
                    f"loss={state['loss']:.6f}"
                )

        # Historical runs
        for r in runs:
            if r["name"] in active_names:
                continue
            loss_str = f"loss={r['loss']:.6f}" if r['loss'] else ""
            lines.append(
                f"  {r['name']}: {r['status']} "
                f"step {r['current_step']}/{r['total_steps']} {loss_str}"
            )

        return ("\n".join(lines), "")

    def _save_preset(self, preset_name, model_type, params):
        """Save current parameters as a preset."""
        from .training.presets_db import PresetsDB

        config = self._build_config(**params)
        db = PresetsDB()
        db.save_training_preset(
            name=preset_name,
            model_type=model_type,
            config=config.to_dict(),
            description=f"{model_type} LoRA training preset",
        )
        return (f"Saved preset: {preset_name}", "")

    def _load_preset(self, preset_name):
        """Load a preset and return its config."""
        from .training.presets_db import PresetsDB

        db = PresetsDB()
        preset = db.get_training_preset(preset_name)
        if not preset:
            return (f"Preset not found: {preset_name}", "")

        return (json.dumps(preset["config"], indent=2), "")

    def _list_presets(self, model_type):
        """List saved presets."""
        from .training.presets_db import PresetsDB

        db = PresetsDB()
        presets = db.list_training_presets(model_type if model_type != "flux1-dev" else None)

        if not presets:
            return ("No presets saved.", "")

        lines = ["Training Presets:"]
        for p in presets:
            lines.append(f"  {p['name']} ({p['model_type']}) — {p['description']}")

        return ("\n".join(lines), "")


class DigitLoRALoader:
    """Load a trained DIGIT LoRA for use in ComfyUI pipelines."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        lora_files = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "lora_name": (sorted(lora_files), {
                    "tooltip": "Select a LoRA from the ComfyUI loras folder.",
                }),
            },
            "optional": {
                "lora_path_override": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Absolute path (overrides lora_name if set)",
                    "tooltip": "Optional absolute path to a DIGIT LoRA directory or .safetensors file. Takes priority over lora_name.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05,
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("lora_path", "trigger_word", "trigger_class", "trigger_phrase", "metadata",)
    FUNCTION = "execute"
    CATEGORY = "DIGIT"

    def execute(self, lora_name, lora_path_override="", strength=1.0):
        """Load LoRA metadata and return trigger info for downstream nodes."""
        import folder_paths

        # Resolve path: override takes priority, otherwise use the browsed lora_name
        if lora_path_override and lora_path_override.strip():
            lora_path = lora_path_override.strip()
        else:
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)

        if not os.path.exists(lora_path):
            return (lora_path, "", "", "", "LoRA path not found")

        # Check for DIGIT metadata
        # If lora_path is a file, look for metadata in the same directory
        if os.path.isfile(lora_path):
            metadata_path = os.path.join(os.path.dirname(lora_path), "digit_metadata.json")
        else:
            metadata_path = os.path.join(lora_path, "digit_metadata.json")

        trigger_word = ""
        trigger_class = ""
        trigger_phrase = ""
        metadata_str = ""

        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            trigger_word = metadata.get("trigger_word", "")
            trigger_class = metadata.get("trigger_class", "")
            trigger_phrase = metadata.get("trigger_phrase", "")
            metadata["applied_strength"] = strength
            metadata_str = json.dumps(metadata, indent=2)
        else:
            metadata_str = json.dumps({
                "path": lora_path,
                "strength": strength,
                "note": "No DIGIT metadata found — may be an external LoRA",
            }, indent=2)

        return (lora_path, trigger_word, trigger_class, trigger_phrase, metadata_str)

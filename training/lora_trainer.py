"""LoRA trainer for Flux and Qwen models using diffusers + PEFT."""

import gc
import math
import os
import time
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

from .config import DigitTrainingConfig
from .dataset import prepare_dataset
from .logger import TrainingLogger
from .sampler import TrainingSampler


def _get_dtype(dtype_str: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype_str, torch.bfloat16)


def _get_optimizer(params, config: DigitTrainingConfig):
    """Create optimizer from config."""
    opt_cfg = config.optimizer
    name = opt_cfg.name.lower()

    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=opt_cfg.learning_rate,
            weight_decay=opt_cfg.weight_decay,
            betas=tuple(opt_cfg.betas),
            eps=opt_cfg.eps,
        )
    elif name == "adamw8bit":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(
                params,
                lr=opt_cfg.learning_rate,
                weight_decay=opt_cfg.weight_decay,
                betas=tuple(opt_cfg.betas),
                eps=opt_cfg.eps,
            )
        except ImportError:
            print("[DIGIT] bitsandbytes not available, falling back to AdamW")
            return torch.optim.AdamW(
                params,
                lr=opt_cfg.learning_rate,
                weight_decay=opt_cfg.weight_decay,
                betas=tuple(opt_cfg.betas),
                eps=opt_cfg.eps,
            )
    elif name == "prodigy":
        try:
            from prodigyopt import Prodigy
            return Prodigy(
                params,
                lr=opt_cfg.learning_rate,
                weight_decay=opt_cfg.weight_decay,
                betas=tuple(opt_cfg.betas),
                d_coef=opt_cfg.prodigy_d_coef,
                growth_rate=opt_cfg.prodigy_growth_rate,
            )
        except ImportError:
            raise ImportError("prodigyopt required for Prodigy optimizer: pip install prodigyopt")
    elif name == "adafactor":
        from transformers.optimization import Adafactor
        return Adafactor(
            params,
            lr=opt_cfg.learning_rate,
            weight_decay=opt_cfg.weight_decay,
            relative_step=False,
            scale_parameter=False,
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def _get_scheduler(optimizer, config: DigitTrainingConfig, num_training_steps: int):
    """Create LR scheduler from config."""
    from transformers import get_scheduler
    return get_scheduler(
        name=config.scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=config.scheduler.warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=config.scheduler.num_cycles if config.scheduler.name == "cosine_with_restarts" else None,
    )


class LoRATrainer:
    """Main LoRA training loop for Flux and Qwen models."""

    def __init__(self, config: DigitTrainingConfig, db=None):
        self.config = config
        self.db = db
        self.run_id = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._stop_requested = False

        # Will be set during setup
        self.pipeline = None
        self.text_encoders = None
        self.vae = None
        self.transformer = None
        self.tokenizers = None
        self.noise_scheduler = None
        self.optimizer = None
        self.lr_scheduler = None
        self.logger = None
        self.sampler = None

    def request_stop(self):
        """Request graceful stop of training."""
        self._stop_requested = True

    def _load_flux_model(self):
        """Load a Flux model (1 or 2) for training."""
        from diffusers import FluxPipeline

        model_path = self.config.model.base_model_path
        dtype = _get_dtype(self.config.model.dtype)

        print(f"[DIGIT] Loading Flux model: {model_path}")
        pipe = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            revision=self.config.model.revision,
        )

        self.pipeline = pipe
        self.transformer = pipe.transformer
        self.vae = pipe.vae
        self.text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
        self.tokenizers = [pipe.tokenizer, pipe.tokenizer_2]

        # Freeze everything except what we'll LoRA
        self.vae.requires_grad_(False)
        for te in self.text_encoders:
            if te is not None:
                te.requires_grad_(False)
        self.transformer.requires_grad_(False)

        # Move VAE and text encoders to device
        vae_dtype = _get_dtype(self.config.model.vae_dtype)
        te_dtype = _get_dtype(self.config.model.text_encoder_dtype)
        self.vae.to(self.device, dtype=vae_dtype)
        for te in self.text_encoders:
            if te is not None:
                te.to(self.device, dtype=te_dtype)

        if self.config.training.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

    def _load_qwen_model(self):
        """Load a Qwen VL model for training."""
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        model_path = self.config.model.base_model_path
        dtype = _get_dtype(self.config.model.dtype)

        print(f"[DIGIT] Loading Qwen model: {model_path}")
        self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.qwen_processor = AutoProcessor.from_pretrained(model_path)

        # Freeze base model
        self.qwen_model.requires_grad_(False)

        if self.config.training.gradient_checkpointing:
            self.qwen_model.gradient_checkpointing_enable()

    def _inject_lora(self):
        """Inject LoRA adapters using PEFT."""
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            target_modules=self.config.lora.target_modules,
            use_dora=self.config.lora.use_dora,
            use_rslora=self.config.lora.use_rslora,
        )

        model_name = self.config.model.name.lower()
        if "qwen" in model_name:
            self.qwen_model = get_peft_model(self.qwen_model, lora_config)
            self.qwen_model.print_trainable_parameters()
        else:
            # Flux — apply LoRA to transformer
            self.transformer = get_peft_model(self.transformer, lora_config)
            self.transformer.print_trainable_parameters()
            self.transformer.to(self.device)

    def _encode_prompt_flux(self, prompt: str):
        """Encode a text prompt for Flux using both text encoders."""
        # Tokenize with both tokenizers
        tok1_out = self.tokenizers[0](
            prompt, padding="max_length",
            max_length=self.tokenizers[0].model_max_length,
            truncation=True, return_tensors="pt",
        ).to(self.device)
        tok2_out = self.tokenizers[1](
            prompt, padding="max_length",
            max_length=512,
            truncation=True, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            te_dtype = _get_dtype(self.config.model.text_encoder_dtype)
            # CLIP text encoder
            prompt_embeds_1 = self.text_encoders[0](
                tok1_out.input_ids
            )[0]
            # T5 text encoder
            prompt_embeds_2 = self.text_encoders[1](
                tok2_out.input_ids
            )[0]

        return prompt_embeds_1, prompt_embeds_2

    def _training_step_flux(self, batch: dict) -> torch.Tensor:
        """Single Flux training step: encode, noise, predict, loss."""
        pixel_values = batch["pixel_values"].to(self.device)
        captions = batch["captions"]

        dtype = _get_dtype(self.config.model.dtype)
        vae_dtype = _get_dtype(self.config.model.vae_dtype)

        # Encode images to latent space
        with torch.no_grad():
            latents = self.vae.encode(
                pixel_values.to(dtype=vae_dtype)
            ).latent_dist.sample()
            latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            latents = latents.to(dtype=dtype)

        # Encode text
        prompt_embeds_list = []
        pooled_prompt_embeds_list = []
        for caption in captions:
            pe1, pe2 = self._encode_prompt_flux(caption)
            prompt_embeds_list.append(pe2)
            pooled_prompt_embeds_list.append(pe1)

        prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
        pooled_prompt_embeds = torch.cat(pooled_prompt_embeds_list, dim=0)

        # Flow matching: sample timestep uniformly
        bsz = latents.shape[0]
        # Uniform timestep in [0, 1]
        t = torch.rand(bsz, device=self.device, dtype=dtype)

        # Sample noise
        noise = torch.randn_like(latents)

        # Noise offset
        if self.config.training.noise_offset > 0:
            noise += self.config.training.noise_offset * torch.randn(
                bsz, latents.shape[1], 1, 1, device=self.device, dtype=dtype
            )

        # Create noisy latents via flow matching interpolation
        # x_t = (1 - t) * x_0 + t * noise
        t_expand = t.view(bsz, 1, 1, 1)
        noisy_latents = (1 - t_expand) * latents + t_expand * noise

        # Pack latents for Flux transformer
        # Flux expects packed latents format
        packed_noisy = self.pipeline._pack_latents(
            noisy_latents,
            batch_size=bsz,
            num_channels_latents=noisy_latents.shape[1],
            height=noisy_latents.shape[2],
            width=noisy_latents.shape[3],
        )

        # Create timestep embedding
        # Flux uses 1000-scale timesteps
        timesteps = (t * 1000).long()

        # Create image IDs for positional encoding
        latent_image_ids = self.pipeline._prepare_latent_image_ids(
            bsz,
            noisy_latents.shape[2] // 2,
            noisy_latents.shape[3] // 2,
            self.device,
            dtype,
        )

        # Get text IDs
        text_ids = torch.zeros(
            bsz, prompt_embeds.shape[1], 3,
            device=self.device, dtype=dtype,
        )

        # Forward pass through transformer
        model_pred = self.transformer(
            hidden_states=packed_noisy,
            timestep=timesteps / 1000.0,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds.squeeze(1),
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

        # Unpack prediction
        model_pred = self.pipeline._unpack_latents(
            model_pred,
            noisy_latents.shape[2],
            noisy_latents.shape[3],
            self.pipeline.vae_scale_factor,
        )

        # Flow matching target: velocity = noise - x_0
        target = noise - latents

        # MSE loss
        loss = F.mse_loss(model_pred, target, reduction="mean")

        # Min-SNR weighting
        if self.config.training.min_snr_gamma > 0:
            snr = (1 - t_expand) / t_expand
            snr = snr.clamp(min=1e-5)
            gamma = self.config.training.min_snr_gamma
            weight = torch.minimum(snr, torch.full_like(snr, gamma)) / snr
            loss = (weight * F.mse_loss(model_pred, target, reduction="none")).mean()

        return loss

    def _training_step_qwen(self, batch: dict) -> torch.Tensor:
        """Single Qwen VL training step."""
        # For Qwen VL, we do supervised fine-tuning on image-caption pairs
        pixel_values = batch["pixel_values"].to(self.device)
        captions = batch["captions"]

        # Build conversation format for Qwen
        messages_batch = []
        for i, caption in enumerate(captions):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe this image."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": caption},
                    ],
                },
            ]
            messages_batch.append(messages)

        # Process with Qwen processor
        texts = [
            self.qwen_processor.apply_chat_template(msgs, tokenize=False)
            for msgs in messages_batch
        ]

        inputs = self.qwen_processor(
            text=texts,
            images=[pv for pv in pixel_values],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Forward pass with labels for loss computation
        inputs["labels"] = inputs["input_ids"].clone()
        outputs = self.qwen_model(**inputs)

        return outputs.loss

    def setup(self):
        """Load model, inject LoRA, create optimizer and scheduler."""
        model_name = self.config.model.name.lower()

        # Load model
        if "qwen" in model_name:
            self._load_qwen_model()
        else:
            self._load_flux_model()

        # Inject LoRA
        self._inject_lora()

        # Get trainable parameters
        if "qwen" in model_name:
            trainable_params = [p for p in self.qwen_model.parameters() if p.requires_grad]
        else:
            trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]

        # Create optimizer
        self.optimizer = _get_optimizer(trainable_params, self.config)

        # Create dataset
        self.dataset, self.dataloader = prepare_dataset(self.config)
        print(f"[DIGIT] Dataset: {len(self.dataset)} images")

        # Calculate total steps
        if self.config.training.max_train_epochs:
            steps_per_epoch = math.ceil(
                len(self.dataset) / self.config.training.batch_size
            )
            total_steps = steps_per_epoch * self.config.training.max_train_epochs
        else:
            total_steps = self.config.training.max_train_steps

        effective_steps = total_steps // self.config.training.gradient_accumulation_steps
        self.total_steps = total_steps

        # Create scheduler
        self.lr_scheduler = _get_scheduler(self.optimizer, self.config, effective_steps)

        # Create logger
        self.logger = TrainingLogger(
            log_dir=self.config.logging.log_dir,
            use_wandb=self.config.logging.use_wandb,
            use_tensorboard=self.config.logging.use_tensorboard,
            wandb_project=self.config.logging.wandb_project,
            wandb_entity=self.config.logging.wandb_entity,
            run_name=self.config.name,
            config=self.config.to_dict(),
        )

        # Create sampler for sample generation
        if self.config.sampling.enabled and "qwen" not in model_name:
            sample_dir = os.path.join(self.config.output.output_dir, "samples")
            self.sampler = TrainingSampler(
                pipeline=self.pipeline,
                prompts=self.config.sampling.prompts,
                output_dir=sample_dir,
                num_inference_steps=self.config.sampling.num_inference_steps,
                guidance_scale=self.config.sampling.guidance_scale,
                width=self.config.sampling.width,
                height=self.config.sampling.height,
                seed=self.config.sampling.seed,
                logger=self.logger,
            )

        # Create run in DB
        if self.db:
            self.run_id = self.db.create_run(
                name=self.config.name,
                config=self.config.to_dict(),
                total_steps=self.total_steps,
            )

    def train(self, progress_callback=None) -> dict:
        """Run the training loop.

        Args:
            progress_callback: Optional callback(step, total_steps, loss, lr).

        Returns:
            Dict with training results.
        """
        model_name = self.config.model.name.lower()
        is_qwen = "qwen" in model_name
        cfg = self.config.training

        # Set seed
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        # Mixed precision context
        if cfg.mixed_precision == "bf16":
            amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        elif cfg.mixed_precision == "fp16":
            amp_ctx = torch.autocast("cuda", dtype=torch.float16)
        else:
            amp_ctx = nullcontext()

        use_scaler = cfg.mixed_precision == "fp16"
        scaler = GradScaler() if use_scaler else None

        # Training state
        global_step = 0
        total_loss = 0.0
        best_loss = float("inf")
        start_time = time.time()

        if self.db and self.run_id:
            self.db.update_run(self.run_id, status="running", started_at=start_time)

        if is_qwen:
            self.qwen_model.train()
        else:
            self.transformer.train()

        print(f"[DIGIT] Starting training: {self.total_steps} steps")
        print(f"[DIGIT] Batch size: {cfg.batch_size}, "
              f"Grad accum: {cfg.gradient_accumulation_steps}, "
              f"LR: {self.config.optimizer.learning_rate}")

        # Generate initial samples
        if self.sampler and not is_qwen:
            self.sampler.generate_samples(0)

        self.optimizer.zero_grad()

        epoch = 0
        while global_step < self.total_steps:
            epoch += 1

            for batch in self.dataloader:
                if self._stop_requested:
                    print("[DIGIT] Stop requested, saving checkpoint...")
                    break

                with amp_ctx:
                    if is_qwen:
                        loss = self._training_step_qwen(batch)
                    else:
                        loss = self._training_step_flux(batch)

                    loss = loss / cfg.gradient_accumulation_steps

                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient accumulation step
                if (global_step + 1) % cfg.gradient_accumulation_steps == 0:
                    if cfg.max_grad_norm > 0:
                        if use_scaler:
                            scaler.unscale_(self.optimizer)
                        if is_qwen:
                            torch.nn.utils.clip_grad_norm_(
                                self.qwen_model.parameters(), cfg.max_grad_norm
                            )
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.transformer.parameters(), cfg.max_grad_norm
                            )

                    if use_scaler:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                global_step += 1
                step_loss = loss.item() * cfg.gradient_accumulation_steps
                total_loss += step_loss

                # Logging
                if global_step % self.config.logging.log_every_n_steps == 0:
                    avg_loss = total_loss / global_step
                    lr = self.lr_scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    steps_per_sec = global_step / elapsed

                    metrics = {
                        "train/loss": step_loss,
                        "train/avg_loss": avg_loss,
                        "train/lr": lr,
                        "train/epoch": epoch,
                        "train/steps_per_sec": steps_per_sec,
                    }
                    self.logger.log_scalars(metrics, global_step)

                    if progress_callback:
                        progress_callback(global_step, self.total_steps, step_loss, lr)

                    if self.db and self.run_id:
                        self.db.update_run(
                            self.run_id,
                            current_step=global_step,
                            loss=step_loss,
                        )

                # Sample generation
                if (self.sampler and not is_qwen and
                        global_step % self.config.sampling.every_n_steps == 0):
                    self.sampler.generate_samples(global_step)

                # Checkpoint saving
                if (self.config.output.save_every_n_steps > 0 and
                        global_step % self.config.output.save_every_n_steps == 0):
                    self._save_checkpoint(global_step)

                if global_step >= self.total_steps:
                    break

            if self._stop_requested:
                break

        # Save final
        if self.config.output.save_final:
            self._save_lora(global_step)

        # Final samples
        if self.sampler and not is_qwen:
            self.sampler.generate_samples(global_step)

        elapsed = time.time() - start_time
        avg_loss = total_loss / max(global_step, 1)

        if self.db and self.run_id:
            self.db.update_run(
                self.run_id,
                status="completed" if not self._stop_requested else "stopped",
                completed_at=time.time(),
                current_step=global_step,
                loss=avg_loss,
                output_path=self.config.output.output_dir,
            )

        self.logger.close()

        result = {
            "steps": global_step,
            "epochs": epoch,
            "avg_loss": avg_loss,
            "elapsed_seconds": elapsed,
            "output_dir": self.config.output.output_dir,
        }

        print(f"[DIGIT] Training complete: {global_step} steps, "
              f"avg loss: {avg_loss:.6f}, time: {elapsed:.1f}s")

        return result

    def _save_checkpoint(self, step: int):
        """Save a training checkpoint."""
        checkpoint_dir = os.path.join(
            self.config.output.output_dir, "checkpoints", f"step_{step:06d}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_name = self.config.model.name.lower()
        if "qwen" in model_name:
            self.qwen_model.save_pretrained(checkpoint_dir)
        else:
            self.transformer.save_pretrained(checkpoint_dir)

        print(f"[DIGIT] Checkpoint saved: {checkpoint_dir}")

    def _save_lora(self, step: int):
        """Save the final LoRA weights."""
        output_dir = self.config.output.output_dir
        os.makedirs(output_dir, exist_ok=True)

        lora_name = self.config.output.lora_name
        lora_dir = os.path.join(output_dir, lora_name)
        os.makedirs(lora_dir, exist_ok=True)

        model_name = self.config.model.name.lower()
        if "qwen" in model_name:
            self.qwen_model.save_pretrained(lora_dir)
        else:
            self.transformer.save_pretrained(lora_dir)

        # Also save config
        from .config import save_config
        config_path = os.path.join(lora_dir, "digit_config.yaml")
        save_config(self.config, config_path)

        # Save metadata
        import json
        metadata = {
            "model": self.config.model.name,
            "base_model": self.config.model.base_model_path,
            "lora_rank": self.config.lora.rank,
            "lora_alpha": self.config.lora.alpha,
            "trigger_word": self.config.trigger.trigger_word,
            "trigger_class": self.config.trigger.trigger_class,
            "trigger_phrase": self.config.trigger.trigger_phrase,
            "training_steps": step,
            "resolution": self.config.dataset.resolution,
            "dataset_size": len(self.dataset),
        }
        with open(os.path.join(lora_dir, "digit_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[DIGIT] LoRA saved: {lora_dir}")

    def cleanup(self):
        """Free GPU memory."""
        attrs = [
            "pipeline", "transformer", "vae", "text_encoders",
            "qwen_model", "qwen_processor",
        ]
        for attr in attrs:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                obj = getattr(self, attr)
                if isinstance(obj, list):
                    for item in obj:
                        if item is not None:
                            del item
                else:
                    del obj
                setattr(self, attr, None)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

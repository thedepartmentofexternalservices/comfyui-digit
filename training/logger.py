"""Training logger with Wandb and Tensorboard support."""

import os
import time
from typing import Optional

import numpy as np


class TrainingLogger:
    """Unified logging to Wandb and/or Tensorboard."""

    def __init__(
        self,
        log_dir: str = "./logs",
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        wandb_project: str = "digit-lora",
        wandb_entity: str = None,
        run_name: str = None,
        config: dict = None,
    ):
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.tb_writer = None
        self.wandb_run = None

        os.makedirs(log_dir, exist_ok=True)

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(log_dir, "tensorboard")
                self.tb_writer = SummaryWriter(log_dir=tb_dir)
            except ImportError:
                print("[DIGIT] tensorboard not installed, disabling TB logging")
                self.use_tensorboard = False

        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=run_name,
                    config=config,
                    dir=log_dir,
                )
            except ImportError:
                print("[DIGIT] wandb not installed, disabling wandb logging")
                self.use_wandb = False

    def log_scalar(self, tag: str, value: float, step: int):
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_scalar(tag, value, step)
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.log({tag: value}, step=step)

    def log_scalars(self, metrics: dict, step: int):
        if self.use_tensorboard and self.tb_writer:
            for tag, value in metrics.items():
                self.tb_writer.add_scalar(tag, value, step)
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.log(metrics, step=step)

    def log_image(self, tag: str, image, step: int):
        """Log an image (PIL Image or numpy array)."""
        if self.use_tensorboard and self.tb_writer:
            if hasattr(image, "convert"):
                # PIL Image -> numpy
                img_array = np.array(image)
            else:
                img_array = image
            # Tensorboard expects CHW
            if img_array.ndim == 3 and img_array.shape[2] in (3, 4):
                img_array = img_array.transpose(2, 0, 1)
            self.tb_writer.add_image(tag, img_array, step)

        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.log({tag: wandb.Image(image)}, step=step)

    def log_images(self, tag: str, images: list, step: int):
        for i, img in enumerate(images):
            self.log_image(f"{tag}/{i}", img, step)

    def log_text(self, tag: str, text: str, step: int):
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_text(tag, text, step)
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.log({tag: wandb.Html(f"<pre>{text}</pre>")}, step=step)

    def flush(self):
        if self.tb_writer:
            self.tb_writer.flush()

    def close(self):
        if self.tb_writer:
            self.tb_writer.close()
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.finish()

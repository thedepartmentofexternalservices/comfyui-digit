"""Sample image generation during training."""

import os
import torch
from typing import Optional
from PIL import Image


class TrainingSampler:
    """Generate sample images during LoRA training to monitor progress."""

    def __init__(
        self,
        pipeline,
        prompts: list,
        output_dir: str,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        width: int = 1024,
        height: int = 1024,
        seed: int = 42,
        logger=None,
    ):
        self.pipeline = pipeline
        self.prompts = prompts
        self.output_dir = output_dir
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.width = width
        self.height = height
        self.seed = seed
        self.logger = logger

        os.makedirs(output_dir, exist_ok=True)

    @torch.no_grad()
    def generate_samples(self, step: int) -> list:
        """Generate sample images for all prompts at the given training step."""
        self.pipeline.eval()
        images = []

        generator = torch.Generator(device=self.pipeline.device)
        generator.manual_seed(self.seed)

        step_dir = os.path.join(self.output_dir, f"step_{step:06d}")
        os.makedirs(step_dir, exist_ok=True)

        for i, prompt in enumerate(self.prompts):
            try:
                result = self.pipeline(
                    prompt=prompt,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    width=self.width,
                    height=self.height,
                    generator=generator,
                )
                image = result.images[0]
                images.append(image)

                # Save to disk
                filename = f"sample_{i:02d}.png"
                image.save(os.path.join(step_dir, filename))

                # Log
                if self.logger:
                    self.logger.log_image(
                        f"samples/prompt_{i}", image, step
                    )
                    self.logger.log_text(
                        f"samples/prompt_{i}/text", prompt, step
                    )

            except Exception as e:
                print(f"[DIGIT] Sample generation failed for prompt {i}: {e}")

        self.pipeline.train()

        # Log all samples as a grid if we have them
        if images and self.logger:
            self.logger.log_images("samples/grid", images, step)

        return images

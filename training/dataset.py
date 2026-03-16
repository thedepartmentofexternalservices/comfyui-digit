"""Training dataset with aspect ratio bucketing for Flux/Qwen LoRA training."""

import math
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def find_image_caption_pairs(dataset_path: str, caption_ext: str = ".txt") -> list:
    """Find all image files with matching caption files."""
    pairs = []
    dataset_path = Path(dataset_path)

    for img_path in sorted(dataset_path.rglob("*")):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        caption_path = img_path.with_suffix(caption_ext)
        if caption_path.exists():
            pairs.append((str(img_path), str(caption_path)))
        else:
            # Allow images without captions (use filename as caption)
            pairs.append((str(img_path), None))

    return pairs


def compute_buckets(
    min_res: int = 512,
    max_res: int = 2048,
    step: int = 64,
    target_area: int = 1024 * 1024,
) -> list:
    """Compute aspect ratio buckets for variable-size training."""
    buckets = []
    for w in range(min_res, max_res + 1, step):
        for h in range(min_res, max_res + 1, step):
            area = w * h
            # Allow buckets within 20% of target area
            if abs(area - target_area) / target_area <= 0.2:
                buckets.append((w, h))
    return buckets


def find_nearest_bucket(width: int, height: int, buckets: list) -> tuple:
    """Find the bucket with the closest aspect ratio."""
    aspect = width / height
    best = None
    best_diff = float("inf")
    for bw, bh in buckets:
        bucket_aspect = bw / bh
        diff = abs(aspect - bucket_aspect)
        if diff < best_diff:
            best_diff = diff
            best = (bw, bh)
    return best


class DigitDataset(Dataset):
    """Training dataset that loads image-caption pairs with bucketing support."""

    def __init__(
        self,
        dataset_path: str,
        resolution: int = 1024,
        caption_ext: str = ".txt",
        center_crop: bool = True,
        random_flip: bool = False,
        use_bucketing: bool = True,
        bucket_step: int = 64,
        min_bucket_resolution: int = 512,
        max_bucket_resolution: int = 2048,
        trigger_word: str = "",
        trigger_class: str = "",
        trigger_phrase: str = "",
    ):
        self.dataset_path = dataset_path
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.trigger_word = trigger_word
        self.trigger_class = trigger_class
        # Auto-build phrase: "ohwx person" or just "ohwx"
        self.trigger_phrase = trigger_phrase
        if not self.trigger_phrase and self.trigger_word:
            self.trigger_phrase = f"{self.trigger_word} {self.trigger_class}".strip() if self.trigger_class else self.trigger_word

        # Find image-caption pairs
        self.pairs = find_image_caption_pairs(dataset_path, caption_ext)
        if not self.pairs:
            raise ValueError(f"No images found in {dataset_path}")

        # Setup bucketing
        self.use_bucketing = use_bucketing
        if use_bucketing:
            target_area = resolution * resolution
            self.buckets = compute_buckets(
                min_bucket_resolution, max_bucket_resolution,
                bucket_step, target_area
            )
            self._assign_buckets()
        else:
            self.buckets = [(resolution, resolution)]
            self.bucket_assignments = {0: list(range(len(self.pairs)))}

    def _assign_buckets(self):
        """Assign each image to its nearest bucket."""
        self.bucket_assignments = {}
        self.image_buckets = []

        for idx, (img_path, _) in enumerate(self.pairs):
            img = Image.open(img_path)
            w, h = img.size
            img.close()

            bucket = find_nearest_bucket(w, h, self.buckets)
            bucket_idx = self.buckets.index(bucket)
            self.image_buckets.append(bucket_idx)

            if bucket_idx not in self.bucket_assignments:
                self.bucket_assignments[bucket_idx] = []
            self.bucket_assignments[bucket_idx].append(idx)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, caption_path = self.pairs[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Get target size
        if self.use_bucketing and hasattr(self, "image_buckets"):
            bucket_idx = self.image_buckets[idx]
            target_w, target_h = self.buckets[bucket_idx]
        else:
            target_w = target_h = self.resolution

        # Resize and crop
        image = self._resize_and_crop(image, target_w, target_h)

        # Random horizontal flip
        if self.random_flip and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Convert to tensor (0-1 range, CHW)
        image_tensor = torch.from_numpy(
            np.array(image).astype(np.float32) / 255.0
        ).permute(2, 0, 1)

        # Load caption
        caption = self._load_caption(img_path, caption_path)

        return {
            "pixel_values": image_tensor,
            "caption": caption,
            "image_path": img_path,
        }

    def _resize_and_crop(self, image: Image.Image, target_w: int, target_h: int) -> Image.Image:
        """Resize image to fill target, then crop."""
        w, h = image.size
        scale = max(target_w / w, target_h / h)
        new_w = int(w * scale + 0.5)
        new_h = int(h * scale + 0.5)
        image = image.resize((new_w, new_h), Image.LANCZOS)

        if self.center_crop:
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
        else:
            left = random.randint(0, max(0, new_w - target_w))
            top = random.randint(0, max(0, new_h - target_h))

        return image.crop((left, top, left + target_w, top + target_h))

    def _load_caption(self, img_path: str, caption_path: Optional[str]) -> str:
        """Load caption from file or generate from filename."""
        caption = ""
        if caption_path and os.path.exists(caption_path):
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
        else:
            # Use filename as fallback
            caption = Path(img_path).stem.replace("_", " ").replace("-", " ")

        # Prepend trigger phrase to caption for training
        if self.trigger_phrase:
            caption = f"{self.trigger_phrase}, {caption}"
        elif self.trigger_word:
            caption = f"{self.trigger_word}, {caption}"

        return caption


class BucketSampler(Sampler):
    """Sampler that yields batches from the same bucket for consistent dimensions."""

    def __init__(self, dataset: DigitDataset, batch_size: int = 1, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        # Build batches per bucket
        all_batches = []
        for bucket_idx, indices in self.dataset.bucket_assignments.items():
            if self.shuffle:
                indices = indices.copy()
                random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:
                    all_batches.append(batch)
                elif len(batch) > 0:
                    # Pad last batch by repeating
                    while len(batch) < self.batch_size:
                        batch.append(random.choice(indices))
                    all_batches.append(batch)

        if self.shuffle:
            random.shuffle(all_batches)

        for batch in all_batches:
            yield from batch

    def __len__(self):
        total = sum(
            math.ceil(len(indices) / self.batch_size) * self.batch_size
            for indices in self.dataset.bucket_assignments.values()
        )
        return total


def prepare_dataset(config) -> tuple:
    """Create dataset and dataloader from config.

    Returns (dataset, dataloader).
    """
    dataset = DigitDataset(
        dataset_path=config.dataset.path,
        resolution=config.dataset.resolution,
        caption_ext=config.dataset.caption_ext,
        center_crop=config.dataset.center_crop,
        random_flip=config.dataset.random_flip,
        use_bucketing=config.dataset.use_bucketing,
        bucket_step=config.dataset.bucket_step,
        min_bucket_resolution=config.dataset.min_bucket_resolution,
        max_bucket_resolution=config.dataset.max_bucket_resolution,
        trigger_word=config.trigger.trigger_word,
        trigger_class=config.trigger.trigger_class,
        trigger_phrase=config.trigger.trigger_phrase,
    )

    sampler = BucketSampler(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=_collate_fn,
    )

    return dataset, dataloader


def _collate_fn(batch):
    """Custom collate that handles variable-size images within same bucket."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    captions = [item["caption"] for item in batch]
    image_paths = [item["image_path"] for item in batch]
    return {
        "pixel_values": pixel_values,
        "captions": captions,
        "image_paths": image_paths,
    }

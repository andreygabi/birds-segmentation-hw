from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MEAN = (.485, .456, .406)
STD = (.229, .224, .225)
PAD_RGB = tuple(round(value * 255) for value in MEAN)


@dataclass(frozen=True)
class Pair:
    image: Path
    mask: Path
    key: str


def image_files(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"No JPG/JPEG/PNG images in {root}")
    by_key = {}
    for path in paths:
        key = path.relative_to(root).with_suffix("").as_posix()
        if key in by_key:
            raise ValueError(f"Image stem collision: {by_key[key]} and {path}")
        by_key[key] = path
    return by_key


def find_pairs(images_root, masks_root):
    images = image_files(images_root)
    masks_root = Path(masks_root)
    masks = {}
    for path in sorted(masks_root.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".png":
            key = path.relative_to(masks_root).with_suffix("").as_posix()
            if key in masks:
                raise ValueError(f"Mask stem collision: {key}")
            masks[key] = path
    missing, orphan = set(images) - set(masks), set(masks) - set(images)
    if missing or orphan:
        raise ValueError(f"Missing masks: {sorted(missing)[:5]}; masks without image: {sorted(orphan)[:5]}")
    return [Pair(path, masks[key], key) for key, path in images.items()]


def read_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def read_mask(path):
    with Image.open(path) as mask:
        if mask.mode == "P":
            mask = mask.convert("RGB")
        array = np.asarray(mask)
        if array.ndim == 3:
            array = array[..., 0]
        if array.dtype == np.bool_:
            array = array.astype(np.uint8) * 255
        if array.ndim != 2:
            raise ValueError(f"Expected a two-dimensional mask: {path}")
        return (array == 255).astype(np.uint8)


def letterbox(image, size, mask=None):
    width, height = image.size
    scale = min(size / width, size / height)
    new_width = min(size, max(1, round(width * scale)))
    new_height = min(size, max(1, round(height * scale)))
    left, top = (size - new_width) // 2, (size - new_height) // 2
    canvas = Image.new("RGB", (size, size), PAD_RGB)
    canvas.paste(image.resize((new_width, new_height), Image.Resampling.BILINEAR), (left, top))
    target = None
    if mask is not None:
        if mask.shape != (height, width):
            raise ValueError("Image and mask dimensions differ")
        target = Image.new("L", (size, size), 0)
        target.paste(Image.fromarray(mask * 255).resize((new_width, new_height), Image.Resampling.NEAREST), (left, top))
    metadata = {"width": width, "height": height, "resized_width": new_width, "resized_height": new_height, "left": left, "top": top}
    return canvas, target, metadata


def image_tensor(image):
    return TF.normalize(TF.to_tensor(image), MEAN, STD)


class BirdDataset(Dataset):
    def __init__(self, pairs, size, augmentation="none", cache=False, cache_workers=8):
        self.pairs = list(pairs)
        self.size = size
        self.augmentation = augmentation
        self.jitter = ColorJitter(brightness=.15, contrast=.15, saturation=.1, hue=.02)
        self.cache = None
        if cache:
            with ThreadPoolExecutor(max_workers=cache_workers) as pool:
                self.cache = list(pool.map(self._prepare, range(len(self.pairs))))

    def __len__(self):
        return len(self.pairs)

    def _prepare(self, index):
        pair = self.pairs[index]
        image, mask, metadata = letterbox(read_image(pair.image), self.size, read_mask(pair.mask))
        return np.asarray(image).copy(), np.asarray(mask).copy(), metadata

    def __getitem__(self, index):
        array, target, metadata = self.cache[index] if self.cache is not None else self._prepare(index)
        image, mask = Image.fromarray(array), Image.fromarray(target)
        if self.augmentation != "none" and random.random() < .5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.augmentation == "strong":
            angle = random.uniform(-10, 10)
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=PAD_RGB)
            mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
            image = self.jitter(image)
        pair = self.pairs[index]
        metadata = {**metadata, "key": pair.key, "mask_path": str(pair.mask), "image_path": str(pair.image)}
        target_tensor = torch.from_numpy((np.asarray(mask).copy() == 255).astype(np.float32)).unsqueeze(0)
        return image_tensor(image), target_tensor, metadata


def collate_pairs(batch):
    images, masks, metadata = zip(*batch)
    return torch.stack(images), torch.stack(masks), list(metadata)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)

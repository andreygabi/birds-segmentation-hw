import argparse
from pathlib import Path

from PIL import Image
import torch

from src.data import image_files, image_tensor, letterbox, read_image
from src.inference import Predictor
from src.metrics import binary_mask
from src.utils import choose_device, seed_everything


def image_batches(images_root, size, batch_size, limit=None):
    files = list(image_files(images_root).items())
    if limit:
        files = files[:limit]
    for start in range(0, len(files), batch_size):
        images, metadata = [], []
        for key, path in files[start:start + batch_size]:
            image, _, meta = letterbox(read_image(path), size)
            images.append(image_tensor(image))
            metadata.append({**meta, "key": key})
        yield torch.stack(images), metadata


def predict(checkpoint, images_root, output_dir, device="auto", backend="torch", onnx_path=None, batch_size=16, threads=6):
    seed_everything(42)
    torch.set_num_threads(threads)
    predictor = Predictor(checkpoint, choose_device(device) if backend == "torch" else "cpu", backend, onnx_path, threads)
    count = 0
    for images, metadata in image_batches(images_root, predictor.config["image_size"], batch_size):
        logits = predictor(images)
        for logit, meta in zip(logits, metadata):
            path = Path(output_dir) / (meta["key"] + ".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(binary_mask(logit, meta, predictor.threshold) * 255).save(path)
            count += 1
    print(f"Saved {count} binary PNG masks to {Path(output_dir).resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    parser.add_argument("--onnx-path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    predict(args.checkpoint, args.images_root, args.output_dir, args.device, args.backend, args.onnx_path, args.batch_size, args.threads)

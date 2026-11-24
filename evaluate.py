import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from predict import image_batches
from src.data import find_pairs, read_mask
from src.inference import Predictor
from src.metrics import binary_mask, image_iou
from src.utils import choose_device, seed_everything, write_json


def evaluate(checkpoint, data_root, output, device="auto", backend="torch", onnx_path=None, batch_size=16, masks_root=None, threads=6):
    seed_everything(42)
    torch.set_num_threads(threads)
    pairs = find_pairs(Path(data_root) / "val/images", Path(data_root) / "val/gt")
    targets = {pair.key: pair.mask for pair in pairs}
    rows = []
    predictor = None if masks_root else Predictor(checkpoint, choose_device(device) if backend == "torch" else "cpu", backend, onnx_path, threads)
    if masks_root:
        for pair in pairs:
            path = Path(masks_root) / (pair.key + ".png")
            with Image.open(path) as image:
                array = np.asarray(image)
            if array.ndim != 2 or not np.isin(array, [0, 255]).all():
                raise ValueError(f"Prediction must be a binary single-channel PNG: {path}")
            rows.append({"key": pair.key, "iou": image_iou(array == 255, read_mask(pair.mask))})
    else:
        for images, metadata in image_batches(Path(data_root) / "val/images", predictor.config["image_size"], batch_size):
            logits = predictor(images)
            for logit, meta in zip(logits, metadata):
                rows.append({"key": meta["key"], "iou": image_iou(binary_mask(logit, meta, predictor.threshold), read_mask(targets[meta["key"]]))})
    frame = pd.DataFrame(rows)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output.with_suffix(".csv"), index=False)
    result = {"images": len(rows), "mean_iou": float(frame.iou.mean()), "backend": "saved_png" if masks_root else backend, "threshold": predictor.threshold if predictor else None, "batch_size": batch_size, "threads": threads, "device": str(predictor.device) if predictor else None}
    write_json(output.with_suffix(".json"), result)
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default="results/evaluation.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    parser.add_argument("--onnx-path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--masks-root")
    args = parser.parse_args()
    if not (args.checkpoint or args.masks_root):
        parser.error("Provide --checkpoint or --masks-root")
    evaluate(args.checkpoint, args.data_root, args.output, args.device, args.backend, args.onnx_path, args.batch_size, args.masks_root, args.threads)

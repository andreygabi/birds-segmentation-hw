import argparse
from pathlib import Path
import platform
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from predict import image_batches
from src.inference import load_checkpoint
from src.metrics import binary_mask
from src.utils import seed_everything, write_json


def run(checkpoint_path, onnx_path, images_root, output, limit=64, batch_sizes=(1, 4, 8), repeats=3, threads=6):
    seed_everything(42)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    model, checkpoint = load_checkpoint(checkpoint_path)
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
    rows, errors, changed, pixels = [], [], 0, 0
    keys = []
    with torch.inference_mode():
        for batch_size in batch_sizes:
            batches = list(image_batches(images_root, checkpoint["config"]["image_size"], batch_size, limit))
            arrays = [np.ascontiguousarray(batch.numpy()) for batch, _ in batches]
            if not keys:
                keys = [meta["key"] for _, metadata in batches for meta in metadata]
            for index, ((batch, metadata), array) in enumerate(zip(batches, arrays)):
                torch_logits = model(batch)
                ort_logits = torch.from_numpy(session.run(None, {"images": array})[0])
                errors.append(float((torch_logits - ort_logits).abs().max()))
                for a, b, meta in zip(torch_logits, ort_logits, metadata):
                    first = binary_mask(a, meta, checkpoint.get("threshold", .5))
                    second = binary_mask(b, meta, checkpoint.get("threshold", .5))
                    changed += int((first != second).sum())
                    pixels += first.size
            for backend in ["torch", "onnx"]:
                def forward_all():
                    if backend == "torch":
                        for batch, _ in batches:
                            model(batch)
                    else:
                        for array in arrays:
                            session.run(None, {"images": array})
                for _ in range(2):
                    forward_all()
                times = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    forward_all()
                    times.append(time.perf_counter() - start)
                median = float(np.median(times))
                rows.append({"backend": backend, "batch_size": batch_size, "images": len(keys), "median_seconds": median, "images_per_second": len(keys) / median, "timings_seconds": times})
                print(rows[-1], flush=True)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path.with_suffix(".csv"), index=False)
    write_json(path.with_suffix(".json"), {"device": "CPU", "processor": platform.processor(), "platform": platform.platform(), "torch": torch.__version__, "onnxruntime": ort.__version__, "dtype": "float32", "intra_op_threads": threads, "inter_op_threads": 1, "warmup_passes": 2, "repeats": repeats, "timed": "Forward only; preloaded normalized tensors. No I/O, preprocessing or postprocessing.", "keys": keys, "max_abs_logit_error": max(errors), "changed_mask_pixels": changed, "compared_mask_pixels": pixels, "rows": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--output", default="results/benchmark.json")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    run(args.checkpoint, args.onnx_path, args.images_root, args.output, args.limit, args.batch_sizes, args.repeats, args.threads)

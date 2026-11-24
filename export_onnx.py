import argparse
from pathlib import Path

import torch

from src.inference import load_checkpoint
from src.utils import sha256, write_json


def export(checkpoint_path, output):
    model, checkpoint = load_checkpoint(checkpoint_path)
    size = checkpoint["config"]["image_size"]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(model, torch.zeros(1, 3, size, size), str(output), input_names=["images"], output_names=["logits"], dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    write_json(output.with_suffix(".json"), {"sha256": sha256(output), "opset": 17, "dtype": "float32", "image_size": size, "threshold": checkpoint.get("threshold", .5), "dynamic_batch": True})
    print(output.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="weights/best.onnx")
    args = parser.parse_args()
    export(args.checkpoint, args.output)

from pathlib import Path
import numpy as np
import torch

from .data import read_mask
from .metrics import binary_mask, image_iou, segmentation_loss
from .model import UNet


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = UNet(pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    return model, checkpoint


class Predictor:
    def __init__(self, checkpoint_path, device, backend="torch", onnx_path=None, threads=6):
        self.device = device
        self.backend = backend
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.config = checkpoint["config"]
        self.threshold = checkpoint.get("threshold", .5)
        if backend == "torch":
            self.model, _ = load_checkpoint(checkpoint_path, device)
        elif backend == "onnx":
            import onnxruntime as ort
            if onnx_path is None:
                raise ValueError("--onnx-path is required for ONNX Runtime")
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
        else:
            raise ValueError(backend)

    @torch.inference_mode()
    def __call__(self, images):
        if self.backend == "torch":
            return self.model(images.to(self.device)).float().cpu()
        array = np.ascontiguousarray(images.numpy(), dtype=np.float32)
        return torch.from_numpy(self.session.run(None, {self.input_name: array})[0])


@torch.inference_mode()
def evaluate_torch(model, loader, device, threshold=.5, dice_weight=.5, original_masks=None):
    model.eval()
    losses, scores, rows = 0.0, [], []
    for images, masks, metadata in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        losses += float(segmentation_loss(logits, masks, dice_weight).cpu()) * len(images)
        logits = logits.float().cpu()
        for logit, meta in zip(logits, metadata):
            target = original_masks[meta["key"]] if original_masks is not None else read_mask(meta["mask_path"])
            score = image_iou(binary_mask(logit, meta, threshold), target)
            scores.append(score)
            rows.append({"key": meta["key"], "iou": score, "height": meta["height"], "width": meta["width"]})
    if not scores:
        raise ValueError("Empty validation dataset")
    return {"loss": losses / len(scores), "mean_iou": float(np.mean(scores)), "images": len(scores)}, rows

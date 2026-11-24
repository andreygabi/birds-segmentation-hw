import argparse
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from predict import image_batches
from src.data import find_pairs, read_image, read_mask
from src.inference import load_checkpoint
from src.metrics import image_iou, restore_logits
from src.utils import choose_device, read_config, save_torch, seed_everything, sha256, write_json


def finalize(runs_dir, data_root, output_dir, device_name):
    seed_everything(42)
    runs_dir, data_root, output_dir = Path(runs_dir), Path(data_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not (run_dir / "best.pt").exists():
            continue
        checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        history = pd.read_csv(run_dir / "history.csv")
        experiments.append({"run": run_dir.name, "image_size": config["image_size"], "loss": "BCE" if config["dice_weight"] == 0 else "0.5 BCE + 0.5 Dice", "epochs": int(history.epoch.max()), "best_epoch": checkpoint["epoch"], "val_mean_iou": checkpoint["val_mean_iou"], "threshold": .5, "train_seconds": float(history.epoch_seconds.sum())})
        shutil.copy(run_dir / "history.csv", output_dir / f"history_{run_dir.name}.csv")
    frame = pd.DataFrame(experiments)
    frame.to_csv(output_dir / "experiments.csv", index=False)
    winner = frame.sort_values(["val_mean_iou", "run"], ascending=[False, True]).iloc[0]
    source = runs_dir / winner["run"] / "best.pt"
    device = choose_device(device_name)
    model, checkpoint = load_checkpoint(source, device)
    pairs = find_pairs(data_root / "val/images", data_root / "val/gt")
    targets = {pair.key: read_mask(pair.mask) for pair in pairs}
    thresholds = np.round(np.arange(.30, .701, .05), 2)
    scores = [[] for _ in thresholds]
    wrong_scores, records, raw_logits = [], [], {}
    with torch.inference_mode():
        for images, metadata in image_batches(data_root / "val/images", checkpoint["config"]["image_size"], 16):
            logits = model(images.to(device)).float().cpu()
            for logit, meta in zip(logits, metadata):
                restored = restore_logits(logit, meta)
                probability = restored.sigmoid().numpy()
                raw_logits[meta["key"]] = probability.astype(np.float32)
                target = targets[meta["key"]]
                for column, threshold in enumerate(thresholds):
                    scores[column].append(image_iou(probability >= threshold, target))
                wrong_scores.append(image_iou(restored.numpy() >= .5, target))
                records.append({"key": meta["key"], "height": meta["height"], "width": meta["width"]})
    threshold_rows = [{"threshold": float(value), "val_mean_iou": float(np.mean(scores[index]))} for index, value in enumerate(thresholds)]
    table = pd.DataFrame(threshold_rows)
    table.to_csv(output_dir / "thresholds.csv", index=False)
    selected = table.sort_values(["val_mean_iou", "threshold"], ascending=[False, False]).iloc[0]
    threshold, final_iou = float(selected.threshold), float(selected.val_mean_iou)
    index = list(thresholds).index(threshold)
    for record, score in zip(records, scores[index]):
        record["iou"] = score
        record["gt_foreground_pixels"] = int(targets[record["key"]].sum())
        record["pred_foreground_pixels"] = int((raw_logits[record["key"]] >= threshold).sum())
    pd.DataFrame(records).to_csv(output_dir / "val_per_image.csv", index=False)
    area_frame = pd.DataFrame(records)
    area_frame["gt_fraction"] = area_frame.gt_foreground_pixels / (area_frame.height * area_frame.width)
    area_frame["group"] = pd.cut(area_frame.gt_fraction, [-1, 0, .05, .15, 1], labels=["empty", "(0, 5%]", "(5%, 15%]", "(15%, 100%]"])
    area_frame.groupby("group", observed=True).agg(images=("iou", "size"), mean_iou=("iou", "mean")).to_csv(output_dir / "iou_by_area.csv")
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    checkpoint["threshold"] = threshold
    checkpoint["val_mean_iou_at_05"] = checkpoint["val_mean_iou"]
    checkpoint["val_mean_iou"] = final_iou
    checkpoint["mask_rule"] = "first_channel == 255"
    save_torch(weights_dir / "best.pt", checkpoint)
    wrong_threshold = 1 / (1 + np.exp(-threshold))
    wrong_selected = np.mean([image_iou(raw_logits[key] >= wrong_threshold, target) for key, target in targets.items()])
    write_json(output_dir / "selection.json", {"selected_run": str(winner["run"]), "selected_epoch": checkpoint["epoch"], "threshold": threshold, "val_mean_iou": final_iou, "val_mean_iou_at_05": float(winner.val_mean_iou), "weights_sha256": sha256(weights_dir / "best.pt"), "images": len(records), "ground_truth": "first channel == 255; intermediate values are background", "postprocessing": {"incorrect_raw_logits_threshold_05_iou": float(np.mean(wrong_scores)), "incorrect_raw_logits_selected_threshold_iou": float(wrong_selected), "all_foreground_iou": float(np.mean([target.mean() for target in targets.values()])), "all_background_iou": float(np.mean([not target.any() for target in targets.values()]))}})
    actual = {"training_config": checkpoint["config"], "selected_epoch": checkpoint["epoch"], "threshold": threshold, "mask_rule": "first_channel == 255", "preprocessing": {"resize": "aspect-preserving letterbox", "image_interpolation": "bilinear", "mask_interpolation": "nearest", "normalization": "ImageNet mean/std", "pad_rgb": [124, 116, 104]}, "source_checkpoint": str(source), "parent_training": "dice_256: own train checkpoint" if winner["run"] == "refine_320" else "ImageNet encoder; random decoder"}
    if winner["run"] == "refine_320":
        parent = torch.load(runs_dir / "dice_256/best.pt", map_location="cpu", weights_only=True)
        actual["parent_config"] = parent["config"]
        actual["parent_selected_epoch"] = parent["epoch"]
    (output_dir / "selected_config.yaml").write_text(yaml.safe_dump(actual, allow_unicode=True, sort_keys=False))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for experiment in experiments:
        history = pd.read_csv(output_dir / f"history_{experiment['run']}.csv")
        axes[0].plot(history.epoch, history.train_loss, marker="o", label=experiment["run"])
        axes[1].plot(history.epoch, history.val_mean_iou, marker="o", label=experiment["run"])
    axes[0].set(title="Train loss", xlabel="Epoch")
    axes[1].set(title="Validation mean image IoU, threshold 0.5", xlabel="Epoch")
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)
    ordered = sorted(records, key=lambda record: record["iou"])
    chosen = [ordered[0], ordered[len(ordered) // 5], ordered[len(ordered) // 2], ordered[4 * len(ordered) // 5]]
    fig, axes = plt.subplots(4, 3, figsize=(10, 12))
    image_paths = {pair.key: pair.image for pair in pairs}
    for row, record in enumerate(chosen):
        key = record["key"]
        axes[row, 0].imshow(read_image(image_paths[key]))
        axes[row, 1].imshow(targets[key], cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(raw_logits[key] >= threshold, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(key.split("/")[-1], fontsize=7)
        axes[row, 1].set_title("Ground truth == 255")
        axes[row, 2].set_title(f"Prediction; IoU={record['iou']:.3f}")
        for axis in axes[row]:
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "examples.png", dpi=130)
    plt.close(fig)
    print({"run": winner["run"], "threshold": threshold, "mean_iou": final_iou}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    finalize(args.runs_dir, args.data_root, args.output_dir, args.device)

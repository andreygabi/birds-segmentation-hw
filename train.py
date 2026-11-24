import argparse
from contextlib import nullcontext
from pathlib import Path
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from src.data import BirdDataset, collate_pairs, find_pairs, read_mask, seed_worker
from src.inference import evaluate_torch
from src.metrics import segmentation_loss
from src.model import UNet
from src.utils import choose_device, now, read_config, save_torch, seed_everything, write_json


def run(config_path, data_root, runs_dir, device_name="auto", resume=None, init_from=None):
    config = read_config(config_path)
    seed_everything(config["seed"])
    device = choose_device(device_name)
    run_dir = Path(runs_dir) / config["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    train_pairs = find_pairs(Path(data_root) / "train/images", Path(data_root) / "train/gt")
    val_pairs = find_pairs(Path(data_root) / "val/images", Path(data_root) / "val/gt")
    write_json(run_dir / "split_keys.json", {"train": [p.key for p in train_pairs], "val": [p.key for p in val_pairs]})
    if set(p.key for p in train_pairs) & set(p.key for p in val_pairs):
        raise ValueError("Same relative image keys occur in train and validation")
    print(f"Preparing {len(train_pairs)} train and {len(val_pairs)} val pairs; size={config['image_size']}, device={device}", flush=True)
    train_data = BirdDataset(train_pairs, config["image_size"], config["augmentation"], config.get("cache_data", False))
    val_data = BirdDataset(val_pairs, config["image_size"], "none", config.get("cache_data", False))
    original_masks = {p.key: read_mask(p.mask) for p in val_pairs}
    loader_args = {"batch_size": config["batch_size"], "num_workers": config.get("num_workers", 0), "collate_fn": collate_pairs, "worker_init_fn": seed_worker, "pin_memory": device.type == "cuda"}
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    model = UNet(pretrained=not (resume or init_from)).to(device)
    if init_from:
        parent = torch.load(init_from, map_location="cpu", weights_only=True)
        model.load_state_dict(parent["model_state"], strict=True)
    encoder_ids = {id(p) for module in model.encoder_modules() for p in module.parameters()}
    groups = [
        {"params": [p for p in model.parameters() if id(p) in encoder_ids], "lr": config["encoder_lr"]},
        {"params": [p for p in model.parameters() if id(p) not in encoder_ids], "lr": config["decoder_lr"]},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["encoder_lr"] * .1)
    amp = bool(config.get("amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    first_epoch, best_score, best_epoch = 1, -1.0, None
    history = []
    if resume:
        last = torch.load(resume, map_location="cpu", weights_only=True)
        if last["config"] != config:
            raise ValueError("Resume configuration differs from checkpoint")
        model.load_state_dict(last["model_state"], strict=True)
        optimizer.load_state_dict(last["optimizer_state"])
        scheduler.load_state_dict(last["scheduler_state"])
        scaler.load_state_dict(last["scaler_state"])
        first_epoch = last["epoch"] + 1
        best_score, best_epoch = last["best_score"], last["best_epoch"]
        history = last["history"]
    write_json(run_dir / "metadata.json", {"started_at": now(), "device": str(device), "train_rows": len(train_data), "val_rows": len(val_data), "init_from": str(init_from) if init_from else None, "resume": str(resume) if resume else None, "amp_used": amp, "encoder_pretraining": "ImageNet1K_V1" if not init_from else "own train checkpoint", "decoder": "custom random initialization in initial run"})
    started = time.perf_counter()
    for epoch in range(first_epoch, config["epochs"] + 1):
        seed_everything(config["seed"] + epoch)
        train_loader = DataLoader(train_data, shuffle=True, generator=torch.Generator().manual_seed(config["seed"] + epoch), **loader_args)
        model.train()
        model.set_encoder_trainable(epoch > config.get("freeze_encoder_epochs", 0))
        total_loss, seen = 0.0, 0
        epoch_start = time.perf_counter()
        for step, (images, masks, _) in enumerate(train_loader, 1):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext():
                logits = model(images)
                loss = segmentation_loss(logits, masks, config["dice_weight"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * len(images)
            seen += len(images)
            if step % 100 == 0:
                print(f"{config['name']} epoch={epoch}/{config['epochs']} batch={step}/{len(train_loader)} loss={total_loss / seen:.4f}", flush=True)
        score, per_image = evaluate_torch(model, val_loader, device, threshold=.5, dice_weight=config["dice_weight"], original_masks=original_masks)
        row = {"epoch": epoch, "train_loss": total_loss / seen, "val_loss": score["loss"], "val_mean_iou": score["mean_iou"], "threshold": .5, "epoch_seconds": time.perf_counter() - epoch_start, "encoder_lr": optimizer.param_groups[0]["lr"], "decoder_lr": optimizer.param_groups[1]["lr"]}
        history.append(row)
        scheduler.step()
        if score["mean_iou"] > best_score:
            best_score, best_epoch = score["mean_iou"], epoch
            save_torch(run_dir / "best.pt", {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "config": config, "epoch": epoch, "val_mean_iou": best_score, "threshold": .5})
            pd.DataFrame(per_image).to_csv(run_dir / "best_val_iou.csv", index=False)
        temporary = run_dir / "history.csv.tmp"
        pd.DataFrame(history).to_csv(temporary, index=False)
        temporary.replace(run_dir / "history.csv")
        save_torch(run_dir / "last.pt", {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(), "config": config, "epoch": epoch, "best_score": best_score, "best_epoch": best_epoch, "history": history})
        write_json(run_dir / "summary.json", {"best_epoch": best_epoch, "best_val_mean_iou": best_score, "completed_epoch": epoch, "train_seconds": time.perf_counter() - started, "images_train": len(train_data), "images_val": len(val_data), "threshold": .5, "device": str(device)})
        print("Completed:", row, "best_iou=", best_score, flush=True)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume")
    parser.add_argument("--init-from")
    args = parser.parse_args()
    run(args.config, args.data_root, args.runs_dir, args.device, args.resume, args.init_from)

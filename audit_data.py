import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from src.data import find_pairs, read_mask
from src.utils import sha256, write_json


def inspect_pair(pair):
    with Image.open(pair.image) as image:
        mode, size = image.mode, image.size
    with Image.open(pair.mask) as mask:
        mask_mode = mask.mode
        raw = np.asarray(mask.convert("RGB") if mask.mode == "P" else mask)
        if raw.ndim == 3:
            raw = raw[..., 0]
        intermediate_pixels = int(((raw != 0) & (raw != 255)).sum())
    target = read_mask(pair.mask)
    if target.shape != (size[1], size[0]):
        raise ValueError(f"Dimensions differ: {pair.key}")
    return {"image_mode": mode, "mask_mode": mask_mode, "width": size[0], "height": size[1], "foreground_fraction": float(target.mean()), "intermediate_pixels": intermediate_pixels, "image_sha256": sha256(pair.image)}


def run(data_root, output):
    results, hashes, keys = {}, {}, {}
    for split in ["train", "val"]:
        pairs = find_pairs(Path(data_root) / split / "images", Path(data_root) / split / "gt")
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = list(pool.map(inspect_pair, pairs))
        hashes[split] = {r["image_sha256"] for r in rows}
        keys[split] = {p.key for p in pairs}
        results[split] = {"pairs": len(pairs), "bird_folders": len({p.key.split('/')[0] for p in pairs}),
                          "image_modes": dict(Counter(r["image_mode"] for r in rows)), "mask_modes": dict(Counter(r["mask_mode"] for r in rows)),
                          "empty_masks": sum(r["foreground_fraction"] == 0 for r in rows),
                          "masks_with_intermediate_values": sum(r["intermediate_pixels"] > 0 for r in rows),
                          "intermediate_pixels": sum(r["intermediate_pixels"] for r in rows),
                          "mean_foreground_fraction": float(np.mean([r["foreground_fraction"] for r in rows])),
                          "width_range": [min(r["width"] for r in rows), max(r["width"] for r in rows)],
                          "height_range": [min(r["height"] for r in rows), max(r["height"] for r in rows)]}
    results["cross_split"] = {"common_relative_keys": len(keys["train"] & keys["val"]), "common_image_hashes": len(hashes["train"] & hashes["val"])}
    write_json(output, results)
    print(results, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="results/data_audit.json")
    args = parser.parse_args()
    run(args.data_root, args.output)

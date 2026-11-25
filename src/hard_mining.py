"""Mine hard training examples for targeted oversampling.

Scores every training image with the current champion. Samples whose true-class
confidence is low (but not absurdly low — those are usually label noise) get a
higher sampling weight. Weights are saved as {relative_path: weight}.

Usage:
  python -m src.hard_mining --checkpoint checkpoints/best_action_model.pth \
      --out data/hard_weights.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from .config import DATA_DIR, DEVICE, NUM_WORKERS, PROJECT_ROOT
from .dataset import eval_transforms, find_image_root
from .infer import checkpoint_img_size, load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_action_model.pth")
    p.add_argument("--out", default="data/hard_weights.json")
    p.add_argument("--boost", type=float, default=2.0,
                   help="extra weight for the hardest ~10%% of samples")
    args = p.parse_args()

    model, classes = load_model(args.checkpoint)
    size = checkpoint_img_size(args.checkpoint)
    root = find_image_root(Path(DATA_DIR))
    train_dir = root / "train" if (root / "train").exists() else root
    ds = datasets.ImageFolder(str(train_dir), transform=eval_transforms(size))
    loader = DataLoader(ds, batch_size=96, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    diffs = []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="mining"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            probs = model(x).softmax(1)
            true_p = probs.gather(1, y[:, None]).squeeze(1)
            diffs.append((1 - true_p).float().cpu().numpy())
    diffs = np.concatenate(diffs)

    p50, p90 = np.percentile(diffs, [50, 90])
    scale = max(p90 - p50, 1e-6)
    weights = 1 + args.boost * np.clip((diffs - p50) / scale, 0, 1)

    mapping = {}
    for (path, _label), w in zip(ds.samples, weights):
        try:
            key = Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            key = str(path)
        mapping[key] = round(float(w), 4)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mapping), encoding="utf-8")
    print(f"Mined {len(mapping)} samples -> {out}")
    print(f"difficulty median={p50:.3f} p90={p90:.3f} "
          f"weight min={weights.min():.2f} max={weights.max():.2f} mean={weights.mean():.2f}")


if __name__ == "__main__":
    main()

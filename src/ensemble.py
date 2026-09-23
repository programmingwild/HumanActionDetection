"""Two-model ensemble (EfficientNet-B0 @224 + EfficientNet-B3 @300).

Averages softmax probabilities; each model runs at its own training resolution.

Usage:
  python -m src.ensemble                                   # test accuracy
  python -m src.ensemble --tta                            # + flip averaging
  python -m src.ensemble --image photo.jpg                # single prediction
"""
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from .config import BATCH_SIZE, DATA_DIR, DEVICE, NUM_WORKERS
from .dataset import eval_transforms, find_image_root
from .infer import checkpoint_img_size, load_model
from .labels import pretty
from pathlib import Path
from PIL import Image


def load_ensemble(paths):
    """Return [(model, classes, tfm)] — asserts identical class lists."""
    members = []
    for p in paths:
        model, classes = load_model(p)
        tfm = eval_transforms(checkpoint_img_size(p))
        members.append((model, classes, tfm))
    base = members[0][1]
    for _, classes, _ in members[1:]:
        if classes != base:
            raise ValueError("Ensemble members have different classes")
    print(f"Ensemble: {len(members)} models, {len(base)} classes")
    return members, base


@torch.no_grad()
def ensemble_probs(members, img: Image.Image, tta=False):
    """Average member softmaxes for one PIL image -> (probs tensor on CPU)."""
    rgb = img.convert("RGB")
    acc = None
    for model, _, tfm in members:
        views = [rgb, rgb.transpose(Image.FLIP_LEFT_RIGHT)] if tta else [rgb]
        for v in views:
            x = tfm(v).unsqueeze(0).to(DEVICE)
            p = model(x).softmax(1)[0].cpu()
            acc = p if acc is None else acc + p
    return acc / (len(members) * (2 if tta else 1))


@torch.no_grad()
def evaluate_ensemble(members, classes, data_dir=DATA_DIR, tta=False):
    root = find_image_root(Path(data_dir))
    val_dir = root / "test" if (root / "test").exists() else root / "val"
    # per-member transforms differ by resolution: evaluate member-by-member
    # on the fly would reload data; simpler: run each member over the set once.
    from collections import defaultdict
    img_paths, labels = [], []
    ds0 = datasets.ImageFolder(str(val_dir if val_dir.exists() else root))
    img_paths = [s[0] for s in ds0.samples]
    labels = [s[1] for s in ds0.samples]
    summed = [torch.zeros(len(classes)) for _ in img_paths]
    for model, _, tfm in members:
        ds = datasets.ImageFolder(str(val_dir if val_dir.exists() else root), transform=tfm)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        i = 0
        for x, _ in tqdm(loader, desc="member", leave=False):
            x = x.to(DEVICE)
            logits = model(x)
            if tta:
                logits = (logits + model(torch.flip(x, dims=[3]))) / 2
            for p in logits.softmax(1).cpu():
                summed[i] += p / len(members)
                i += 1
    pred = [int(s.argmax()) for s in summed]
    acc = sum(a == b for a, b in zip(pred, labels)) / len(labels)
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                   default=["checkpoints/best_effb0.pth", "checkpoints/best_effb3.pth"])
    p.add_argument("--data", default=str(DATA_DIR))
    p.add_argument("--tta", action="store_true")
    p.add_argument("--image", default=None)
    args = p.parse_args()

    members, classes = load_ensemble(args.models)
    if args.image:
        probs = ensemble_probs(members, Image.open(args.image), args.tta)
        i = int(probs.argmax())
        print(f"Ensemble: {pretty(classes[i])} ({probs[i]:.4f})")
    else:
        acc = evaluate_ensemble(members, classes, args.data, args.tta)
        print(f"Ensemble accuracy: {acc:.4f} (TTA={args.tta})")


if __name__ == "__main__":
    main()

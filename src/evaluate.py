"""Evaluate checkpoint: accuracy, report, confusion matrix."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from .config import BATCH_SIZE, CHECKPOINT_PATH, CLASS_NAMES_PATH, DATA_DIR, DEVICE, IMG_SIZE, NUM_WORKERS
from .dataset import eval_transforms, find_image_root
from .model import get_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA_DIR))
    p.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    p.add_argument("--tta", action="store_true",
                   help="test-time augmentation: average original + horizontal flip")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    classes = ckpt.get("classes") or json.loads(open(CLASS_NAMES_PATH).read())
    img_size = int(ckpt.get("img_size", IMG_SIZE))
    model = get_model(
        len(classes), arch=ckpt.get("arch", "mobilenet_v3_small"), pretrained=False
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    from pathlib import Path
    from torchvision import datasets
    from torch.utils.data import DataLoader
    root = find_image_root(Path(args.data))
    val_dir = root / "test" if (root / "test").exists() else root / "val"
    ds = datasets.ImageFolder(str(val_dir if val_dir.exists() else root),
                              transform=eval_transforms(img_size))
    if ds.classes != classes:
        raise ValueError(
            f"Dataset classes {ds.classes} do not match checkpoint classes {classes}"
        )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    ys, ps = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="eval"):
            x = x.to(DEVICE)
            logits = model(x)
            if args.tta:  # average with horizontally flipped view
                logits = (logits + model(torch.flip(x, dims=[3]))) / 2
            pred = logits.argmax(1).cpu()
            ys += y.tolist()
            ps += pred.tolist()
    acc = sum(a == b for a, b in zip(ys, ps)) / len(ys)
    print(f"Accuracy: {acc:.4f} (TTA={args.tta})")

    names = ds.classes if hasattr(ds, "classes") else classes
    print(classification_report(ys, ps, target_names=names, digits=4))
    cm = confusion_matrix(ys, ps)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cm)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("pred"); ax.set_ylabel("true"); ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion_matrix.png")


if __name__ == "__main__":
    main()

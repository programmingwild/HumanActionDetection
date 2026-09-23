"""Train loop with best-checkpoint saving.

Upgrades: AdamW, label smoothing, mixed precision (AMP) on CUDA,
reproducible seed, --out so experiments never clobber the best model.

Usage:
  python -m src.train --epochs 12
  python -m src.train --arch efficientnet_b0 --epochs 25 --batch-size 64 --out checkpoints/best_effb0.pth
"""
import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import (
    ARCH, BATCH_SIZE, CHECKPOINT_DIR, CHECKPOINT_PATH, CLASS_NAMES_PATH,
    DATA_DIR, DEVICE, IMG_SIZE, LR, NUM_EPOCHS, NUM_WORKERS,
)
from .dataset import get_dataloaders
from .model import get_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mixup_data(x, y, alpha=0.4):
    """Return mixed inputs and paired targets for mixup regularization."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, mixup_alpha=0.0):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if mixup_alpha > 0:
            x, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(x)
                if mixup_alpha > 0:
                    loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
                else:
                    loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            if mixup_alpha > 0:
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            else:
                loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * len(y)
        total += len(y)
        correct += (out.argmax(1) == y).sum().item()
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in tqdm(loader, desc="val", leave=False):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * len(y)
        total += len(y)
        correct += (out.argmax(1) == y).sum().item()
    return loss_sum / total, correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA_DIR))
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--arch", default=ARCH)
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--mixup", type=float, default=0.0, help="mixup alpha (0 = off, try 0.4)")
    p.add_argument("--randaugment", action="store_true")
    p.add_argument("--out", default=str(CHECKPOINT_PATH))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--resume", default=None, help="checkpoint path to resume weights/best from")
    p.add_argument("--start-epoch", type=int, default=1)
    args = p.parse_args()

    set_seed(args.seed)
    use_amp = torch.cuda.is_available() and not args.no_amp
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"Device={DEVICE} AMP={use_amp} arch={args.arch}")

    train_loader, val_loader, class_names = get_dataloaders(
        args.data, batch_size=args.batch_size, num_workers=NUM_WORKERS,
        img_size=args.img_size, use_randaugment=args.randaugment)
    print(f"Classes ({len(class_names)}): {class_names}")

    model = get_model(len(class_names), arch=args.arch).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)

    best_acc = 0.0
    if args.resume:
        r = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(r["model_state"])
        best_acc = float(r.get("acc", 0.0))
        print(f"Resumed weights from {args.resume} (best={best_acc:.4f}), "
              f"starting at epoch {args.start_epoch}")
    for epoch in range(args.start_epoch, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scaler, args.mixup)
        va_loss, va_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs} "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={va_loss:.4f} acc={va_acc:.4f}")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model_state": model.state_dict(), "arch": args.arch,
                        "classes": class_names, "acc": best_acc,
                        "img_size": args.img_size, "epoch": epoch},
                       args.out)
            print(f"  -> saved {args.out} (acc={best_acc:.4f})")
    print(f"Done. Best val acc={best_acc:.4f}")


if __name__ == "__main__":
    main()

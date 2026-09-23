"""Train 3D-CNN on KTH clips.

Usage:
  python -m src.download_data --dataset vafaeii/kth-action-recognition-dataset --dest data_kth
  # organize flat KTH files -> data_kth/<class>/*.avi (auto-done below)
  python -m src.train_video --data data_kth --epochs 10 --arch r3d_18
"""
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm

from .config import (CHECKPOINT_DIR, DEVICE, VIDEO_ARCH, VIDEO_BATCH_SIZE,
                     VIDEO_DATA_DIR, VIDEO_CHECKPOINT_PATH, VIDEO_EPOCHS)
from .model_3d import get_video_model
from .video_dataset import get_video_dataloaders, organize_kth


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in tqdm(loader, desc="train" if train else "val", leave=False):
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(y)
            total += len(y)
            correct += (out.argmax(1) == y).sum().item()
    return loss_sum / total, correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(VIDEO_DATA_DIR))
    p.add_argument("--epochs", type=int, default=VIDEO_EPOCHS)
    p.add_argument("--batch-size", type=int, default=VIDEO_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--arch", default=VIDEO_ARCH)
    args = p.parse_args()

    organize_kth(args.data)
    train_loader, val_loader, classes = get_video_dataloaders(args.data, batch_size=args.batch_size)
    model = get_video_model(len(classes), arch=args.arch).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, DEVICE, True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, DEVICE, False)
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs} train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={va_loss:.4f} acc={va_acc:.4f}")
        if va_acc > best:
            best = va_acc
            torch.save({"model_state": model.state_dict(), "arch": args.arch,
                        "classes": classes, "acc": best}, VIDEO_CHECKPOINT_PATH)
            print(f"  -> saved {VIDEO_CHECKPOINT_PATH} (acc={best:.4f})")
    print(f"Done. Best val acc={best:.4f}")


if __name__ == "__main__":
    main()

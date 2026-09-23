"""Clip dataset for video action recognition (KTH / UCF-style).

Expected layout: root/<class>/*.avi|mp4
KTH Kaggle mirror is flat (person01_boxing_d1.avi) -> use organize_kth() first.

Each item: clip tensor (C, T, H, W), label.
"""
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms

from .config import CLIP_LEN, VIDEO_BATCH_SIZE, VIDEO_IMG_SIZE
from .dataset import IMAGENET_MEAN, IMAGENET_STD

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}


def organize_kth(src_dir, dest_dir=None):
    """Sort flat KTH files personXX_<action>_dX.avi into <action>/ folders."""
    src, dest = Path(src_dir), Path(dest_dir or src_dir)
    moved, skipped = 0, 0
    for f in src.rglob("*"):
        if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
            if f.parent.name.lower() in ("boxing", "handclapping", "handwaving",
                                          "jogging", "running", "walking"):
                continue  # already organized
            parts = f.stem.split("_")
            action = next((p.lower() for p in parts
                           if p.lower() in ("boxing", "handclapping", "handwaving",
                                            "jogging", "running", "walking")), None)
            if action is None:
                skipped += 1
                continue
            target = dest / action
            target.mkdir(parents=True, exist_ok=True)
            if f.parent != target:
                f.rename(target / f.name)
                moved += 1
    print(f"organize_kth: moved={moved} skipped={skipped}")
    return dest


def read_frames_uniform(path, clip_len=CLIP_LEN):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames: {path}")
    # uniform sample (or tile if too short)
    import numpy as np
    idx = np.linspace(0, len(frames) - 1, clip_len).astype(int)
    return [frames[i] for i in idx]


class VideoClipDataset(Dataset):
    def __init__(self, root, clip_len=CLIP_LEN, img_size=VIDEO_IMG_SIZE, train=True):
        self.root = Path(root)
        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.classes = classes
        self.cls_to_idx = {c: i for i, c in enumerate(classes)}
        self.items = []
        for c in classes:
            for f in (self.root / c).iterdir():
                if f.suffix.lower() in VIDEO_EXTS:
                    self.items.append((f, self.cls_to_idx[c]))
        if not self.items:
            raise RuntimeError(f"No videos under {root}/<class>/*.avi")
        self.clip_len, self.train = clip_len, train
        self.tfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.flip = transforms.RandomHorizontalFlip(p=1.0)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        frames = read_frames_uniform(path, self.clip_len)
        import random
        do_flip = self.train and random.random() < 0.5
        tensors = []
        for fr in frames:
            t = self.tfm(fr)
            if do_flip:
                t = self.flip(t)
            tensors.append(t)
        clip = torch.stack(tensors).permute(1, 0, 2, 3)  # (C,T,H,W)
        return clip, label


def get_video_dataloaders(data_dir, batch_size=VIDEO_BATCH_SIZE, val_split=0.2, num_workers=2):
    data_dir = Path(data_dir)
    if not any(d.is_dir() for d in data_dir.iterdir()):
        raise RuntimeError(f"{data_dir} is empty. Download KTH first.")
    # allow train/ subdir or flat class layout
    root = data_dir / "train" if (data_dir / "train").exists() else data_dir
    train_full = VideoClipDataset(root, train=True)
    eval_full = VideoClipDataset(root, train=False)
    n_val = max(1, int(len(train_full) * val_split))
    if n_val >= len(train_full):
        raise ValueError("Video dataset must contain at least two clips")
    g = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(len(train_full)), [len(train_full) - n_val, n_val], generator=g
    )
    train_set = Subset(train_full, train_indices.indices)
    val_set = Subset(eval_full, val_indices.indices)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    print(f"Video clips: train={len(train_set)} val={len(val_set)} classes={train_full.classes}")
    return train_loader, val_loader, train_full.classes

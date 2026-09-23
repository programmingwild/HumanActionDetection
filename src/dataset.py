"""Dataset + dataloaders (ImageFolder-compatible)."""
from pathlib import Path

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .config import BATCH_SIZE, IMG_SIZE, NUM_WORKERS, VAL_SPLIT

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transforms(size=IMG_SIZE, use_randaugment=False):
    aug = [transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
           transforms.RandomHorizontalFlip()]
    if use_randaugment:
        aug.append(transforms.RandAugment(num_ops=2, magnitude=9))
    else:
        aug += [transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)]
    aug += [transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25)]
    return transforms.Compose(aug)


def eval_transforms(size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def find_image_root(data_dir: Path) -> Path:
    """Handle data/train+test, data/train+val, or flat data/<class>/... layouts."""
    data_dir = Path(data_dir)
    if (data_dir / "train").exists():
        return data_dir
    # one nested level (e.g. data/archive/train)
    for sub in data_dir.iterdir():
        if sub.is_dir() and (sub / "train").exists():
            return sub
    return data_dir


def validation_size(dataset_size, val_split):
    if dataset_size < 2:
        raise ValueError("Dataset must contain at least two images")
    return min(max(1, int(dataset_size * val_split)), dataset_size - 1)


def dataset_classes(dataset):
    base = dataset
    while hasattr(base, "dataset"):
        base = base.dataset
    return base.classes


def get_dataloaders(data_dir, batch_size=BATCH_SIZE, val_split=VAL_SPLIT,
                    num_workers=NUM_WORKERS, img_size=IMG_SIZE, use_randaugment=False):
    root = find_image_root(Path(data_dir))
    print(f"Using dataset root: {root}")

    if (root / "train").exists():
        train_ds = datasets.ImageFolder(
            str(root / "train"), transform=train_transforms(img_size, use_randaugment))
        # prefer test/, else val/, else split train
        if (root / "test").exists():
            val_ds = datasets.ImageFolder(str(root / "test"), transform=eval_transforms(img_size))
        elif (root / "val").exists():
            val_ds = datasets.ImageFolder(str(root / "val"), transform=eval_transforms(img_size))
        else:
            n_val = validation_size(len(train_ds), val_split)
            train_ds, val_ds = random_split(train_ds, [len(train_ds) - n_val, n_val])
            val_ds.dataset.transform = None  # random_split shares dataset; fix below
            # rebuild val with eval transforms via Subset + transform wrapper
            from torch.utils.data import Subset
            from torchvision.datasets import ImageFolder
            full = ImageFolder(str(root / "train"), transform=eval_transforms(img_size))
            val_idx = val_ds.indices
            train_idx = train_ds.indices
            train_ds = Subset(ImageFolder(
                str(root / "train"),
                transform=train_transforms(img_size, use_randaugment)), train_idx)
            val_ds = Subset(full, val_idx)
        if dataset_classes(train_ds) != dataset_classes(val_ds):
            raise ValueError(
                f"Train classes {dataset_classes(train_ds)} do not match "
                f"validation classes {dataset_classes(val_ds)}"
            )
    else:
        # flat <class>/ layout -> random split 85/15
        full_train = datasets.ImageFolder(
            str(root), transform=train_transforms(img_size, use_randaugment))
        full_eval = datasets.ImageFolder(str(root), transform=eval_transforms(img_size))
        n = len(full_train)
        n_val = validation_size(n, val_split)
        n_train = n - n_val
        from torch.utils.data import Subset
        import torch
        g = torch.Generator().manual_seed(42)
        train_idx, val_idx = random_split(range(n), [n_train, n_val], generator=g)
        train_ds = Subset(full_train, train_idx.indices if hasattr(train_idx, "indices") else train_idx)
        val_ds = Subset(full_eval, val_idx.indices if hasattr(val_idx, "indices") else val_idx)

    class_names = None
    for ds in (train_ds, val_ds):
        base = ds.dataset if hasattr(ds, "dataset") else ds
        if hasattr(base, "classes"):
            class_names = base.classes
            break

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Classes: {class_names}")
    return train_loader, val_loader, class_names

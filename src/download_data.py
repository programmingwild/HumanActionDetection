"""Download the Kaggle dataset.

Setup (one time):
  1. Create Kaggle account -> Settings -> Create New API Token -> saves kaggle.json
  2. Place it at C:\\Users\\<you>\\.kaggle\\kaggle.json
     (or set KAGGLE_USERNAME / KAGGLE_KEY env vars)
  3. pip install -r requirements.txt

Usage:
  python -m src.download_data
  python -m src.download_data --dataset vafaeii/kth-action-recognition-dataset
"""
import argparse
import shutil
import zipfile
from pathlib import Path

from .config import DATA_DIR, KAGGLE_DATASET


def download_with_kaggle_cli(dataset: str, dest: Path):
    import subprocess
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(dest), "--unzip"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def download_with_kagglehub(dataset: str, dest: Path):
    import kagglehub
    print(f"Downloading {dataset} via kagglehub...")
    src = Path(kagglehub.dataset_download(dataset))
    dest.mkdir(parents=True, exist_ok=True)
    # kagglehub downloads to cache; copy contents into data/
    for item in src.iterdir():
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    print(f"Copied to {dest}")


def print_tree(root: Path, max_depth: int = 3):
    for i, p in enumerate(sorted(root.rglob("*"))):
        depth = len(p.relative_to(root).parts)
        if depth > max_depth:
            continue
        if p.is_dir():
            print(f"[DIR]  {p.relative_to(root)}")
        elif i < 40:
            print(f"       {p.relative_to(root)}")


def maybe_flatten_single_subdir(dest: Path):
    """Some zips extract to data/<dataset-name>/train... -> move up one level."""
    subdirs = [d for d in dest.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        inner = subdirs[0]
        if (inner / "train").exists() or any((inner).glob("*")):
            # If inner looks like the real root and outer has nothing else, flatten
            if not (dest / "train").exists():
                for item in inner.iterdir():
                    shutil.move(str(item), str(dest / item.name))
                inner.rmdir()
                print(f"Flattened {inner.name}/ into data/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=KAGGLE_DATASET)
    parser.add_argument("--dest", default=str(DATA_DIR))
    args = parser.parse_args()

    dest = Path(args.dest)
    try:
        download_with_kaggle_cli(args.dataset, dest)
    except Exception as e:
        print(f"kaggle CLI failed ({e}), trying kagglehub...")
        download_with_kagglehub(args.dataset, dest)

    # Unzip any leftover zips
    for z in dest.glob("*.zip"):
        print(f"Unzipping {z.name}...")
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(dest)
        z.unlink()

    maybe_flatten_single_subdir(dest)
    print("\nDataset layout:")
    print_tree(dest)


if __name__ == "__main__":
    main()

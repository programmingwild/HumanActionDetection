"""Central config for Human Action Detection project."""
from pathlib import Path
import torch

# Kaggle dataset choice (image-based, 15 everyday actions, train/test split included).
# Why this one: small (~218 MB), ImageFolder-ready, diverse classes, trains in minutes.
# Alternative video-native option: vafaeii/kth-action-recognition-dataset (600 videos, 6 classes).
KAGGLE_DATASET = "shashankrapolu/human-action-recognition-dataset"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_action_model.pth"
CLASS_NAMES_PATH = CHECKPOINT_DIR / "class_names.json"

# 15 classes in this dataset
CLASS_NAMES = [
    "calling", "clapping", "cycling", "dancing", "drinking",
    "eating", "fighting", "hugging", "laughing", "listening_to_music",
    "running", "sitting", "sleeping", "texting", "using_laptop",
]

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 12
LR = 3e-4
NUM_WORKERS = 2
VAL_SPLIT = 0.15  # only used if dataset has no test/ folder
ARCH = "mobilenet_v3_small"  # fast + accurate; alt: resnet18, efficientnet_b0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Video inference smoothing
SMOOTH_WINDOW = 15
FRAME_STRIDE = 2  # predict every Nth frame for speed

# ---- True-temporal video branch (KTH, 3D-CNN) ----
KAGGLE_VIDEO_DATASET = "vafaeii/kth-action-recognition-dataset"  # 600 videos, 6 classes
VIDEO_DATA_DIR = PROJECT_ROOT / "data_kth"
VIDEO_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_video_model.pth"
KTH_CLASSES = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]
CLIP_LEN = 16        # frames per clip for R3D
VIDEO_IMG_SIZE = 112  # 112 = fast, 224 = accurate
VIDEO_BATCH_SIZE = 8
VIDEO_EPOCHS = 10
VIDEO_ARCH = "r3d_18"  # alt: mc3_18

# ---- YOLO person-crop + action head ----
YOLO_WEIGHTS = str(CHECKPOINT_DIR / "yolov8n.pt")  # auto-downloaded by ultralytics
YOLO_CONF = 0.4

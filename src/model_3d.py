"""3D-CNN video models (true temporal)."""
import torch.nn as nn
from torchvision import models


def get_video_model(num_classes: int, arch: str = "r3d_18", pretrained: bool = True):
    arch = arch.lower()
    if arch == "r3d_18":
        w = models.video.R3D_18_Weights.DEFAULT if pretrained else None
        model = models.video.r3d_18(weights=w)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "mc3_18":
        w = models.video.MC3_18_Weights.DEFAULT if pretrained else None
        model = models.video.mc3_18(weights=w)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown video arch: {arch} (use r3d_18, mc3_18)")
    return model

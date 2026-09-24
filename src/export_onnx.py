"""Export 2D classifiers (and YOLO) to ONNX for light serving.

Usage:
  python -m src.export_onnx                      # B0 + B3 + YOLOv8n
  python -m src.export_onnx --arch efficientnet_b0 --checkpoint checkpoints/best_effb0.pth
"""
import argparse
from pathlib import Path

import torch

from .config import CHECKPOINT_DIR
from .model import get_model


def export_classifier(arch, checkpoint, out, img_size=224):
    from .infer import load_model
    model, classes = load_model(checkpoint)
    model = model.to("cpu").eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model, dummy, out, opset_version=14, dynamo=False,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"Wrote {out} ({Path(out).stat().st_size / 1024:.0f} KB) arch={arch} img={img_size}")


def export_yolo(out):
    from ultralytics import YOLO
    from .config import YOLO_WEIGHTS
    model = YOLO(YOLO_WEIGHTS)
    model.export(format="onnx", imgsz=640)
    # ultralytics saves beside weights as yolov8n.onnx
    src = Path(YOLO_WEIGHTS).with_suffix(".onnx")
    src.rename(out)
    print(f"Wrote {out} ({Path(out).stat().st_size / 1024:.0f} KB)")


def export_r3d(checkpoint, out, clip_len=16, size=112):
    import torch
    from .model_3d import get_video_model
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = get_video_model(len(ckpt["classes"]), arch=ckpt.get("arch", "r3d_18"),
                            pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to("cpu").eval()
    dummy = torch.randn(1, 3, clip_len, size, size)
    torch.onnx.export(
        model, dummy, out, opset_version=14, dynamo=False,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"Wrote {out} ({Path(out).stat().st_size / 1024:.0f} KB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--out", default=None)
    p.add_argument("--yolo-only", action="store_true")
    args = p.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if args.yolo_only:
        export_yolo(str(CHECKPOINT_DIR / "yolov8n.onnx"))
        return
    if args.arch and args.checkpoint:
        out = args.out or str(CHECKPOINT_DIR / f"{args.arch}.onnx")
        export_classifier(args.arch, args.checkpoint, out, args.img_size)
        return
    export_classifier("efficientnet_b0", "checkpoints/best_effb0.pth",
                      str(CHECKPOINT_DIR / "effb0.onnx"), 224)
    export_classifier("efficientnet_b3", "checkpoints/best_action_model.pth",
                      str(CHECKPOINT_DIR / "effb3.onnx"), 300)
    from .config import VIDEO_CHECKPOINT_PATH, CLIP_LEN, VIDEO_IMG_SIZE
    export_r3d(str(VIDEO_CHECKPOINT_PATH), str(CHECKPOINT_DIR / "r3d18.onnx"),
               CLIP_LEN, VIDEO_IMG_SIZE)
    export_yolo(str(CHECKPOINT_DIR / "yolov8n.onnx"))


if __name__ == "__main__":
    main()

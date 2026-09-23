"""Sliding-window inference with trained 3D-CNN.

  Buffers last CLIP_LEN frames, runs R3D every --stride frames,
  averages recent clip probs for stability.

Usage:
  python -m src.inference_video_3d input.avi -o out.mp4
  python -m src.inference_video_3d 0  (webcam; pass digits as source)
"""
import argparse
from collections import deque

import cv2
import torch
from torchvision import transforms

from .config import CLIP_LEN, DEVICE, VIDEO_CHECKPOINT_PATH, VIDEO_IMG_SIZE
from .dataset import IMAGENET_MEAN, IMAGENET_STD
from .labels import pretty
from .model_3d import get_video_model


def load_video_model(checkpoint=VIDEO_CHECKPOINT_PATH):
    ckpt = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    classes = ckpt.get("classes") or ckpt.get("class_names")
    state_dict = ckpt.get("model_state") or ckpt.get("state_dict")
    if classes is None or state_dict is None:
        raise KeyError(f"Checkpoint {checkpoint} is missing classes or model state")
    model = get_video_model(
        len(classes), arch=ckpt.get("arch", "r3d_18"), pretrained=False
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model, classes


def get_tfm():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((VIDEO_IMG_SIZE, VIDEO_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@torch.no_grad()
def run(source, out=None, stride=4, window=4, checkpoint=VIDEO_CHECKPOINT_PATH):
    model, classes = load_video_model(checkpoint)
    tfm = get_tfm()
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {source}")
    frame_buf, prob_buf = deque(maxlen=CLIP_LEN), deque(maxlen=window)
    writer, label, conf, i = None, "...", 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_buf.append(tfm(rgb))
        if len(frame_buf) == CLIP_LEN and i % stride == 0:
            clip = torch.stack(list(frame_buf)).permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)
            probs = model(clip).softmax(1)[0].cpu()
            prob_buf.append(probs)
            avg = torch.stack(list(prob_buf)).mean(0)
            label, conf = pretty(classes[avg.argmax().item()]), avg.max().item()
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(frame, f"3D:{label} {conf:.2f}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 128, 0), 2)
        if out and writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        if writer:
            writer.write(frame)
        if str(source).isdigit():
            cv2.imshow("3d-action", frame)
            if cv2.waitKey(1) == 27:
                break
        i += 1
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Saved -> {out}" if out else "Done")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--checkpoint", default=str(VIDEO_CHECKPOINT_PATH))
    args = p.parse_args()
    run(args.source, args.out, args.stride, args.window, args.checkpoint)


if __name__ == "__main__":
    main()

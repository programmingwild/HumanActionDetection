"""Image / video / webcam inference with temporal smoothing."""
import argparse
import json
from collections import deque

import cv2
import torch
from PIL import Image

from .config import CHECKPOINT_PATH, DEVICE, FRAME_STRIDE, IMG_SIZE, SMOOTH_WINDOW
from .dataset import IMAGENET_MEAN, IMAGENET_STD
from .labels import pretty
from .model import get_model
from torchvision import transforms


def load_model(checkpoint=CHECKPOINT_PATH):
    ckpt = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    classes = ckpt.get("classes") or ckpt.get("class_names")
    state_dict = ckpt.get("model_state") or ckpt.get("state_dict")
    if classes is None or state_dict is None:
        raise KeyError(f"Checkpoint {checkpoint} is missing classes or model state")
    model = get_model(
        len(classes), arch=ckpt.get("arch", "mobilenet_v3_small"), pretrained=False
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model, classes


def checkpoint_img_size(checkpoint=CHECKPOINT_PATH):
    """Read the training resolution stored in a checkpoint (default 224)."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return int(ckpt.get("img_size", IMG_SIZE))


def get_tf(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@torch.no_grad()
def predict_image(model, classes, path, tf=None):
    tf = tf or get_tf()
    img = Image.open(path).convert("RGB")
    x = tf(img).unsqueeze(0).to(DEVICE)
    probs = model(x).softmax(1)[0]
    i = probs.argmax().item()
    return pretty(classes[i]), probs[i].item()


def predict_video(model, classes, src, out=None, stride=FRAME_STRIDE, window=SMOOTH_WINDOW,
                  img_size=IMG_SIZE):
    tf = get_tf(img_size)
    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else str(src))
    buf = deque(maxlen=window)
    w = h = None
    writer = None
    with torch.no_grad():
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                x = tf(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
                buf.append(model(x).softmax(1)[0].cpu())
                smoothed = torch.stack(list(buf)).mean(0)
                label = pretty(classes[smoothed.argmax().item()])
                conf = smoothed.max().item()
                cv2.putText(frame, f"{label} {conf:.2f}", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            if out and writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            if writer is not None:
                writer.write(frame)
            if str(src).isdigit():
                cv2.imshow("action", frame)
                if cv2.waitKey(1) == 27:
                    break
            idx += 1
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=None)
    p.add_argument("--video", default=None)
    p.add_argument("--webcam", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    args = p.parse_args()
    model, classes = load_model(args.checkpoint)
    img_size = checkpoint_img_size(args.checkpoint)
    if args.image:
        print(predict_image(model, classes, args.image, tf=get_tf(img_size)))
    elif args.video:
        predict_video(model, classes, args.video, out=args.out, img_size=img_size)
    elif args.webcam:
        predict_video(model, classes, "0", img_size=img_size)
    else:
        p.error("pass --image, --video, or --webcam")


if __name__ == "__main__":
    main()

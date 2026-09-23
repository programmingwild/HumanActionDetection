"""YOLO person detection + 2D action classifier head.

True 'detection': YOLO localizes each person, our CNN classifies the crop.
Draws per-person box + action label. Falls back to full-frame if no person found.

Usage:
  pip install ultralytics opencv-python
  python -m src.yolo_action --source 0                    # webcam
  python -m src.yolo_action --source input.mp4 --out out.mp4
  python -m src.yolo_action --source photo.jpg --out pred.jpg
  python -m src.yolo_action --source input.mp4 --no-yolo  # full-frame only (no detector)
"""
import argparse

import cv2
import torch
from PIL import Image

from .config import CHECKPOINT_PATH, DEVICE, YOLO_CONF, YOLO_WEIGHTS
from .infer import checkpoint_img_size, get_tf, load_model
from .labels import pretty


def get_yolo(weights=YOLO_WEIGHTS):
    from ultralytics import YOLO
    return YOLO(weights)


@torch.no_grad()
def classify_crop(model, classes, tf, crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    x = tf(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    probs = model(x).softmax(1)[0]
    i = probs.argmax().item()
    return classes[i], probs[i].item()


def person_boxes(yolo, frame, conf=YOLO_CONF):
    """Return person boxes [[x1,y1,x2,y2,conf]] (class 0 = person)."""
    res = yolo.predict(frame, conf=conf, classes=[0], verbose=False)[0]
    boxes = []
    if res.boxes is not None:
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            boxes.append([x1, y1, x2, y2, float(b.conf[0])])
    return boxes


def annotate_frame(frame, boxes, model, classes, tf):
    h, w = frame.shape[:2]
    if not boxes:  # no person -> classify full frame
        label, conf = classify_crop(model, classes, tf, frame)
        cv2.putText(frame, f"{pretty(label)} {conf:.2f}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return frame
    for x1, y1, x2, y2, det_conf in boxes:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        label, conf = classify_crop(model, classes, tf, crop)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1 - 28), (x1 + 260, y1), (0, 255, 0), -1)
        cv2.putText(frame, f"{pretty(label)} {conf:.2f}", (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0")
    p.add_argument("--out", default=None)
    p.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    p.add_argument("--yolo-weights", default=YOLO_WEIGHTS)
    p.add_argument("--conf", type=float, default=YOLO_CONF)
    p.add_argument("--no-yolo", action="store_true", help="skip detector, full-frame classify")
    args = p.parse_args()

    model, classes = load_model(args.checkpoint)
    tf = get_tf(checkpoint_img_size(args.checkpoint))
    yolo = None if args.no_yolo else get_yolo(args.yolo_weights)

    src = args.source
    if src.lower().endswith((".jpg", ".jpeg", ".png")):
        frame = cv2.imread(src)
        if frame is None:
            raise SystemExit(f"Cannot read image {src}")
        boxes = [] if yolo is None else person_boxes(yolo, frame, args.conf)
        cv2.imwrite(args.out or "yolo_action_pred.jpg", annotate_frame(frame, boxes, model, classes, tf))
        print(f"Saved -> {args.out or 'yolo_action_pred.jpg'}")
        return

    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else str(src))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {src}")
    writer = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = [] if yolo is None else person_boxes(yolo, frame, args.conf)
        frame = annotate_frame(frame, boxes, model, classes, tf)
        if args.out and writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        if writer:
            writer.write(frame)
        if str(src).isdigit():
            cv2.imshow("yolo+action (ESC quits)", frame)
            if cv2.waitKey(1) == 27:
                break
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

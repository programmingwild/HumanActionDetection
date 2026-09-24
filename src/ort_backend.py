"""Torch-free inference backend (onnxruntime + numpy + PIL/cv2 only).

Powers the Streamlit app so Streamlit Community Cloud (1GB RAM) can serve it.
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .labels import pretty

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASSES_2D = ["calling", "clapping", "cycling", "dancing", "drinking", "eating",
              "fighting", "hugging", "laughing", "listening_to_music", "running",
              "sitting", "sleeping", "texting", "using_laptop"]
CLASSES_3D = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]


def preprocess_pil(img: Image.Image, size: int):
    arr = np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR)).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)[None]


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class OrtEnsemble:
    """B0@224 + B3@300 averaged (matches torch ensemble bit-near)."""

    def __init__(self, b0_path, b3_path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.s0 = ort.InferenceSession(str(b0_path), sess_options=opts,
                                       providers=["CPUExecutionProvider"])
        self.s3 = ort.InferenceSession(str(b3_path), sess_options=opts,
                                       providers=["CPUExecutionProvider"])

    def probs(self, img: Image.Image):
        p0 = softmax(self.s0.run(None, {"input": preprocess_pil(img, 224)})[0][0])
        p3 = softmax(self.s3.run(None, {"input": preprocess_pil(img, 300)})[0][0])
        return (p0 + p3) / 2

    def top(self, img: Image.Image):
        p = self.probs(img)
        i = int(p.argmax())
        return pretty(CLASSES_2D[i]), float(p[i]), {pretty(c): float(v) for c, v in zip(CLASSES_2D, p)}


def yolo_person_boxes(sess, frame_bgr, conf=0.4, imgsz=640):
    """frame BGR -> [[x1,y1,x2,y2,conf]] for class 0 (person). Pure numpy NMS."""
    import cv2
    h0, w0 = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    scale = min(imgsz / w0, imgsz / h0)
    nw, nh = int(w0 * scale), int(h0 * scale)
    canvas = np.zeros((imgsz, imgsz, 3), np.uint8)
    canvas[:nh, :nw] = np.asarray(Image.fromarray(rgb).resize((nw, nh)))
    x = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    out = sess.run(None, {"images": x})[0][0]  # (84, 8400)
    boxes_cxcywh, scores = out[:4], out[4:]
    person = scores[0]
    keep = np.where(person >= conf)[0]
    boxes = []
    for i in keep:
        cx, cy, w, h = boxes_cxcywh[:, i]
        x1 = (cx - w / 2) / scale
        y1 = (cy - h / 2) / scale
        x2 = (cx + w / 2) / scale
        y2 = (cy + h / 2) / scale
        boxes.append([x1, y1, x2, y2, float(person[i])])
    # NMS
    boxes.sort(key=lambda b: -b[4])
    final = []
    for b in boxes:
        keep_b = True
        for f in final:
            ix1, iy1 = max(b[0], f[0]), max(b[1], f[1])
            ix2, iy2 = min(b[2], f[2]), min(b[3], f[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area = (b[2] - b[0]) * (b[3] - b[1]) + (f[2] - f[0]) * (f[3] - f[1]) - inter
            if area > 0 and inter / area > 0.7:
                keep_b = False
                break
        if keep_b:
            b[0], b[1] = max(0, b[0]), max(0, b[1])
            b[2], b[3] = min(w0, b[2]), min(h0, b[3])
            final.append(b)
    return [[int(b[0]), int(b[1]), int(b[2]), int(b[3]), b[4]] for b in final]


class OrtVideo:
    def __init__(self, path, clip_len=16, size=112):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(str(path), sess_options=opts,
                                         providers=["CPUExecutionProvider"])
        self.clip_len, self.size = clip_len, size

    def probs(self, frames_rgb):
        idx = np.linspace(0, len(frames_rgb) - 1, self.clip_len).astype(int)
        arrs = []
        for i in idx:
            a = np.asarray(frames_rgb[i].convert("RGB").resize((self.size, self.size),
                                                               Image.BILINEAR)).astype(np.float32) / 255.0
            arrs.append(((a - MEAN) / STD).transpose(2, 0, 1))
        clip = np.stack(arrs).transpose(1, 0, 2, 3)[None].astype(np.float32)  # (1,C,T,H,W)
        return softmax(self.sess.run(None, {"input": clip})[0][0])


def ckpt_dir():
    return Path(__file__).resolve().parents[1] / "checkpoints"

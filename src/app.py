"""Human Action Detection — editorial Gradio studio.

A full-page, chapter-based interface: still-image classification, YOLO person
detection with per-person action labels, and a 3D video branch.

Usage:
  python -m src.app
  python -m src.app --share --port 7860
  SKIP_3D=1 python -m src.app   # lite mode for CPU-only hosting
"""
import html as _html
import math
import os
import threading
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .config import CHECKPOINT_PATH, DEVICE, VIDEO_CHECKPOINT_PATH, YOLO_CONF
from .ensemble import ensemble_probs, load_ensemble
from .infer import checkpoint_img_size, get_tf, load_model
from .labels import pretty
from .inference_video_3d import get_tfm as get_3d_tfm, load_video_model
from .model_3d import get_video_model  # noqa: F401
from .yolo_action import get_yolo, person_boxes

# ----------------------------------------------------------------------------
# Models — only the image ensemble blocks first paint.
# YOLO and the 126 MB 3D checkpoint are loaded on demand / warmed in background.
# ----------------------------------------------------------------------------
print("Loading image ensemble (B0 @224 + B3 @300)...")
try:
    ens_members, classes2d = load_ensemble(["checkpoints/best_effb0.pth", str(CHECKPOINT_PATH)])
    status_ens = "B0+B3 ensemble"
except Exception as e:
    print(f"Ensemble unavailable, falling back to single model: {e}")
    ens_members = None
    model2d, classes2d = load_model(str(CHECKPOINT_PATH))
    tf2d = get_tf(checkpoint_img_size(str(CHECKPOINT_PATH)))
    status_ens = "single model"

_yolo = None
_yolo_failed = False
_yolo_lock = threading.Lock()
_3d = None
_3d_failed = False
_3d_lock = threading.Lock()
SKIP_3D = os.environ.get("SKIP_3D") == "1"
status_yolo = "Ready"
status_3d = "Disabled (lite)" if SKIP_3D else "Ready"


def get_detector():
    """YOLOv8n — loaded once, on first use (or by the background warm-up)."""
    global _yolo, _yolo_failed
    if _yolo is not None or _yolo_failed:
        return _yolo
    with _yolo_lock:
        if _yolo is None and not _yolo_failed:
            try:
                print("Loading YOLOv8n...")
                _yolo = get_yolo()
            except Exception as e:
                print(f"YOLO unavailable: {e}")
                _yolo_failed = True
    return _yolo


def get_motion_model():
    """R3D-18 — the 126 MB checkpoint is never read at startup."""
    global _3d, _3d_failed
    if SKIP_3D or _3d is not None or _3d_failed:
        return _3d
    with _3d_lock:
        if _3d is None and not _3d_failed:
            try:
                print("Loading 3D video model...")
                model, classes = load_video_model(str(VIDEO_CHECKPOINT_PATH))
                _3d = (model, classes, get_3d_tfm())
            except Exception as e:
                print(f"3D model unavailable: {e}")
                _3d_failed = True
    return _3d


def warmup():
    """Fetch optional models after the UI is served, so scrolling never waits."""
    get_detector()
    get_motion_model()


def probs_for_image(img):
    """Ensemble probabilities for a PIL image (CPU tensor)."""
    if ens_members is not None:
        return ensemble_probs(ens_members, img)
    x = tf2d(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    return model2d(x).softmax(1)[0].cpu()


def top_for_image(img):
    probs = probs_for_image(img)
    i = int(probs.argmax())
    return classes2d[i], float(probs[i])

MODEL_FOOT = (f"{status_ens} · {len(ens_members)} models" if ens_members
              else "single model · 1 model")
SAMPLE_IMG = "data/test/cycling/Image_10739.jpg"
SAMPLE_VIDEO = "data/sample_boxing.mp4"
HERE = Path(__file__).resolve().parent

CSS = (HERE / "ui" / "theme.css").read_text(encoding="utf-8")
JS = (HERE / "ui" / "app.js").read_text(encoding="utf-8")

HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,300..800&family=Instrument+Serif:ital@0;1&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<meta name="theme-color" content="#070707">
"""


def _existing(paths):
    return [p for p in paths if Path(p).exists()]


def _font(size, bold=True):
    win = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidates = [
        win / ("arialbd.ttf" if bold else "arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ]
    for p in candidates:
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ----------------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------------
def render_empty(kind="still"):
    cls = " motion" if kind == "motion" else ""
    glyph = "✳"
    title = "Awaiting a frame" if kind != "motion" else "Awaiting a clip"
    copy = ("Drop a photo, use your webcam, or pick a sample. "
            "Inference starts automatically.") if kind != "motion" else (
            "Drop a short clip. Sixteen frames are sampled across time and read by R3D-18.")
    return f"""
<div class="verdict empty{cls}">
  <div class="empty-glyph">{glyph}</div>
  <div class="empty-k">{title}</div>
  <p class="empty-p">{copy}</p>
  <div class="empty-line"></div>
  <span class="empty-arr">↘</span>
</div>
"""


def render_verdict(probs, kicker="LIVE INFERENCE", tag="B0 + B3 · 300 PX",
                   note="TOP-1 OF 15 CLASSES", kind="still", latency=None, foot=None):
    items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    if not items:
        return render_empty(kind)
    top, conf = items[0]
    circ = 2 * math.pi * 56
    offset = circ * (1 - min(max(conf, 0.0), 1.0))
    bars = []
    for name, p in items[:5]:
        pct = p * 100
        bars.append(
            f'<div class="bar"><span class="bt">{_html.escape(name)}</span>'
            f'<div class="t"><i style="width:{pct:.2f}%"></i></div>'
            f'<span class="p">{pct:.1f}</span></div>'
        )
    latency_txt = f"{latency * 1000:.0f} ms" if latency is not None else "—"
    cls = " motion" if kind == "motion" else ""
    return f"""
<div class="verdict{cls}">
  <div class="v-head"><span class="live"><i></i>{_html.escape(kicker)}</span><span>{_html.escape(tag)}</span></div>
  <div class="v-main">
    <div>
      <div class="v-label">Predicted action</div>
      <div class="v-action">{_html.escape(top)}</div>
      <div class="v-note">{_html.escape(note)} · {latency_txt}</div>
    </div>
    <div class="ring">
      <svg viewBox="0 0 120 120">
        <circle class="bg" cx="60" cy="60" r="56"></circle>
        <circle class="fg" cx="60" cy="60" r="56" stroke-dasharray="{circ:.1f}" stroke-dashoffset="{offset:.1f}"></circle>
      </svg>
      <div class="in">{conf * 100:.1f}<small>%</small></div>
    </div>
  </div>
  <div class="v-bars">{''.join(bars)}</div>
  <div class="v-foot"><span>{_html.escape(foot or MODEL_FOOT)}</span><span>Inference {latency_txt}</span></div>
</div>
"""


def render_det_empty():
    return """
<div class="det-result">
  <div class="det-head">
    <div><div class="k">Scene read</div><b>0 <small>people</small></b></div>
    <span class="pill">Awaiting frame</span>
  </div>
  <div class="det-note">Drop an image to localise every person and classify each crop independently.</div>
  <div class="v-foot"><span>YOLOv8n · B0 + B3</span><span>—</span></div>
</div>
"""


def render_detection(rows, conf, latency, fallback_probs=None):
    if not rows and fallback_probs is not None:
        items = sorted(fallback_probs.items(), key=lambda kv: kv[1], reverse=True)
        top, prob = items[0]
        bars = "".join(
            f'<div class="bar"><span class="bt">{_html.escape(n)}</span>'
            f'<div class="t"><i style="width:{p * 100:.2f}%"></i></div>'
            f'<span class="p">{p * 100:.1f}</span></div>'
            for n, p in items[:4]
        )
        return f"""
<div class="det-result">
  <div class="det-head">
    <div><div class="k">No person detected</div><b>{_html.escape(top)} <small>full-frame</small></b></div>
    <span class="pill">Fallback read</span>
  </div>
  <div class="det-list" style="padding-top:14px">{bars}</div>
  <div class="det-note">Nobody crossed the {conf:.2f} confidence threshold, so the whole frame was classified.
  Lower the threshold to find smaller or partially occluded people.</div>
  <div class="v-foot"><span>YOLOv8n · {conf:.2f} threshold</span><span>{latency * 1000:.0f} ms</span></div>
</div>
"""
    rows_html = "".join(
        f'<div class="det-row"><span class="det-i">P{r["i"]}</span>'
        f'<span>{_html.escape(r["action"])}</span>'
        f'<div class="det-m"><i style="width:{r["prob"] * 100:.1f}%"></i></div>'
        f'<span class="det-p">{r["prob"] * 100:.1f}%</span></div>'
        for r in rows
    )
    n = len(rows)
    return f"""
<div class="det-result">
  <div class="det-head">
    <div><div class="k">Scene read</div><b>{n:02d} <small>{"person" if n == 1 else "people"}</small></b></div>
    <span class="pill">Conf {conf:.2f}</span>
  </div>
  <div class="det-list">{rows_html}</div>
  <div class="det-note">Each bounding box is cropped and classified on its own by the 2D action head.</div>
  <div class="v-foot"><span>YOLOv8n · {status_ens}</span><span>{latency * 1000:.0f} ms</span></div>
</div>
"""


def make_placeholder(label="DETECTION OUTPUT", sub="YOLOV8N + B0/B3"):
    w, h = 1200, 760
    img = Image.new("RGB", (w, h), (10, 10, 9))
    d = ImageDraw.Draw(img)
    for x in range(0, w, 60):
        d.line([(x, 0), (x, h)], fill=(16, 16, 14), width=1)
    for y in range(0, h, 60):
        d.line([(0, y), (w, y)], fill=(16, 16, 14), width=1)
    cx, cy = w // 2, h // 2
    acid = (215, 255, 63)
    for r in (54, 110, 168):
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(39, 39, 32), width=1)
    d.line([(cx - 24, cy), (cx + 24, cy)], fill=acid, width=2)
    d.line([(cx, cy - 24), (cx, cy + 24)], fill=acid, width=2)
    f_big = _font(21)
    f_small = _font(12)
    tw = d.textbbox((0, 0), label, font=f_big)[2]
    d.text(((w - tw) / 2, cy + 52), label, font=f_big, fill=(242, 238, 228))
    tw2 = d.textbbox((0, 0), sub, font=f_small)[2]
    d.text(((w - tw2) / 2, cy + 88), sub, font=f_small, fill=(92, 92, 85))
    d.text((22, h - 30), "ACTIONSCOPE / LIVE FRAME", font=f_small, fill=(70, 70, 63))
    return img


def draw_detections(frame_bgr, results):
    """Premium box render: acid brackets + mono label chips on a dark plate."""
    h, w = frame_bgr.shape[:2]
    base = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    over = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(over)
    acid = (215, 255, 63, 255)
    plate = (8, 8, 7, 235)
    lw = max(2, int(w * 0.0022))
    fsize = max(13, int(w * 0.0135))
    font = _font(fsize)
    L = max(14, int(w * 0.016))

    placed = []
    for r in results:
        x1, y1, x2, y2 = r["box"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        # corner brackets
        for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                                 (x1, y2, 1, -1), (x2, y2, -1, -1)):
            od.line([(px, py), (px + dx * L, py)], fill=acid, width=lw)
            od.line([(px, py), (px, py + dy * L)], fill=acid, width=lw)
        # label — stack downward when labels would collide
        txt = f'P{r["i"]} · {r["action"].upper()} {r["prob"] * 100:.0f}%'
        tb = od.textbbox((0, 0), txt, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        padx, pady = 9, 6
        lh = th + pady * 2
        lx = min(max(6, x1), w - tw - padx * 2 - 6)
        ly = y1 - lh - 5
        if ly < 6:
            ly = min(y2 + 5, h - lh - 6)
        for _ in range(14):
            rect = (lx, ly, lx + tw + padx * 2, ly + lh)
            if not any(not (rect[2] < p[0] or rect[0] > p[2] or
                            rect[3] < p[1] or rect[1] > p[3]) for p in placed):
                break
            ly = ly + lh + 4
            if ly + lh > h - 6:
                ly = 6
        placed.append((lx, ly, lx + tw + padx * 2, ly + lh))
        od.rounded_rectangle([lx, ly, lx + tw + padx * 2, ly + lh],
                             radius=6, fill=plate, outline=acid, width=1)
        od.text((lx + padx, ly + pady - tb[1]), txt, font=font, fill=acid)
    out = Image.alpha_composite(base, over)
    d = ImageDraw.Draw(out)
    small = _font(max(10, int(w * 0.009)))
    tag = "ACTIONSCOPE / YOLO + ACTION"
    d.text((18, h - 26), tag, font=small, fill=(120, 120, 108, 255))
    return out.convert("RGB")


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
@torch.no_grad()
def classify_image(img: Image.Image):
    if img is None:
        return render_empty()
    t0 = time.perf_counter()
    probs = probs_for_image(img)
    latency = time.perf_counter() - t0
    result = {pretty(c): float(p) for c, p in zip(classes2d, probs)}
    return render_verdict(result, latency=latency)


@torch.no_grad()
def detect_image(img: Image.Image, conf: float = YOLO_CONF):
    if img is None:
        return None, render_det_empty()
    t0 = time.perf_counter()
    frame = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    detector = get_detector()
    boxes = person_boxes(detector, frame, conf) if detector is not None else []
    rows = []
    if boxes:
        for i, (x1, y1, x2, y2, _det) in enumerate(boxes, 1):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            action, prob = top_for_image(Image.fromarray(rgb))
            rows.append({"i": i, "action": pretty(action), "prob": prob,
                         "box": (x1, y1, x2, y2)})
    latency = time.perf_counter() - t0
    if rows:
        annotated = draw_detections(frame, rows)
        return annotated, render_detection(rows, conf, latency)
    # no person: classify the full frame and report honestly
    probs = probs_for_image(img)
    result = {pretty(c): float(p) for c, p in zip(classes2d, probs)}
    return img, render_detection([], conf, time.perf_counter() - t0, result)


@torch.no_grad()
def classify_video_clip(video_path):
    if not video_path:
        return render_empty("motion")
    loaded = get_motion_model()
    if loaded is None:
        msg = "This deployment runs in lite mode." if SKIP_3D else "The 3D checkpoint could not be loaded."
        return f'<div class="verdict empty motion"><div class="empty-k">3D model unavailable</div><p class="empty-p">{msg}</p></div>'
    model3d, classes3d, tf3d = loaded
    from .config import CLIP_LEN
    from .video_dataset import read_frames_uniform
    t0 = time.perf_counter()
    try:
        frames = read_frames_uniform(video_path, CLIP_LEN)
        clip = torch.stack([tf3d(f) for f in frames]).permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)
        probs = model3d(clip).softmax(1)[0].cpu()
    except Exception as e:
        print(f"Video inference failed: {e}")
        return ('<div class="verdict empty motion"><div class="empty-k">Clip unreadable</div>'
                '<p class="empty-p">Try a short MP4, AVI or MOV with a single clear action.</p></div>')
    latency = time.perf_counter() - t0
    result = {pretty(c): float(p) for c, p in zip(classes3d, probs)}
    return render_verdict(result, kicker="LIVE INFERENCE · 3D", tag="R3D-18 · 16 FRAMES",
                          note="TOP-1 OF 6 KTH ACTIONS", kind="motion", latency=latency,
                          foot="R3D-18 · KTH · 6 CLASSES")


# ----------------------------------------------------------------------------
# Samples
# ----------------------------------------------------------------------------
def load_sample_image():
    paths = _existing([SAMPLE_IMG])
    if not paths:
        return None, render_empty()
    img = Image.open(paths[0]).convert("RGB")
    return img, classify_image(img)


def load_sample_detection():
    paths = _existing([SAMPLE_IMG])
    if not paths:
        return None, None, render_det_empty()
    img = Image.open(paths[0]).convert("RGB")
    out, res = detect_image(img, YOLO_CONF)
    return img, out, res


def load_sample_video():
    paths = _existing([SAMPLE_VIDEO])
    if not paths:
        return None, render_empty("motion")
    return paths[0], classify_video_clip(paths[0])


# ----------------------------------------------------------------------------
# Static interface fragments
# ----------------------------------------------------------------------------
PRELOADER = """
<div id="preloader">
  <div class="pre-top"><span>ActionScope® / Vision stack</span><span>Initialising · MMXXVI</span></div>
  <div class="pre-mid">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.1">
      <path d="M12 2v20M2 12h20M4.9 4.9l14.2 14.2M19.1 4.9 4.9 19.1"/>
    </svg>
  </div>
  <div class="pre-bottom">
    <div style="flex:1;max-width:480px">
      <div class="pre-count">000<small>%</small></div>
      <div class="pre-bar"><i></i></div>
    </div>
    <div class="pre-brand">HAR-15 · KTH<br>PyTorch / CUDA</div>
  </div>
</div>
"""

NAV = f"""
<div class="aurora"></div>
<div class="grid-bg"></div>
{PRELOADER}
<nav class="nav">
  <div class="nav-in">
    <a class="logo" href="#top">
      <span class="logo-mark">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4">
          <path d="M12 2v20M2 12h20M4.9 4.9l14.2 14.2M19.1 4.9 4.9 19.1"/>
        </svg>
      </span>
      <span><b>ACTIONSCOPE<sup>®</sup></b><small>Human action studio</small></span>
    </a>
    <div class="nav-links">
      <a href="#chapter-01" class="on">01 Still</a>
      <a href="#chapter-02">02 Detect</a>
      <a href="#chapter-03">03 Motion</a>
    </div>
    <div class="nav-status">
      <span class="live"><i class="pulse"></i><b>System online</b></span>
      <span class="clock">--:--:--</span>
    </div>
  </div>
  <div class="nav-progress"></div>
</nav>
"""

HERO = f"""
<section class="hero" id="top">
  <div class="eyebrow"><i></i>Computer vision studio — MMXXVI</div>
  <div class="hero-grid">
    <div class="hero-copy">
      <h1>
        <span class="row">Action</span>
        <span class="row">is the
          <svg class="spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2">
            <path d="M12 2v20M2 12h20M4.9 4.9l14.2 14.2M19.1 4.9 4.9 19.1"/>
          </svg>
        </span>
        <span class="row"><em>signal.</em></span>
      </h1>
      <p>Three vision systems, one studio. <b>Read stills. Find every body. Follow motion through time.</b>
      Built in PyTorch, tuned for the real world.</p>
      <div class="cta-row">
        <a class="btn acid" href="#chapter-01">Start analysing <span class="ar">↘</span></a>
        <a class="btn ghost" href="#how">How it works <span class="ar">→</span></a>
      </div>
    </div>
    <div class="orb">
      <div class="orb-glow"></div>
      <div class="orb-dashed"></div>
      <svg class="orb-text" viewBox="0 0 400 400">
        <defs><path id="cp" d="M200,200 m-168,0 a168,168 0 1,1 336,0 a168,168 0 1,1 -336,0"/></defs>
        <text><textPath href="#cp">SEE EVERY MOVE · READ THE SIGNAL · SEE EVERY MOVE · READ THE SIGNAL ·</textPath></text>
      </svg>
      <div class="orb-ring"></div>
      <div class="orb-scan"></div>
      <div class="orb-core">
        <div class="orb-num">86.9<small>%</small></div>
        <div class="orb-cap">Image top-1</div>
      </div>
      <div class="chip c1"><i></i>YOLOv8n</div>
      <div class="chip c2"><i></i>B0 + B3 ensemble</div>
      <div class="chip c3"><i></i>R3D-18</div>
      <div class="chip c4"><i></i>CUDA live</div>
    </div>
  </div>
  <div class="hero-base">
    <div class="mono"><b>15</b> everyday actions &nbsp;/&nbsp; <b>03</b> model heads &nbsp;/&nbsp; <b>01</b> GPU</div>
    <div class="scroll-cue"><i></i>Scroll to explore</div>
    <div class="mono">KTH · HAR-15 &nbsp;/&nbsp; PyTorch + Gradio</div>
  </div>
</section>
"""

STATS = """
<section class="stats">
  <div class="stat"><i>[01]</i><b data-count="15" data-decimals="0">15</b><span>Everyday actions</span></div>
  <div class="stat"><i>[02]</i><b data-count="86.9" data-decimals="1" data-suffix="%">86.9%</b><span>Image top-1 · ensemble</span></div>
  <div class="stat"><i>[03]</i><b data-count="91.6" data-decimals="1" data-suffix="%">91.6%</b><span>Video validation · R3D-18</span></div>
  <div class="stat"><i>[04]</i><b data-count="16" data-decimals="0" data-suffix="f">16f</b><span>Frames per motion clip</span></div>
</section>
"""

_TICK_A = "CALLING ✳ CLAPPING ✳ CYCLING ✳ DANCING ✳ DRINKING ✳ EATING ✳ FIGHTING ✳ HUGGING ✳ RUNNING ✳ SLEEPING ✳ "
_TICK_B = "STILL — DETECT EVERY BODY — FOLLOW THE MOVEMENT — STILL — DETECT EVERY BODY — FOLLOW THE MOVEMENT — "
TICKER = f"""
<div class="ticker-wrap">
  <div class="band"><div class="marquee"><span>{_TICK_A}{_TICK_A}</span></div></div>
  <div class="band alt"><div class="marquee"><span>{'<b>' + _TICK_B.replace(' — ', '</b> <i>—</i> <b>') + '</b>'}</span></div></div>
</div>
"""


def chapter(index, kicker, title, copy, meta):
    meta_html = "".join(
        f"<div><span>{_html.escape(k)}</span><b>{_html.escape(v)}</b></div>" for k, v in meta
    )
    return f"""
<section class="chapter" id="chapter-{index}">
  <div class="ch-num">{index}</div>
  <div class="ch-main">
    <div class="ch-kick">{kicker}</div>
    <h2>{title}</h2>
    <p>{copy}</p>
  </div>
  <div class="ch-meta">{meta_html}</div>
</section>
"""


CH01 = chapter(
    "01", "Chapter 01 — Still image", "Read the <em>moment.</em>",
    "Two EfficientNet heads read a single frame and place it across fifteen everyday "
    "actions — from cycling to sleeping. Drop a photo; the ensemble answers in milliseconds.",
    [("Classes", "15"), ("Backbone", "B0 + B3"), ("Input", "300 px"), ("Top-1", "86.9 %")],
)
CH02 = chapter(
    "02", "Chapter 02 — Detection", "Find every <em>body.</em>",
    "YOLOv8n locates each person, then the action head labels every crop independently. "
    "Crowds, couples, passing strangers — everyone gets their own read.",
    [("Detector", "YOLOv8n"), ("Mode", "Per person"), ("Threshold", "Tunable")],
)
CH03 = chapter(
    "03", "Chapter 03 — Motion", "Follow the <em>movement.</em>",
    "R3D-18 samples sixteen frames and reads the spacetime between them. Six KTH actions, "
    "one clip at a time. Short, single-action clips work best.",
    [("Model", "R3D-18"), ("Frames", "16"), ("Classes", "6"), ("Val", "91.6 %")],
)

STUDIO_HEAD_01 = f"""
<div class="studio-head">
  <div class="l"><i></i>Studio / image classifier</div>
  <div class="r"><span>GPU · {DEVICE.upper()}</span><span>{status_ens}</span><span>Ready</span></div>
</div>
"""
STUDIO_HEAD_02 = f"""
<div class="studio-head">
  <div class="l"><i></i>Studio / person detection</div>
  <div class="r"><span>GPU · {DEVICE.upper()}</span><span>YOLOv8n</span><span>{status_yolo}</span></div>
</div>
"""
STUDIO_HEAD_03 = f"""
<div class="studio-head">
  <div class="l"><i></i>Studio / motion classifier</div>
  <div class="r"><span>GPU · {DEVICE.upper()}</span><span>R3D-18</span><span>{status_3d}</span></div>
</div>
"""

PANE_SOURCE = '<div class="pane-k"><span>Source frame</span><span>JPG · PNG · WEBP</span></div>'
PANE_OUTPUT = '<div class="pane-k"><span>Output</span><span>Top-5 probabilities</span></div>'
PANE_DET_IN = '<div class="pane-k"><span>Source frame</span><span>YOLO + action head</span></div>'
PANE_DET_OUT = '<div class="pane-k"><span>Detections</span><span>Per-person labels</span></div>'
PANE_VIDEO = '<div class="pane-k"><span>Source clip</span><span>AVI · MP4 · MOV</span></div>'
PANE_MOTION = '<div class="pane-k"><span>Output</span><span>Top-5 KTH actions</span></div>'
AUTO_NOTE = '<div class="auto-note">Auto-runs on upload · drop, paste, or snap</div>'
DET_LOG = """
<div class="det-log">
  <div class="dl-head"><span>Detection log</span><span><i></i>Live</span></div>
  <div class="dl-wave"></div>
  <div class="dl-rows">
    <div><span>Detector</span><b>YOLOv8n</b></div>
    <div><span>Classifier</span><b>B0 + B3</b></div>
    <div><span>Output</span><b>Per-person labels</b></div>
  </div>
</div>
"""

HOW = f"""
<section class="how" id="how">
  <div class="how-title">
    <div class="how-kick">Field notes — the machinery</div>
    <h3>How it <em>works.</em></h3>
    <p style="margin-top:20px;color:var(--muted);font-size:13.5px;line-height:1.7;max-width:240px">
      Three heads, one pipeline. Every prediction below is measured, not projected.
    </p>
  </div>
  <div class="how-grid">
    <div class="how-item">
      <div class="n">01 / Still</div>
      <b>EfficientNet B0 + B3</b>
      <p>A 300px frame passes through two ImageNet-pretrained backbones fine-tuned on
      <b>10,710 HAR-15 images</b>. Softmax probabilities are averaged into one verdict over
      fifteen everyday actions.</p>
    </div>
    <div class="how-item">
      <div class="n">02 / Detect</div>
      <b>YOLOv8n + action head</b>
      <p>The detector finds every person box; each crop is classified independently by the
      2D head. Lower the confidence slider to catch smaller bodies, raise it to suppress
      false boxes.</p>
    </div>
    <div class="how-item">
      <div class="n">03 / Motion</div>
      <b>R3D-18 spatiotemporal</b>
      <p>Sixteen uniformly sampled RGB frames become a <b>(C, T, H, W)</b> tensor. The 3D
      CNN reads motion across time and scores six KTH actions — boxing to walking.</p>
    </div>
    <div class="spec-strip">
      <div><b>Image data</b>HAR-15 · 10,710 train / 1,890 test</div>
      <div><b>Video data</b>KTH · 600 clips · 25 fps · 6 classes</div>
      <div><b>Runtime</b>PyTorch 2.11 · CUDA 12.8 · Gradio 6</div>
      <div><b>Reported</b>86.9 % image · 91.6 % video</div>
    </div>
  </div>
</section>
"""

FOOTER = f"""
<div class="site-footer">
  <div class="foot-word">ActionScope</div>
  <div class="foot-grid">
    <div><b>System</b>B0 + B3 ensemble<br>YOLOv8n detector<br>R3D-18 motion</div>
    <div><b>Data</b>HAR-15 · 10,710 train<br>KTH · 600 clips<br>15 + 6 classes</div>
    <div><b>Stack</b>PyTorch 2.11<br>Gradio 6<br>CUDA 12.8</div>
    <div><b>Colophon</b>Set in Bricolage Grotesque<br>&amp; Instrument Serif<br>© MMXXVI</div>
  </div>
  <div class="foot-end">
    <span>ActionScope® — human action detection</span>
    <span>Built for the frame · RTX 5060</span>
  </div>
</div>
"""


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
theme = gr.themes.Base(primary_hue="lime", neutral_hue="neutral")

with gr.Blocks(title="ActionScope — Human Action Detection", analytics_enabled=False) as demo:
    gr.HTML(NAV, elem_classes=["bare"], apply_default_css=False)
    gr.HTML(HERO, elem_classes=["bare"], apply_default_css=False)
    gr.HTML(STATS, elem_classes=["bare"], apply_default_css=False)
    gr.HTML(TICKER, elem_classes=["bare"], apply_default_css=False)

    # ---- Chapter 01: still image -------------------------------------------
    gr.HTML(CH01, elem_classes=["bare"], apply_default_css=False)
    with gr.Column(elem_classes=["studio"]):
        gr.HTML(STUDIO_HEAD_01, elem_classes=["bare"], apply_default_css=False)
        with gr.Row(elem_classes=["studio-grid"]):
            with gr.Column(elem_classes=["pane"]):
                gr.HTML(PANE_SOURCE, elem_classes=["bare"], apply_default_css=False)
                inp = gr.Image(type="pil", sources=["upload", "webcam", "clipboard"],
                               container=False, height=342, elem_classes=["dropzone"])
                btn_img = gr.Button("RUN ANALYSIS  ↗", variant="primary", elem_classes=["run"])
                gr.HTML(AUTO_NOTE, elem_classes=["bare"], apply_default_css=False)
                sample_img = gr.Button("USE A SAMPLE FRAME  ↗", elem_classes=["sample-btn"],
                                       visible=bool(_existing([SAMPLE_IMG])))
            with gr.Column(elem_classes=["pane"]):
                gr.HTML(PANE_OUTPUT, elem_classes=["bare"], apply_default_css=False)
                out = gr.HTML(render_empty(), elem_classes=["verdict-slot"],
                              apply_default_css=False)
        btn_img.click(classify_image, inp, out)
        inp.change(classify_image, inp, out)
        sample_img.click(load_sample_image, None, [inp, out])

    # ---- Chapter 02: detection ---------------------------------------------
    gr.HTML(CH02, elem_classes=["bare"], apply_default_css=False)
    with gr.Column(elem_classes=["studio"]):
        gr.HTML(STUDIO_HEAD_02, elem_classes=["bare"], apply_default_css=False)
        with gr.Row(elem_classes=["studio-grid"]):
            with gr.Column(elem_classes=["pane"]):
                gr.HTML(PANE_DET_IN, elem_classes=["bare"], apply_default_css=False)
                inp2 = gr.Image(type="pil", sources=["upload", "webcam", "clipboard"],
                                container=False, height=300, elem_classes=["dropzone"])
                conf_slider = gr.Slider(0.1, 0.9, value=YOLO_CONF, step=0.05,
                                        label="Detection confidence",
                                        info="Lower finds more · higher is stricter",
                                        show_label=True, elem_classes=["conf-slider"])
                btn_det = gr.Button("DETECT PEOPLE  ↗", variant="primary", elem_classes=["run"])
                sample_det = gr.Button("USE A SAMPLE FRAME  ↗", elem_classes=["sample-btn"],
                                       visible=bool(_existing([SAMPLE_IMG])))
                gr.HTML(DET_LOG, elem_classes=["bare"], apply_default_css=False)
            with gr.Column(elem_classes=["pane"]):
                gr.HTML(PANE_DET_OUT, elem_classes=["bare"], apply_default_css=False)
                out2 = gr.Image(value=make_placeholder(), interactive=False, show_label=False,
                                container=False, height=356, elem_classes=["det-out"])
                det_html = gr.HTML(render_det_empty(), elem_classes=["verdict-slot"],
                                   apply_default_css=False)
        btn_det.click(detect_image, [inp2, conf_slider], [out2, det_html])
        inp2.change(detect_image, [inp2, conf_slider], [out2, det_html])
        conf_slider.change(detect_image, [inp2, conf_slider], [out2, det_html])
        sample_det.click(load_sample_detection, None, [inp2, out2, det_html])

    # ---- Chapter 03: motion ------------------------------------------------
    gr.HTML(CH03, elem_classes=["bare"], apply_default_css=False)
    with gr.Column(elem_classes=["studio", "motion"]):
        gr.HTML(STUDIO_HEAD_03, elem_classes=["bare"], apply_default_css=False)
        with gr.Row(elem_classes=["studio-grid"]):
            with gr.Column(elem_classes=["pane"]):
                gr.HTML(PANE_VIDEO, elem_classes=["bare"], apply_default_css=False)
                inp3 = gr.Video(sources=["upload"], container=False,
                                height=330, elem_classes=["videozone"])
                btn_vid = gr.Button("READ THE MOTION  ↗", variant="primary", elem_classes=["run"])
                gr.HTML(AUTO_NOTE, elem_classes=["bare"], apply_default_css=False)
                sample_vid = gr.Button("USE A SAMPLE CLIP  ↗", elem_classes=["sample-btn"],
                                       visible=bool(_existing([SAMPLE_VIDEO])) and not SKIP_3D)
            with gr.Column(elem_classes=["pane"]):
                gr.HTML(PANE_MOTION, elem_classes=["bare"], apply_default_css=False)
                out3 = gr.HTML(render_empty("motion"), elem_classes=["verdict-slot"],
                               apply_default_css=False)
        btn_vid.click(classify_video_clip, inp3, out3)
        inp3.change(classify_video_clip, inp3, out3)
        sample_vid.click(load_sample_video, None, [inp3, out3])

    gr.HTML(HOW, elem_classes=["bare"], apply_default_css=False)
    gr.HTML(FOOTER, elem_classes=["bare"], apply_default_css=False)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true", help="public gradio.live link")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    demo.queue(default_concurrency_limit=4)
    # Warm optional models in the background: the UI is already interactive.
    threading.Thread(target=warmup, name="model-warmup", daemon=True).start()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share,
                theme=theme, css=CSS, js=JS, head=HEAD, footer_links=[])

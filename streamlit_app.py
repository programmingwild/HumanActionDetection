"""ActionScope — Streamlit Community Cloud edition of "Action is the signal."

Torch-free: onnxruntime + numpy + PIL + opencv-headless only.
Run locally:  streamlit run streamlit_app.py
Deploy: Streamlit Cloud -> repo -> streamlit_app.py (auto-redeploy on push to main).
"""
import base64
import hashlib
import html as _html
import io
import math
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from src.labels import pretty
from src.ort_backend import CLASSES_2D, CLASSES_3D, OrtEnsemble, OrtVideo, ckpt_dir, yolo_person_boxes

st.set_page_config(page_title="ActionScope — Human Action Detection", layout="wide",
                   initial_sidebar_state="collapsed")

HERE = Path(__file__).resolve().parent
CSS = (HERE / "src" / "ui" / "streamlit.css").read_text(encoding="utf-8")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
st.html("<div class='aurora'></div><div class='grid-bg'></div>")


# ----------------------------------------------------------------------------
# Model loaders — lazy so the page paints before the 126 MB video model loads.
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_ensemble():
    d = ckpt_dir()
    return OrtEnsemble(d / "effb0.onnx", d / "effb3.onnx")


@st.cache_resource(show_spinner=False)
def get_yolo():
    d = ckpt_dir()
    return ort.InferenceSession(str(d / "yolov8n.onnx"), providers=["CPUExecutionProvider"])


@st.cache_resource(show_spinner=False)
def get_video():
    d = ckpt_dir()
    return OrtVideo(d / "r3d18.onnx")


MODEL_FOOT = "B0 + B3 ensemble · 2 models (ONNX)"

# ----------------------------------------------------------------------------
# HTML rendering
# ----------------------------------------------------------------------------
def data_uri(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def frame_html(img, annotated=False):
    src = data_uri(img)
    return (f'<div class="frame{" det" if annotated else ""}">'
            f'<div class="frame-blur" style="background-image:url(&quot;{src}&quot;)"></div>'
            f'<img src="{src}" alt="frame"></div>')


def frame_placeholder(label="Awaiting a frame", sub="drop a photo · jpg / png"):
    return (f'<div class="frame"><div class="empty-glyph">✳</div>'
            f'<div style="text-align:center;position:relative;z-index:2">'
            f'<div class="empty-k">{label}</div><div class="empty-p">{sub}</div></div></div>')


def render_empty(kind="still"):
    motion = kind == "motion"
    cls = " motion" if motion else ""
    title = "Awaiting a clip" if motion else "Awaiting a frame"
    copy = ("Drop a short clip. Sixteen frames are sampled across time and read by R3D-18."
            if motion else "Drop a photo. The B0 + B3 ensemble answers in milliseconds.")
    return f'''<div class="verdict empty{cls}">
      <div class="empty-glyph">✳</div><div class="empty-k">{title}</div>
      <p class="empty-p">{copy}</p><div class="empty-line"></div></div>'''


def render_verdict(probs, kind="still", latency=None, foot=None):
    items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    if not items:
        return render_empty(kind)
    top, conf = items[0]
    circ = 2 * math.pi * 56
    offset = circ * (1 - min(max(conf, 0.0), 1.0))
    bars = "".join(
        f'<div class="bar"><span class="bt">{_html.escape(n)}</span>'
        f'<div class="t"><i style="width:{p * 100:.2f}%"></i></div>'
        f'<span class="p">{p * 100:.1f}</span></div>'
        for n, p in items[:5]
    )
    ms = f"{latency * 1000:.0f} ms" if latency is not None else "—"
    cls = " motion" if kind == "motion" else ""
    kicker = "LIVE INFERENCE · 3D" if kind == "motion" else "LIVE INFERENCE"
    tag = "R3D-18 · 16 FRAMES" if kind == "motion" else "B0 + B3 · 224 / 300 PX"
    note = "TOP-1 OF 6 KTH ACTIONS" if kind == "motion" else "TOP-1 OF 15 CLASSES"
    foot = foot or ("R3D-18 · KTH · 6 CLASSES" if kind == "motion" else MODEL_FOOT)
    return f'''<div class="verdict{cls}">
      <div class="v-head"><span class="live"><i></i>{kicker}</span><span>{tag}</span></div>
      <div class="v-main">
        <div><div class="v-label">Predicted action</div>
          <div class="v-action">{_html.escape(top)}</div>
          <div class="v-note">{note} · {ms}</div></div>
        <div class="ring"><svg viewBox="0 0 120 120">
          <circle class="bg" cx="60" cy="60" r="56"></circle>
          <circle class="fg" cx="60" cy="60" r="56" stroke-dasharray="{circ:.1f}" stroke-dashoffset="{offset:.1f}"></circle>
        </svg><div class="in">{conf * 100:.1f}<small>%</small></div></div>
      </div><div class="v-bars">{bars}</div>
      <div class="v-foot"><span>{_html.escape(foot)}</span><span>Inference {ms}</span></div>
    </div>'''


def render_det_empty():
    return '''<div class="det-result"><div class="det-head">
      <div><div class="k">Scene read</div><b>0 <small>people</small></b></div>
      <span class="pill">Awaiting frame</span></div>
      <div class="det-note">Drop an image to localise every person and classify each crop independently.</div>
      <div class="v-foot"><span>YOLOv8n · B0 + B3</span><span>—</span></div></div>'''


def render_detection(rows, conf, latency, fallback=None):
    if not rows and fallback:
        items = sorted(fallback.items(), key=lambda kv: kv[1], reverse=True)
        top, p = items[0]
        bars = "".join(
            f'<div class="bar"><span class="bt">{_html.escape(n)}</span>'
            f'<div class="t"><i style="width:{v * 100:.2f}%"></i></div>'
            f'<span class="p">{v * 100:.1f}</span></div>' for n, v in items[:4])
        return f'''<div class="det-result"><div class="det-head">
          <div><div class="k">No person detected</div><b>{_html.escape(top)} <small>full frame</small></b></div>
          <span class="pill">Fallback read</span></div>
          <div class="det-list" style="padding-top:14px">{bars}</div>
          <div class="det-note">Nobody crossed the {conf:.2f} threshold, so the whole frame was classified. Lower it to catch smaller bodies.</div>
          <div class="v-foot"><span>YOLOv8n · threshold {conf:.2f}</span><span>{latency * 1000:.0f} ms</span></div></div>'''
    rows_html = "".join(
        f'<div class="det-row"><span class="det-i">P{r["i"]}</span>'
        f'<span>{_html.escape(r["action"])}</span>'
        f'<div class="det-m"><i style="width:{r["prob"] * 100:.1f}%"></i></div>'
        f'<span class="det-p">{r["prob"] * 100:.1f}%</span></div>' for r in rows)
    n = len(rows)
    return f'''<div class="det-result"><div class="det-head">
      <div><div class="k">Scene read</div><b>{n:02d} <small>{"person" if n == 1 else "people"}</small></b></div>
      <span class="pill">Conf {conf:.2f}</span></div>
      <div class="det-list">{rows_html}</div>
      <div class="det-note">Each bounding box is cropped and classified on its own by the 2D action head.</div>
      <div class="v-foot"><span>YOLOv8n · B0 + B3</span><span>{latency * 1000:.0f} ms</span></div></div>'''


def _font(size):
    win = Path("C:\\Windows")
    for p in (win / "Fonts" / "arialbd.ttf",
              Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
              Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")):
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_detections(frame_bgr, rows):
    h, w = frame_bgr.shape[:2]
    base = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    over = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    acid = (215, 255, 63, 255)
    plate = (8, 8, 7, 235)
    lw = max(2, int(w * 0.0022))
    font = _font(max(13, int(w * 0.0135)))
    L = max(14, int(w * 0.016))
    placed = []
    for r in rows:
        x1, y1, x2, y2 = r["box"]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        for px, py, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            d.line([(px, py), (px + dx * L, py)], fill=acid, width=lw)
            d.line([(px, py), (px, py + dy * L)], fill=acid, width=lw)
        txt = f'P{r["i"]} · {r["action"].upper()} {r["prob"] * 100:.0f}%'
        tb = d.textbbox((0, 0), txt, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pdx, pdy = 9, 6
        lh = th + pdy * 2
        lx = min(max(6, x1), w - tw - pdx * 2 - 6)
        ly = y1 - lh - 5
        if ly < 6:
            ly = min(y2 + 5, h - lh - 6)
        for _ in range(14):
            rect = (lx, ly, lx + tw + pdx * 2, ly + lh)
            if not any(not (rect[2] < p[0] or rect[0] > p[2] or rect[3] < p[1] or rect[1] > p[3])
                       for p in placed):
                break
            ly += lh + 4
            if ly + lh > h - 6:
                ly = 6
        placed.append((lx, ly, lx + tw + pdx * 2, ly + lh))
        d.rounded_rectangle([lx, ly, lx + tw + pdx * 2, ly + lh], radius=6, fill=plate, outline=acid, width=1)
        d.text((lx + pdx, ly + pdy - tb[1]), txt, font=font, fill=acid)
    return Image.alpha_composite(base, over).convert("RGB")


def file_key(upload, extra=""):
    if upload is None:
        return None
    return hashlib.md5(upload.getvalue()).hexdigest() + extra


def save_upload(upload):
    suffix = Path(upload.name).suffix or ".bin"
    tmp = __import__("tempfile").NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(upload.getvalue())
    tmp.close()
    return tmp.name


def read_frames(path, clip_len=16):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    frames = []
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    for i in np.linspace(0, max(0, count - 1), clip_len).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise RuntimeError("No frames")
    return frames


# ----------------------------------------------------------------------------
# Static interface
# ----------------------------------------------------------------------------
NAV = '''<div class="nav">
  <div class="logo"><span class="logo-mark"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M12 2v20M2 12h20M4.9 4.9l14.2 14.2M19.1 4.9 4.9 19.1"/></svg></span>
    <span><b>ACTIONSCOPE<sup>®</sup></b><small>Human action studio</small></span></div>
  <div class="nav-links"><span class="on">01 Still</span><span>02 Detect</span><span>03 Motion</span></div>
  <div class="nav-status"><span class="live"><i class="pulse"></i>System online</span><span>ONNX · CPU</span></div>
</div>'''

HERO = '''<section class="hero"><div class="eyebrow"><i></i>Computer vision studio — MMXXVI</div>
<div class="hero-grid"><div>
  <h1><span class="row">Action</span><span class="row">is the <svg class="spin-mark" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2"><path d="M12 2v20M2 12h20M4.9 4.9l14.2 14.2M19.1 4.9 4.9 19.1"/></svg></span><span class="row"><em>signal.</em></span></h1>
  <p>Three vision systems, one studio. <b>Read stills. Find every body. Follow motion through time.</b> Torch-free on ONNX Runtime, tuned for the free tier.</p>
  <div class="cta-row"><span class="btn acid">Start analysing &nbsp;↘</span><span class="btn">ONNX · CPU · live</span></div>
</div><div class="orb"><div class="orb-glow"></div><div class="orb-dashed"></div><div class="orb-ring"></div>
  <div class="orb-core"><div class="orb-num">86.0<small>%</small></div><div class="orb-cap">Image top-1</div></div>
  <div class="chip c1"><i></i>YOLOv8n</div><div class="chip c2"><i></i>B0 + B3</div>
  <div class="chip c3"><i></i>R3D-18</div><div class="chip c4"><i></i>ONNX live</div>
</div></div>
<div class="hero-base"><span><b>15</b> everyday actions &nbsp;/&nbsp; <b>03</b> model heads</span>
<span>KTH · HAR-15 &nbsp;/&nbsp; Streamlit Cloud</span></div></section>'''

STATS = '''<section class="stats">
<div class="stat"><i>[01]</i><b>15</b><span>Everyday actions</span></div>
<div class="stat"><i>[02]</i><b>86.0<small>%</small></b><span>Image top-1 · ensemble</span></div>
<div class="stat"><i>[03]</i><b>91.6<small>%</small></b><span>Video validation · R3D-18</span></div>
<div class="stat"><i>[04]</i><b>16<small>f</small></b><span>Frames per motion clip</span></div></section>'''

_TICK_A = "CALLING ✳ CLAPPING ✳ CYCLING ✳ DANCING ✳ DRINKING ✳ EATING ✳ FIGHTING ✳ HUGGING ✳ RUNNING ✳ SLEEPING ✳ "
_TICK_B = "STILL — DETECT EVERY BODY — FOLLOW THE MOVEMENT — STILL — DETECT EVERY BODY — FOLLOW THE MOVEMENT — "
TICKER = f'''<div class="ticker-wrap"><div class="band"><div class="marquee"><span>{_TICK_A * 2}</span></div></div>
<div class="band alt"><div class="marquee"><span>{_TICK_B * 2}</span></div></div></div>'''


def chapter(n, kicker, title, copy, meta):
    rows = "".join(f"<div><span>{_html.escape(k)}</span><b>{_html.escape(v)}</b></div>" for k, v in meta)
    return f'''<section class="chapter" id="chapter-0{n}"><div class="ch-num">0{n}</div>
      <div class="ch-main"><div class="ch-kick">{kicker}</div><h2>{title}</h2><p>{copy}</p></div>
      <div class="ch-meta">{rows}</div></section>'''


CH1 = chapter(1, "Chapter 01 — Still image", "Read the <em>moment.</em>",
              "Two EfficientNet heads read a single frame and place it across fifteen everyday actions "
              "— from cycling to sleeping. Drop a photo; the ensemble answers in milliseconds.",
              [("Classes", "15"), ("Backbone", "B0 + B3"), ("Input", "224 / 300 px"), ("Top-1", "86.0 %")])
CH2 = chapter(2, "Chapter 02 — Detection", "Find every <em>body.</em>",
              "YOLOv8n locates each person, then the action head labels every crop independently. "
              "Crowds, couples, passing strangers — everyone gets their own read.",
              [("Detector", "YOLOv8n"), ("Mode", "Per person"), ("Threshold", "Tunable")])
CH3 = chapter(3, "Chapter 03 — Motion", "Follow the <em>movement.</em>",
              "R3D-18 samples sixteen frames and reads the spacetime between them. Six KTH actions, "
              "one clip at a time. Short, single-action clips work best.",
              [("Model", "R3D-18"), ("Frames", "16"), ("Classes", "6"), ("Val", "91.6 %")])

HOW = '''<section class="how"><div class="how-title"><div class="how-kick">Field notes — the machinery</div>
<h3>How it <em>works.</em></h3><p>Three heads, one pipeline. Every prediction below is measured, not projected.</p></div>
<div class="how-grid">
  <div class="how-item"><div class="n">01 / Still</div><b>EfficientNet B0 + B3</b>
    <p>A frame passes through two ImageNet-pretrained backbones fine-tuned on <b>10,710 HAR-15 images</b>. Their softmax probabilities are averaged into one verdict.</p></div>
  <div class="how-item"><div class="n">02 / Detect</div><b>YOLOv8n + action head</b>
    <p>The detector finds every person box; each crop is classified independently by the 2D head. Lower the confidence slider to catch smaller bodies.</p></div>
  <div class="how-item"><div class="n">03 / Motion</div><b>R3D-18 spatiotemporal</b>
    <p>Sixteen uniformly sampled frames become a <b>(C, T, H, W)</b> tensor. The 3D CNN reads motion across time and scores six KTH actions.</p></div>
  <div class="spec-strip">
    <div><b>Image data</b>HAR-15 · 10,710 train / 1,890 test</div>
    <div><b>Video data</b>KTH · 600 clips · 6 classes</div>
    <div><b>Runtime</b>ONNX Runtime · Streamlit Cloud · CPU</div>
    <div><b>Reported</b>86.0 % image · 91.6 % video</div></div>
</div></section>'''

FOOTER = '''<div class="site-footer"><div class="foot-word">ActionScope</div>
<div class="foot-grid"><div><b>System</b>B0 + B3 ensemble<br>YOLOv8n detector<br>R3D-18 motion</div>
<div><b>Data</b>HAR-15 · 10,710 train<br>KTH · 600 clips<br>15 + 6 classes</div>
<div><b>Stack</b>ONNX Runtime<br>Streamlit Cloud<br>CPU optimized</div>
<div><b>Colophon</b>Bricolage Grotesque<br>&amp; Instrument Serif<br>© MMXXVI</div></div>
<div class="foot-end"><span>ActionScope® — human action detection</span><span>Built for the frame</span></div></div>'''

PANE_SRC = '<div class="pane-k"><span>Source frame</span><span>JPG · PNG</span></div>'
PANE_OUT = '<div class="pane-k"><span>Output</span><span>Top-5 probabilities</span></div>'
PANE_DET_IN = '<div class="pane-k"><span>Source frame</span><span>YOLO + action head</span></div>'
PANE_DET_OUT = '<div class="pane-k"><span>Detections</span><span>Per-person labels</span></div>'
PANE_VID = '<div class="pane-k"><span>Source clip</span><span>MP4 · AVI · MOV</span></div>'
PANE_MOTION = '<div class="pane-k"><span>Output</span><span>Top-5 KTH actions</span></div>'
DET_LOG = '''<div class="det-log"><div class="dl-head"><span>Detection log</span><span><i></i>Live</span></div>
<div class="dl-wave"></div><div class="dl-rows"><div><span>Detector</span><b>YOLOv8n ONNX</b></div>
<div><span>Classifier</span><b>B0 + B3</b></div><div><span>Output</span><b>Per-person labels</b></div></div></div>'''
AUTO_NOTE = '<div class="auto-note">Auto-runs on upload · inference on CPU</div>'

st.html(NAV + HERO + STATS + TICKER + CH1)

# ----------------------------------------------------------------------------
# Chapter 01 — still image
# ----------------------------------------------------------------------------
st.html('<div class="studio-head"><div class="l"><i></i>Studio / image classifier</div>'
        '<div class="r"><span>ONNX · CPU</span><span>B0+B3 ensemble</span><span>Ready</span></div></div>')
c1, c2 = st.columns([1, 1.06], gap="small")
with c1:
    st.html(PANE_SRC)
    up1 = st.file_uploader("Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="img1")
    if up1:
        st.html(frame_html(Image.open(up1)))
    else:
        st.html(frame_placeholder())
    st.html(AUTO_NOTE)
with c2:
    st.html(PANE_OUT)
    slot1 = st.empty()
    slot1.html(render_empty())

if up1:
    key = file_key(up1)
    if st.session_state.get("img1_key") != key:
        t0 = time.perf_counter()
        with st.spinner("Scoring ensemble…"):
            _, _, probs = get_ensemble().top(Image.open(up1).convert("RGB"))
        st.session_state.img1_key = key
        st.session_state.img1_html = render_verdict(probs, latency=time.perf_counter() - t0)
    slot1.html(st.session_state.img1_html)

# ----------------------------------------------------------------------------
# Chapter 02 — person detection
# ----------------------------------------------------------------------------
st.html(CH2)
st.html('<div class="studio-head"><div class="l"><i></i>Studio / person detection</div>'
        '<div class="r"><span>ONNX · CPU</span><span>YOLOv8n</span><span>Ready</span></div></div>')
c3, c4 = st.columns([1, 1.06], gap="small")
with c3:
    st.html(PANE_DET_IN)
    up2 = st.file_uploader("Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="img2")
    if up2:
        st.html(frame_html(Image.open(up2)))
    else:
        st.html(frame_placeholder())
    thr = st.slider("Detection confidence", 0.1, 0.9, 0.4, 0.05,
                    help="Lower finds more people · higher is stricter")
    st.html(DET_LOG)
with c4:
    st.html(PANE_DET_OUT)
    slot2img = st.empty()
    slot2img.html(frame_placeholder("Awaiting a frame", "annotated output"))
    slot2 = st.empty()
    slot2.html(render_det_empty())

if up2:
    key = file_key(up2, f":{thr:.2f}")
    if st.session_state.get("img2_key") != key:
        t0 = time.perf_counter()
        with st.spinner("Detecting people…"):
            img = Image.open(up2).convert("RGB")
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            boxes = yolo_person_boxes(get_yolo(), frame, thr)
            rows, fallback = [], None
            ens = get_ensemble()
            for i, (x1, y1, x2, y2, _dc) in enumerate(boxes, 1):
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
                crop = Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
                a, p, _ = ens.top(crop)
                rows.append({"i": i, "action": a, "prob": p, "box": (x1, y1, x2, y2)})
            if not rows:
                _, _, fallback = ens.top(img)
                annotated = img
            else:
                annotated = draw_detections(frame, rows)
            latency = time.perf_counter() - t0
        st.session_state.img2_key = key
        st.session_state.img2_img = frame_html(annotated, annotated=True) if rows else frame_html(annotated)
        st.session_state.img2_html = render_detection(rows, thr, latency, fallback)
    slot2img.html(st.session_state.img2_img)
    slot2.html(st.session_state.img2_html)

# ----------------------------------------------------------------------------
# Chapter 03 — video motion
# ----------------------------------------------------------------------------
st.html(CH3)
st.html('<div class="studio-head"><div class="l"><i></i>Studio / motion classifier</div>'
        '<div class="r"><span>ONNX · CPU</span><span>R3D-18</span><span>Ready</span></div></div>')
c5, c6 = st.columns([1, 1.06], gap="small")
with c5:
    st.html(PANE_VID)
    up3 = st.file_uploader("Clip", type=["mp4", "avi", "mov"], label_visibility="collapsed", key="vid1")
    if up3:
        try:
            preview = read_frames(save_upload(up3), 1)[0]
            st.html(frame_html(preview))
        except Exception:
            st.html(frame_placeholder("Clip loaded", "preview unavailable"))
    else:
        st.html(frame_placeholder("Awaiting a clip", "short single-action clips work best"))
    st.html(AUTO_NOTE)
with c6:
    st.html(PANE_MOTION)
    slot3 = st.empty()
    slot3.html(render_empty("motion"))

if up3:
    key = file_key(up3)
    if st.session_state.get("vid1_key") != key:
        t0 = time.perf_counter()
        try:
            with st.spinner("Reading 16 frames…"):
                probs = get_video().probs(read_frames(save_upload(up3)))
            names = {pretty(c): float(p) for c, p in zip(CLASSES_3D, probs)}
            st.session_state.vid1_html = render_verdict(names, kind="motion", latency=time.perf_counter() - t0)
        except Exception:
            st.session_state.vid1_html = ('<div class="verdict empty motion"><div class="empty-k">Clip unreadable</div>'
                                          '<p class="empty-p">Try a short MP4 with one clear action.</p></div>')
        st.session_state.vid1_key = key
    slot3.html(st.session_state.vid1_html)

st.html(HOW + FOOTER)

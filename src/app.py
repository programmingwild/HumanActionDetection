"""Human Action Detection — premium dark Gradio experience.

Tabs: image classification (15 actions) · YOLO person detection · 3D video (KTH, 6 actions).

Usage:
  python -m src.app
  python -m src.app --share --port 7860
  SKIP_3D=1 python -m src.app   # lite mode for CPU-only hosting
"""
import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image

from .config import CHECKPOINT_PATH, DEVICE, VIDEO_CHECKPOINT_PATH, YOLO_CONF
from .ensemble import ensemble_probs, load_ensemble
from .infer import checkpoint_img_size, get_tf, load_model
from .labels import pretty
from .inference_video_3d import get_tfm as get_3d_tfm, load_video_model
from .model_3d import get_video_model  # noqa: F401
from .yolo_action import annotate_frame, classify_crop, get_yolo, person_boxes

print("Loading 2D action model...")
model2d, classes2d = load_model(str(CHECKPOINT_PATH))
tf2d = get_tf(checkpoint_img_size(str(CHECKPOINT_PATH)))

print("Loading ensemble partner (B0 @224)...")
try:
    ens_members, _ = load_ensemble(["checkpoints/best_effb0.pth", str(CHECKPOINT_PATH)])
    status_ens = "B0+B3 ensemble"
except Exception as e:
    print(f"Ensemble unavailable: {e}")
    ens_members, status_ens = None, "single model"
status_2d = "Ready"

print("Loading YOLO...")
try:
    yolo = get_yolo()
    status_yolo = "Ready"
except Exception as e:
    print(f"YOLO unavailable: {e}")
    yolo = None
    status_yolo = "Unavailable"

print("Loading 3D video model...")
import os
if os.environ.get("SKIP_3D") == "1":
    print("SKIP_3D=1 — skipping 3D model (lite/CPU deployment).")
    model3d, classes3d, tf3d = None, None, None
    status_3d = "Disabled (lite)"
else:
    try:
        model3d, classes3d = load_video_model(str(VIDEO_CHECKPOINT_PATH))
        tf3d = get_3d_tfm()
        status_3d = "Ready"
    except Exception as e:
        print(f"3D model unavailable: {e}")
        model3d, classes3d, tf3d = None, None, None
        status_3d = "Unavailable"

ACC_2D, ACC_3D = "86.0%", "91.6%"

# ----------------------------------------------------------------------------
# Design system — "Noir Lab": deep-space dark, aurora gradients, glass cards
# ----------------------------------------------------------------------------
CSS = """
@import url('https://fonts.cdnfonts.com/css/product-sans');

.gradio-container, .gradio-container * {
  font-family: 'Product Sans', 'Google Sans', 'Segoe UI', Roboto, sans-serif !important;
}
.gradio-container { max-width: 1220px !important; margin: auto; }

/* ---- foundation: gallery monochrome ---- */
:root {
  --body-background-fill: #0b0b0c;
  --background-fill-primary: #101012;
  --background-fill-secondary: #141417;
  --border-color-primary: #26262b;
  --body-text-color: #f4f3ef;
  --body-text-color-subdued: #a1a1aa;
  --block-title-text-color: #f4f3ef;
  --block-label-text-color: #a1a1aa;
  --button-primary-background-fill: #f4f3ef;
  --button-primary-text-color: #0b0b0c;
  --button-secondary-background-fill: transparent;
  --button-secondary-text-color: #f4f3ef;
  --button-secondary-border-color: #26262b;
  --accent: #c9a25e;
}

/* ---- film grain ---- */
.grain { position: fixed; inset: 0; z-index: 60; pointer-events: none; opacity: 0.05;
  background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="160" height="160"><filter id="n"><feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2"/></filter><rect width="160" height="160" filter="url(%23n)" opacity="1"/></svg>'); }

/* ---- top bar ---- */
.topbar { display: flex; align-items: center; justify-content: space-between;
  padding: 20px 4px 16px 4px; border-bottom: 1px solid #26262b;
  animation: fadeUp 0.7s ease both; }
.brand { display: flex; align-items: center; gap: 13px; }
.mark { width: 30px; height: 30px; border-radius: 50%; border: 1px solid #f4f3ef;
  position: relative; }
.mark::after { content: ''; position: absolute; inset: 8px; border-radius: 50%;
  background: #f4f3ef; }
.brand b { font-size: 16px; font-weight: 700; letter-spacing: 0.04em; }
.brand span { display: block; font-size: 11px; letter-spacing: 0.18em; color: #a1a1aa; }
.meta { font-size: 11px; letter-spacing: 0.22em; color: #a1a1aa; }
.meta .on { color: #f4f3ef; }

/* ---- editorial hero ---- */
.hero { position: relative; padding: 56px 4px 40px 4px;
  border-bottom: 1px solid #26262b; margin-bottom: 0;
  animation: fadeUp 0.7s 0.08s ease both; }
.eyebrow { display: flex; align-items: center; gap: 14px; font-size: 11px;
  letter-spacing: 0.32em; color: var(--accent); margin-bottom: 22px; }
.eyebrow::after { content: ''; height: 1px; width: 120px; background: var(--accent);
  opacity: 0.55; }
.hero h1 { margin: 0; font-size: 52px; line-height: 1.02; font-weight: 400;
  letter-spacing: -0.02em; color: #f4f3ef; max-width: 800px; }
.hero h1 em { font-style: normal; color: #8e8e96; }
.hero p { margin: 20px 0 0 0; font-size: 15.5px; line-height: 1.7; color: #a1a1aa;
  max-width: 620px; }
.spec { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; margin-top: 34px; }
@media (max-width: 800px) { .spec { grid-template-columns: 1fr; } }
.spec > div { border-top: 1px solid #3a3a41; padding-top: 12px; }
.spec .k { font-size: 10.5px; letter-spacing: 0.26em; color: #71717a; }
.spec .v { font-size: 15px; color: #f4f3ef; margin-top: 5px;
  font-variant-numeric: tabular-nums; }

/* ---- ledger status ---- */
.bento { display: grid; grid-template-columns: repeat(3, 1fr); margin: 26px 0;
  border-top: 1px solid #26262b; border-bottom: 1px solid #26262b;
  animation: fadeUp 0.7s 0.16s ease both; }
@media (max-width: 800px) { .bento { grid-template-columns: 1fr; } }
.bcard { padding: 20px 26px 22px 26px; border-left: 1px solid #26262b; }
.bcard:first-child { border-left: none; }
@media (max-width: 800px) { .bcard { border-left: none; border-top: 1px solid #26262b; }
  .bcard:first-child { border-top: none; } }
.bcard .idx { font-size: 11px; letter-spacing: 0.2em; color: var(--accent); }
.bcard .lbl { font-size: 10.5px; letter-spacing: 0.26em; color: #a1a1aa; margin-top: 10px; }
.bcard .num { font-size: 40px; font-weight: 400; letter-spacing: -0.02em;
  color: #f4f3ef; margin: 4px 0 2px 0; font-variant-numeric: tabular-nums; }
.bcard .num small { font-size: 15px; color: #71717a; }
.bcard .sub { font-size: 12.5px; color: #71717a; }
.dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  background: #f4f3ef; margin-right: 7px; }
.dot.off { background: #3f3f46; }

/* ---- workflow ledger ---- */
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px;
  margin: 26px 0; animation: fadeUp 0.7s 0.22s ease both; }
@media (max-width: 800px) { .steps { grid-template-columns: 1fr; } }
.step { border-top: 1px solid #3a3a41; padding-top: 12px; font-size: 13px;
  line-height: 1.6; color: #a1a1aa; }
.step b { display: block; color: #f4f3ef; font-size: 13.5px; font-weight: 700;
  margin-bottom: 4px; }
.step .n { color: var(--accent); font-size: 11px; letter-spacing: 0.2em; margin-right: 8px; }

/* ---- verdict panel ---- */
.verdict { border: 1px solid #3a3a41; border-radius: 10px; padding: 24px 26px;
  margin-top: 14px; background: #101012; animation: fadeUp 0.45s ease both; }
.verdict .k { font-size: 10.5px; letter-spacing: 0.3em; color: var(--accent); }
.verdict .v { font-size: 38px; font-weight: 400; letter-spacing: -0.02em;
  color: #f4f3ef; margin: 6px 0 14px 0; font-variant-numeric: tabular-nums; }
.meter { height: 2px; background: #26262b; overflow: hidden; }
.meter > div { height: 100%; background: var(--accent); }
.verdict .pct { font-size: 12.5px; color: #a1a1aa; margin-top: 10px;
  font-variant-numeric: tabular-nums; letter-spacing: 0.06em; }

/* ---- buttons & inputs ---- */
button.primary, button.lg.primary { background: #f4f3ef !important; color: #0b0b0c !important;
  border: none !important; font-weight: 700 !important; border-radius: 8px !important;
  padding: 12px !important; letter-spacing: 0.04em !important;
  transition: opacity 0.13s ease, transform 0.13s ease !important; }
button.primary:hover { opacity: 0.88 !important; transform: translateY(-1px); }
button.secondary { border: 1px solid #3a3a41 !important; border-radius: 8px !important; }

/* ---- ticker marquee ---- */
.ticker { overflow: hidden; border-top: 1px solid #26262b;
  border-bottom: 1px solid #26262b; margin: 0 0 26px 0; padding: 10px 0; }
.ticker-inner { display: inline-block; white-space: nowrap; font-size: 11px;
  letter-spacing: 0.3em; color: #52525b; animation: tick 26s linear infinite; }
.ticker-inner b { color: var(--accent); font-weight: 400; }
@keyframes tick { to { transform: translateX(-50%); } }

/* ---- outlined display type ---- */
.hero h1 .hollow { color: transparent; -webkit-text-stroke: 1px #52525b; }

@keyframes fadeUp { from { opacity: 0; transform: translateY(14px); }
  to { opacity: 1; transform: none; } }
.tab-nav, .tabs { animation: fadeUp 0.7s 0.28s ease both; }
.tab-nav button.selected { color: #f4f3ef !important;
  border-bottom: 2px solid var(--accent) !important; }

footer { text-align: center; color: #52525b; font-size: 12px;
  letter-spacing: 0.08em; margin: 26px 0 10px 0; }
.howto { font-size: 14px; line-height: 1.7; color: #a1a1aa; }
.howto b { color: #f4f3ef; }
"""

TOPBAR = f"""
<div class="grain"></div>
<div class="topbar">
<div class="brand"><div class="mark"></div>
<div><b>ACTIONSCOPE</b><span>HUMAN ACTION DETECTION</span></div></div>
<div class="meta"><span class="on">OPERATIONAL</span> &nbsp;/&nbsp; {DEVICE.upper()}</div>
</div>
"""

HEADER = """
<div class="hero">
<div class="eyebrow">VISION SYSTEM — N°01</div>
<h1>Every movement,<br><span class="hollow">precisely understood.</span></h1>
<p>Classify everyday actions in still images, detect each person in frame and label
them individually, or score motion over time with a spatiotemporal network.</p>
<div class="spec">
<div><div class="k">IMAGE MODEL</div><div class="v">B0 + B3 ensemble · 15 actions · 86.0%</div></div>
<div><div class="k">DETECTION</div><div class="v">YOLOv8n · per-person labels</div></div>
<div><div class="k">VIDEO MODEL</div><div class="v">R3D-18 · KTH · 91.6%</div></div>
</div></div>
"""

STATUS = f"""
<div class="bento">
<div class="bcard"><div class="idx">01</div>
<div class="lbl"><span class="dot"></span>IMAGE CLASSIFIER</div>
<div class="num">86.0<small> %</small></div>
<div class="sub">B0 + B3 ensemble · 15 actions · {DEVICE.upper()}</div></div>
<div class="bcard"><div class="idx">02</div>
<div class="lbl"><span class="dot{' off' if yolo is None else ''}"></span>PERSON DETECTOR</div>
<div class="num">YOLO<small>v8n</small></div>
<div class="sub">{status_yolo} · per-person labels · tunable confidence</div></div>
<div class="bcard"><div class="idx">03</div>
<div class="lbl"><span class="dot{' off' if model3d is None else ''}"></span>VIDEO MODEL</div>
<div class="num">91.6<small> %</small></div>
<div class="sub">R3D-18 · 6 KTH actions · {status_3d}</div></div>
</div>
"""

STEPS = """
<div class="steps">
<div class="step"><b><span class="n">01 —</span>Provide input</b>Upload, paste, or snap with webcam — or tap an example.</div>
<div class="step"><b><span class="n">02 —</span>Analyze</b>Run the model; tune confidence for detection.</div>
<div class="step"><b><span class="n">03 —</span>Explore</b>Read the verdict, inspect probabilities, try another tab.</div>
</div>
"""

TICK = "15 ACTIONS <b>·</b> 86.0% ENSEMBLE <b>·</b> YOLOV8 DETECTION <b>·</b> R3D-18 VIDEO <b>·</b> PYTORCH + GRADIO <b>·</b> "
TICKER = f'<div class="ticker"><div class="ticker-inner">{TICK}{TICK}</div></div>'

HOW_IT_WORKS = """
**Image classification** — photo → 300px → EfficientNet-B3 fine-tuned on 10,710 HAR images →
probabilities over 15 actions: calling, clapping, cycling, dancing, drinking, eating, fighting,
hugging, laughing, listening to music, running, sitting, sleeping, texting, using a laptop
(86.0% test as B0+B3 ensemble; 85.2% single B3).

**Person detection + action** — YOLOv8n finds every person; each crop is classified by the 2D model
and drawn back with its label. Lower the confidence slider if people are missed; raise it to remove
false boxes.

**Video clip (3D)** — 16 uniformly sampled frames → R3D-18 spatiotemporal CNN trained on 600 KTH clips
(boxing, handclapping, handwaving, jogging, running, walking) — 91.6% validation. Short,
single-action clips work best.
"""

FOOTER = f"""
<footer>ACTIONSCOPE — EFFICIENTNET-B0 · YOLOV8N · R3D-18 &nbsp;&nbsp;/&nbsp;&nbsp; PYTORCH + GRADIO · {DEVICE.upper()}</footer>
"""


# ----------------------------------------------------------------------------
# Backend
# ----------------------------------------------------------------------------
def verdict_html(kind, top, conf):
    pct = conf * 100
    return f"""
<div class="verdict"><div class="k">{kind}</div><div class="v">{top}</div>
<div class="meter"><div style="width:{pct:.1f}%"></div></div>
<div class="pct">{pct:.1f}% confidence</div></div>
"""


@torch.no_grad()
def classify_image(img: Image.Image):
    if img is None:
        return {}, verdict_html("TOP PREDICTION · ENSEMBLE", "Upload an image", 0.0)
    if ens_members is not None:
        probs = ensemble_probs(ens_members, img)
    else:
        x = tf2d(img.convert("RGB")).unsqueeze(0).to(DEVICE)
        probs = model2d(x).softmax(1)[0].cpu()
    result = {pretty(c): float(p) for c, p in zip(classes2d, probs)}
    top = max(result, key=result.get)
    return result, verdict_html("TOP PREDICTION · ENSEMBLE", top, result[top])


@torch.no_grad()
def detect_image(img: Image.Image, conf: float = YOLO_CONF):
    if img is None:
        return None, "Upload a photo first.", pd.DataFrame(columns=["Person", "Action", "Confidence", "Box"])
    frame = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    boxes = person_boxes(yolo, frame, conf) if yolo is not None else []
    out = annotate_frame(frame, boxes, model2d, classes2d, tf2d)
    annotated = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    if not boxes:
        probs = classify_image(img)[0]
        top = max(probs, key=probs.get)
        summary = (f"No person at confidence {conf:.2f} — full-frame call: "
                   f"{top} ({probs[top] * 100:.1f}%). Try lowering the slider.")
        rows = []
    else:
        rows = []
        for i, (x1, y1, x2, y2, det) in enumerate(boxes, 1):
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            action, prob = classify_crop(model2d, classes2d, tf2d, crop)
            rows.append([f"Person {i}", pretty(action), f"{prob * 100:.1f}%", f"[{x1},{y1},{x2},{y2}]"])
        summary = (f"{len(rows)} person(s) at confidence {conf:.2f} — "
                   f"each box carries its own action label.")
    table = pd.DataFrame(rows, columns=["Person", "Action", "Confidence", "Box"])
    return annotated, summary, table


@torch.no_grad()
def classify_video_clip(video_path):
    if not video_path:
        return {"Upload a clip first": 1.0}, verdict_html("TOP PREDICTION · 3D", "Waiting", 0.0)
    if model3d is None:
        return {"3D model not loaded": 1.0}, verdict_html("3D MODEL", "unavailable", 0.0)
    from .config import CLIP_LEN
    from .video_dataset import read_frames_uniform
    frames = read_frames_uniform(video_path, CLIP_LEN)
    clip = torch.stack([tf3d(f) for f in frames]).permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)
    probs = model3d(clip).softmax(1)[0].cpu()
    result = {pretty(c): float(p) for c, p in zip(classes3d, probs)}
    top = max(result, key=result.get)
    return result, verdict_html("TOP PREDICTION · 3D", top, result[top])


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
theme = gr.themes.Soft(primary_hue="neutral", secondary_hue="neutral")

with gr.Blocks(title="ActionScope — Human Action Detection") as demo:
    gr.HTML(TOPBAR)
    gr.HTML(HEADER)
    gr.HTML(STATUS)
    gr.HTML(STEPS)
    gr.HTML(TICKER)

    with gr.Tab("01 · Image classification"):
        gr.Markdown("What is the person doing? Upload, paste, or snap a photo.")
        with gr.Row():
            inp = gr.Image(type="pil", label="Input photo", sources=["upload", "webcam", "clipboard"])
            with gr.Column():
                out = gr.Label(label="All actions", num_top_classes=5)
                banner = gr.HTML(label="Verdict")
        with gr.Row():
            btn_img = gr.Button("Classify action", variant="primary")
            clr_img = gr.ClearButton(inp, value="Clear")
        gr.Examples(examples=["data/test/cycling/Image_10739.jpg",
                              "data/test/dancing/Image_10758.jpg"],
                    inputs=inp, label="Tap an example")
        btn_img.click(classify_image, inp, [out, banner])
        inp.change(classify_image, inp, [out, banner])

    with gr.Tab("02 · Person detection + action"):
        gr.Markdown("YOLO finds every person — each box gets its own action label.")
        conf_slider = gr.Slider(0.1, 0.9, value=YOLO_CONF, step=0.05,
                                label="Detection confidence — lower finds more, higher is stricter")
        with gr.Row():
            inp2 = gr.Image(type="pil", label="Input photo",
                            sources=["upload", "webcam", "clipboard"])
            out2 = gr.Image(label="Detections", interactive=False)
        summary = gr.Textbox(label="Result summary", interactive=False)
        people_table = gr.Dataframe(label="Per-person results", interactive=False, wrap=True)
        with gr.Row():
            btn_det = gr.Button("Detect people", variant="primary")
            clr_det = gr.ClearButton(inp2, value="Clear")
        gr.Examples(examples=["data/test/cycling/Image_10739.jpg"], inputs=inp2,
                    label="Tap an example")
        btn_det.click(detect_image, [inp2, conf_slider], [out2, summary, people_table])
        inp2.change(detect_image, [inp2, conf_slider], [out2, summary, people_table])
        conf_slider.change(detect_image, [inp2, conf_slider], [out2, summary, people_table])

    with gr.Tab("03 · Video clip (3D)"):
        gr.Markdown("Motion-aware prediction — short clips, one KTH action: boxing, handclapping, "
                    "handwaving, jogging, running, walking.")
        with gr.Row():
            inp3 = gr.Video(label="Input clip")
            with gr.Column():
                out3 = gr.Label(label="All actions", num_top_classes=3)
                banner3 = gr.HTML(label="Verdict")
        with gr.Row():
            btn_vid = gr.Button("Classify clip", variant="primary")
            clr_vid = gr.ClearButton(inp3, value="Clear")
        gr.Examples(examples=["data_kth/boxing/person01_boxing_d1_uncomp.avi"], inputs=inp3,
                    label="Tap an example")
        btn_vid.click(classify_video_clip, inp3, [out3, banner3])
        inp3.change(classify_video_clip, inp3, [out3, banner3])

    with gr.Accordion("How it works — models, data, tips", open=False):
        gr.Markdown(HOW_IT_WORKS, elem_classes="howto")

    gr.HTML(FOOTER)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true", help="public gradio.live link")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share,
                theme=theme, css=CSS)

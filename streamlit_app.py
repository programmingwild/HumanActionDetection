"""ActionScope on Streamlit Community Cloud (free, permanent, no card).

Torch-free: onnxruntime + numpy + PIL + opencv-headless only (~400MB RAM).
Run locally:  streamlit run streamlit_app.py
Deploy: share.streamlit.com -> repo -> streamlit_app.py
"""
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.labels import pretty
from src.ort_backend import CLASSES_3D, OrtEnsemble, OrtVideo, ckpt_dir, yolo_person_boxes
import onnxruntime as ort

st.set_page_config(page_title="ActionScope — Human Action Detection", layout="wide")

CSS = """
<style>
@import url('https://fonts.cdnfonts.com/css/product-sans');
html, body, .stApp { font-family: 'Product Sans','Google Sans',sans-serif; }
[data-testid="stFileUploader"] button { white-space: nowrap; }
:root { --accent: #c9a25e; }
.topbar { display:flex; justify-content:space-between; align-items:center;
  padding: 6px 2px 14px 2px; border-bottom:1px solid #26262b; margin-bottom:18px; }
.brand b { letter-spacing:.04em; } .brand span { font-size:11px; letter-spacing:.18em; color:#888; }
.hero-eyebrow { font-size:11px; letter-spacing:.32em; color:#c9a25e; margin-bottom:10px; }
.hero h1 { font-size:46px; line-height:1.02; letter-spacing:-.02em; font-weight:400; margin:0; }
.hero h1 .hollow { color:transparent; -webkit-text-stroke:1px #666; }
.hero p { color:#888; max-width:620px; }
.verdict { border:1px solid #3a3a41; border-radius:10px; padding:20px 24px; margin-top:12px; }
.verdict .k { font-size:10.5px; letter-spacing:.3em; color:#c9a25e; }
.verdict .v { font-size:34px; margin:4px 0 10px 0; }
.meter { height:2px; background:#26262b; } .meter div { height:100%; background:#c9a25e; }
.verdict .pct { font-size:12.5px; color:#888; margin-top:8px; }
.spec { display:grid; grid-template-columns:repeat(3,1fr); gap:20px; margin:22px 0 6px 0; }
.spec div { border-top:1px solid #3a3a41; padding-top:10px; font-size:13px; color:#888; }
.spec b { display:block; color:inherit; font-size:14px; }
footer { text-align:center; color:#666; font-size:12px; margin-top:26px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown('<div class="topbar"><div class="brand"><b>ACTIONSCOPE</b><br><span>HUMAN ACTION DETECTION</span></div>'
            '<div class="brand"><span>ONNX · STREAMLIT CLOUD</span></div></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-eyebrow">VISION SYSTEM — N°01</div>', unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>Every movement,<br><span class="hollow">precisely understood.</span></h1>'
            '<p>Ensemble image classifier (86.0%), YOLO person detection, and 3D video '
            'classification — running torch-free on ONNX Runtime.</p></div>', unsafe_allow_html=True)
st.markdown('<div class="spec"><div><b>IMAGE MODEL</b>B0 + B3 ensemble · 86.0%</div>'
            '<div><b>DETECTION</b>YOLOv8n ONNX · per-person labels</div>'
            '<div><b>VIDEO MODEL</b>R3D-18 ONNX · 91.6% KTH</div></div>', unsafe_allow_html=True)


@st.cache_resource
def load_all():
    d = ckpt_dir()
    ens = OrtEnsemble(d / "effb0.onnx", d / "effb3.onnx")
    yolo = ort.InferenceSession(str(d / "yolov8n.onnx"), providers=["CPUExecutionProvider"])
    vid = OrtVideo(d / "r3d18.onnx")
    return ens, yolo, vid


ens, yolo_sess, vid = load_all()


def verdict(kind, top, conf):
    st.markdown(f'<div class="verdict"><div class="k">{kind}</div><div class="v">{top}</div>'
                f'<div class="meter"><div style="width:{conf*100:.1f}%"></div></div>'
                f'<div class="pct">{conf*100:.1f}% confidence</div></div>', unsafe_allow_html=True)


tab1, tab2, tab3 = st.tabs(["01 · Image", "02 · Detection", "03 · Video"])

with tab1:
    f = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
    if f:
        img = Image.open(f)
        st.image(img, width=420)
        if st.button("Classify action", type="primary"):
            with st.spinner("Scoring ensemble…"):
                top, conf, probs = ens.top(img)
            verdict("TOP PREDICTION · ENSEMBLE", top, conf)
            st.bar_chart({k: v for k, v in sorted(probs.items(), key=lambda x: -x[1])[:5]})

with tab2:
    f2 = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"], key="det")
    conf_thr = st.slider("Detection confidence", 0.1, 0.9, 0.4, 0.05)
    if f2 and st.button("Detect people", type="primary"):
        img = Image.open(f2).convert("RGB")
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        with st.spinner("Detecting…"):
            boxes = yolo_person_boxes(yolo_sess, frame, conf_thr)
        rows = []
        for i, (x1, y1, x2, y2, dc) in enumerate(boxes, 1):
            crop = Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
            a, p, _ = ens.top(crop)
            rows.append({"Person": f"Person {i}", "Action": a, "Confidence": f"{p*100:.1f}%"})
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{a} {p:.2f}", (x1 + 5, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=560)
        st.write(f"Detected {len(rows)} person(s)." if rows else "No person found — try lowering confidence.")
        if rows:
            st.dataframe(rows, use_container_width=True)

with tab3:
    f3 = st.file_uploader("Upload a short clip", type=["mp4", "avi", "mov"])
    if f3 and st.button("Classify clip", type="primary"):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as t:
            t.write(f3.read())
            path = t.name
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        cap.release()
        if len(frames) < 4:
            st.error("Could not read enough frames from this clip.")
        else:
            with st.spinner("Scoring 3D model (slow on CPU)…"):
                p = vid.probs(frames)
            i = int(p.argmax())
            verdict("TOP PREDICTION · 3D", pretty(CLASSES_3D[i]), float(p[i]))
            st.bar_chart({pretty(c): float(v) for c, v in zip(CLASSES_3D, p)})

st.markdown("<footer>ACTIONSCOPE — ONNX ENSEMBLE · YOLOV8n · R3D-18 · STREAMLIT</footer>",
            unsafe_allow_html=True)

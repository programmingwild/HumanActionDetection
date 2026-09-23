"""Final QA: checkpoints, metrics spot-check, every feature path.

Usage: python docs/qa_check.py
Exits nonzero on any failure.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from src.infer import checkpoint_img_size, load_model  # noqa: E402
from src.labels import pretty  # noqa: E402
from src.model_3d import get_video_model  # noqa: E402

PASS = []


def check(name, fn):
    try:
        detail = fn()
        PASS.append((name, True, detail))
        print(f"PASS {name} {detail}")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        PASS.append((name, False, str(e) or type(e).__name__))
        print(f"FAIL {name}: {e}")


def ckpt_info(path):
    return torch.load(path, map_location="cpu", weights_only=False)


# 1. checkpoints load + forward
def t_b3():
    m, c = load_model("checkpoints/best_action_model.pth")
    dev = next(m.parameters()).device
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 300, 300).to(dev))
    assert tuple(out.shape) == (1, 15)
    info = ckpt_info("checkpoints/best_action_model.pth")
    return f"arch={info['arch']} acc={info['acc']:.4f} img={info.get('img_size')}"


def t_b0():
    m, c = load_model("checkpoints/best_effb0.pth")
    dev = next(m.parameters()).device
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 224, 224).to(dev))
    assert tuple(out.shape) == (1, 15)
    return f"acc={ckpt_info('checkpoints/best_effb0.pth')['acc']:.4f}"


def t_r3d():
    ck = ckpt_info("checkpoints/best_video_model.pth")
    m = get_video_model(len(ck["classes"]), arch=ck.get("arch", "r3d_18"), pretrained=False)
    m.load_state_dict(ck["model_state"])
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 16, 112, 112))
    assert tuple(out.shape) == (1, 6)
    return f"acc={ck['acc']:.4f}"


# 2. labels + guards
def t_labels():
    assert pretty("listening_to_music") == "Listening to Music"
    assert pretty("cycling") == "Cycling"
    assert pretty("handwaving") == "Hand Waving"
    return "8 names mapped"


def t_guards():
    import src.app as app
    assert app.classify_image(None)[0] == {}
    assert app.detect_image(None, 0.4)[1].startswith("Upload")
    assert "Upload a clip first" in app.classify_video_clip(None)[0]
    return "None-inputs handled"


# 3. feature paths (real data)
def t_infer_image():
    from src.infer import get_tf, predict_image
    m, c = load_model("checkpoints/best_action_model.pth")
    tf = get_tf(checkpoint_img_size("checkpoints/best_action_model.pth"))
    label, conf = predict_image(m, c, "data/test/cycling/Image_10739.jpg", tf=tf)
    assert label == "Cycling" and conf > 0.5, (label, conf)
    return f"{label} {conf:.2f}"


def t_app_paths():
    import src.app as app
    img = Image.open("data/test/cycling/Image_10739.jpg")
    d, _ = app.classify_image(img)
    assert max(d, key=d.get) == "Cycling"
    a, s, t = app.detect_image(img, 0.4)
    assert len(t) > 0
    r = app.classify_video_clip("data_kth/boxing/person01_boxing_d1_uncomp.avi")
    assert max(r[0], key=r[0].get) == "Boxing"
    return "image+detect+video OK"


for name, fn in [("ckpt-b3", t_b3), ("ckpt-b0", t_b0), ("ckpt-r3d", t_r3d),
                 ("labels", t_labels), ("guards", t_guards),
                 ("infer-image", t_infer_image), ("app-paths", t_app_paths)]:
    check(name, fn)

fails = [n for n, ok, _ in PASS if not ok]
print(f"\n{PASS and len(PASS) - len(fails)}/{len(PASS)} passed")
sys.exit(1 if fails else 0)

"""Build the PDF project report with fpdf2 (stdlib + fpdf2 only).

Usage: python docs/build_report.py  ->  Human_Action_Detection_Report.pdf
"""
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Human_Action_Detection_Report.pdf"
CM_IMG = ROOT / "confusion_matrix.png"

INDIGO = (79, 70, 229)
DARK = (17, 24, 39)
GRAY = (100, 116, 139)
LIGHT = (241, 245, 249)


def clean(s):
    return (s.replace("\u00b7", "-").replace("\u2014", "-").replace("\u2013", "-")
             .replace("\u2192", "->").replace("\u2713", "v").replace("\u00d7", "x"))


class Report(FPDF):
    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 10, f"Human Action Detection - Project Report  |  Page {self.page_no() - 1}",
                  align="C")

    def section(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*INDIGO)
        self.cell(0, 10, clean(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*INDIGO)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)
        self.set_text_color(*DARK)

    def body(self, text):
        self.set_font("Helvetica", "", 10.5)
        self.set_text_color(*DARK)
        self.multi_cell(0, 6, clean(text), new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def bullets(self, items):
        self.set_font("Helvetica", "", 10.5)
        self.set_text_color(*DARK)
        for it in items:
            self.multi_cell(0, 6, "-  " + clean(it), new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def table(self, headers, rows, widths=None):
        w = (self.w - self.l_margin - self.r_margin)
        widths = widths or [w / len(headers)] * len(headers)
        self.set_font("Helvetica", "B", 10)
        self.set_fill_color(*INDIGO)
        self.set_text_color(255, 255, 255)
        for h, cw in zip(headers, widths):
            self.cell(cw, 8, clean(h), border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*DARK)
        fill = False
        for row in rows:
            if fill:
                self.set_fill_color(*LIGHT)
            for val, cw in zip(row, widths):
                self.cell(cw, 7.5, clean(str(val)), border=1, fill=fill, align="C")
            self.ln()
            fill = not fill
        self.ln(3)


pdf = Report()
pdf.set_auto_page_break(True, margin=20)

# ---- Cover ----
pdf.add_page()
pdf.ln(45)
pdf.set_font("Helvetica", "B", 30)
pdf.set_text_color(*INDIGO)
pdf.multi_cell(0, 13, "Human Action\nDetection", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(4)
pdf.set_font("Helvetica", "", 13)
pdf.set_text_color(*GRAY)
pdf.cell(0, 8, "2D Classification  |  3D Video Models  |  YOLO Detection + Web App",
         align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(10)
pdf.set_font("Helvetica", "B", 12)
pdf.set_text_color(*DARK)
for line in ["EfficientNet-B3 @300px: 85.24% test",
             "R3D-18 on KTH: 91.6% validation",
             "Stack: PyTorch + YOLOv8 + Gradio | RTX 5060"]:
    pdf.cell(0, 8, line, align="C", new_x="LMARGIN", new_y="NEXT")

# ---- Sections ----
pdf.add_page()
pdf.section("1. Introduction and Objectives")
pdf.body("ActionScope is an end-to-end human action detection system in PyTorch with three "
         "branches: a 2D frame classifier for 15 everyday actions, a true-temporal 3D CNN for "
         "6 video actions, and YOLOv8 person detection fused with the 2D head for per-person "
         "labels. A premium Gradio app serves all three.")
pdf.bullets(["Classify 15 routine activities in still images.",
             "Model motion over time with 3D CNNs on real clips.",
             "Detect people (YOLO) and label each person's action.",
             "Ship a polished demo, honest metrics, and deployment artifacts."])

pdf.section("2. Datasets (Kaggle)")
pdf.table(["Dataset", "Size", "Classes", "Split"],
          [["HAR-15 (images)", "~218 MB, 12,600 imgs", "15", "10,710 / 1,890"],
           ["KTH (video)", "600 clips, 25 subj.", "6", "train / val (80/20)"]],
          widths=[52, 55, 30, 53])
pdf.body("HAR-15 classes: calling, clapping, cycling, dancing, drinking, eating, fighting, "
         "hugging, laughing, listening_to_music, running, sitting, sleeping, texting, "
         "using_laptop. KTH classes: boxing, handclapping, handwaving, jogging, running, "
         "walking. Flat KTH filenames are auto-organized into per-class folders.")

pdf.section("3. Methodology")
pdf.bullets(["2D: ImageNet-pretrained EfficientNet-B3 @300px, head replaced; AdamW, cosine schedule, "
             "label smoothing 0.1, AMP mixed precision, RandomErasing/RandAugment/MixUp options; "
             "resolution stored in checkpoint and honored by all inference paths.",
             "3D: R3D-18 on 16-frame 112px clips; sliding-window inference with probability "
             "averaging. (Torchvision S3D was tested and removed: its final pooling crashes "
             "on 112px inputs.)",
             "Detection: YOLOv8n person boxes -> 2D crop classification -> annotated frame; "
             "full-frame fallback; tunable confidence.",
             "App: dark Gradio UI with verdict banners, per-person table, examples, lite mode."])

pdf.section("4. Results")
pdf.table(["Model", "Epochs", "Accuracy", "Note"],
          [["MobileNetV3-Small", "12", "74.02%", "baseline"],
           ["EfficientNet-B0", "25", "82.75%", "mid-step"],
           ["EfficientNet-B3 @300", "23", "85.24%", "champion"],
           ["R3D-18 (KTH val)", "10", "91.6%", "video branch"]])
pdf.body("Strongest 2D classes (F1): cycling 0.984, eating 0.949, dancing 0.887. Hardest: "
         "sitting 0.706, calling 0.777, texting 0.789 - visually near-identical poses plus "
         "label noise, a data ceiling rather than a pure modeling gap. 95% top-1 is therefore "
         "not a realistic target on HAR-15; the honest path (B3 @300px + aug + TTA + ensemble) "
         "points to the high-80s/low-90s.")
if CM_IMG.exists():
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*DARK)
    pdf.cell(0, 8, "Confusion matrix - EfficientNet-B3 (test set)", new_x="LMARGIN", new_y="NEXT")
    pdf.image(str(CM_IMG), x=25, w=160)
    pdf.ln(2)

pdf.section("5. Reproduce")
pdf.body("pip install -r requirements.txt + CUDA torch (cu128) + numpy<2. "
         "Download HAR-15 and KTH via src.download_data, train with src.train / src.train_video, "
         "evaluate with src.evaluate (--tta), launch with python -m src.app. Full commands are in "
         "docs/PROJECT_DOCUMENTATION.md.")
pdf.bullets(["python -m src.infer --image photo.jpg | --video in.mp4 --out out.mp4 | --webcam",
             "python -m src.yolo_action --source 0 --out out.mp4",
             "python -m src.inference_video_3d clip.avi -o out_3d.mp4",
             "Deploy: Hugging Face Spaces (free), --share link, or Docker on a GPU VM."])

pdf.section("6. Limitations and Future Work")
pdf.bullets(["2D sees single frames; 3D covers only 6 KTH actions.",
             "Small/occluded YOLO crops inherit 2D confusions.",
             "Next: B3 @300px + ensemble, cross-subject KTH eval, pose branch, ONNX export."])

pdf.output(str(OUT))
print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {pdf.pages_count} pages)")

# HumanActionDetection
ActionScope: human action detection in PyTorch. EfficientNet B0+B3 ensemble (86.9% on 15 everyday actions), R3D-18 video classifier (91.6% on KTH-6), YOLOv8 per-person detection, served through a dark Gradio demo and a torch-free Streamlit Cloud app. Trained on RTX 5060. Kaggle HAR-15 + KTH datasets.

## Improve the image model with hard examples
Weak classes (sitting, calling, texting) are visually ambiguous, so more of the same data helps little. Mine hard training samples, then fine-tune with RandAugment + MixUp and weighted sampling:

```bash
python -m src.hard_mining --checkpoint checkpoints/best_action_model.pth --out data/hard_weights.json
python -m src.train --arch efficientnet_b3 --img-size 300 --epochs 8 --lr 3e-5 \
  --randaugment --mixup 0.2 --hard-weights data/hard_weights.json \
  --resume checkpoints/best_action_model.pth --out checkpoints/best_b3_hard.pth
python -m src.evaluate --tta --checkpoint checkpoints/best_b3_hard.pth
python -m src.export_onnx --arch efficientnet_b3 --checkpoint checkpoints/best_b3_hard.pth \
  --img-size 300 --out checkpoints/effb3.onnx
```
Promote only if held-out accuracy beats the current checkpoint.

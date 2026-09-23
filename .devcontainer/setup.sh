#!/bin/bash
# First-boot setup: deps + large weights. Runs once on codespace creation.
set -e
git lfs pull || true
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install --quiet -r requirements.txt
pip install --quiet "numpy<2"
python -c "import torch, gradio; print('deps ok:', torch.__version__)"
ls -la checkpoints/

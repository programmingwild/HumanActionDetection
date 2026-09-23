#!/bin/bash
# Oracle Always Free (Ampere ARM) one-shot setup. Run from repo root on the VM.
# Usage: bash deploy/setup_oracle.sh
set -e
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git git-lfs libgl1 libglib2.0-0 curl
git lfs pull || true
python3 -m venv .venv
.venv/bin/pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install --quiet -r requirements.txt
.venv/bin/pip install --quiet "numpy<2"
# open the port on the VM firewall (also add TCP 7860 ingress in the OCI console security list)
sudo iptables -I INPUT -p tcp --dport 7860 -j ACCEPT 2>/dev/null || true
sudo netfilter-persistent save 2>/dev/null || true
# install 24/7 service
sudo cp deploy/actionscope.service /etc/systemd/system/actionscope.service
sudo systemctl daemon-reload
sudo systemctl enable --now actionscope
sleep 3
sudo systemctl is-active actionscope
echo "Open: http://$(curl -s --max-time 5 ifconfig.me):7860"
echo "Logs: sudo journalctl -u actionscope -f"

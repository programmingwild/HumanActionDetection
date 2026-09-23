# GPU image (use pytorch/pytorch:2.11.0-cpu for CPU-only hosts)
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir "numpy<2"

COPY src/ src/
COPY checkpoints/ checkpoints/
COPY README.md .

ENV PORT=7860
EXPOSE 7860
CMD ["python", "-m", "src.app", "--port", "7860"]

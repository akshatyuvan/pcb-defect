# Serving image for the PCB defect API.
# Built and run on Apple Silicon -> linux/arm64. The PyPI torch wheel for
# aarch64 is CPU-only by construction, so no extra index and no CUDA bloat.
FROM python:3.12-slim

# Serving deps only (no mlflow, no opencv, no matplotlib). See
# requirements-serving.txt for why each omission is deliberate.
WORKDIR /app
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# torch spawns an OpenMP thread pool sized to the host's core count, which on a
# 4GB Docker allocation means several hundred MB of arenas for no throughput
# gain on a 1.17M-param model. Pin it in the env AND in config.TORCH_THREADS.
ENV OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PCB_TORCH_THREADS=2 \
    PCB_DEVICE=cpu \
    PCB_MODEL_DIR=/app/artifacts/serving \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY src/ ./src/
COPY artifacts/serving/ ./artifacts/serving/

# Non-root. Not security theatre: if this ever bind-mounts a host dir, root in
# the container writing root-owned files onto your Desktop is a real annoyance.
RUN useradd -m -u 10001 pcb && chown -R pcb:pcb /app
USER pcb

EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4).status==200 else 1)"

# ONE worker. Each uvicorn worker is a separate process with its own copy of
# the model (~1GB RSS). Day 8 scales consumers, which hold no model, not this.
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

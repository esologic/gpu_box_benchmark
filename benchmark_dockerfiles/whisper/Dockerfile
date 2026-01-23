FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG SESSION_ID=fallback

LABEL gpu_box_benchmark_session=$SESSION_ID

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# PyTorch with CUDA
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Whisper
RUN pip install --no-cache-dir openai-whisper

# Build-Time download of the model to use.
RUN python3 -c "import whisper; whisper.load_model('medium', device='cpu')"

WORKDIR /app

# Copy benchmark script and fixed audio
COPY run_whisper_benchmark.py /app/run_whisper_benchmark.py
COPY moby_dick_audio.ogg /app/audio/input.ogg

ENTRYPOINT ["python3", "/app/run_whisper_benchmark.py"]

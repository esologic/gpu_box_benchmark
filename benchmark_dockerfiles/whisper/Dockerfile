# Use 11.8: The latest CUDA version that still supports Tesla K80 (Kepler)
FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Whisper and pre-download model
RUN pip install --no-cache-dir torch==1.13.1 torchvision==0.14.1 openai-whisper
RUN python3 -c "import whisper; whisper.load_model('medium', device='cpu')"

WORKDIR /app

# Copy benchmark script and fixed audio
COPY run_whisper_benchmark.py /app/run_whisper_benchmark.py
COPY moby_dick_audio.ogg /app/audio/input.ogg

ENTRYPOINT ["python3", "/app/run_whisper_benchmark.py"]
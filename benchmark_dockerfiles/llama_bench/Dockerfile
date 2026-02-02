FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor -o /usr/share/keyrings/kitware.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/kitware.gpg] https://apt.kitware.com/ubuntu/ focal main" > /etc/apt/sources.list.d/kitware.list && \
    apt-get update && apt-get install -y cmake && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone https://github.com/ggml-org/llama.cpp.git .

# Compile llama.cpp with Kepler-specific flags
RUN cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="37" \
  -DGGML_CUDA_GRAPHS=OFF \
  && cmake --build build -j$(nproc)

# Install binaries and library to system path
RUN cp build/bin/* /usr/local/bin/ && \
    cp build/bin/*.so* /usr/local/lib/ || true

# Install Python tools
RUN pip install --no-cache-dir -U huggingface_hub hf_transfer
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Download Models
WORKDIR /models

# Qwen 2.5 1.5B
RUN bash -c 'set -e; for i in 1 2 3 4 5; do \
  hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir . && break || sleep 10; \
done'

# Llama 3 8B Instruct
RUN bash -c 'set -e; for i in 1 2 3 4 5; do \
  hf download bartowski/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --local-dir . && break || sleep 10; \
done'

# Qwen 1.5 MoE A2.7B
RUN bash -c 'set -e; for i in 1 2 3 4 5; do \
  hf download tensorblock/Qwen1.5-MoE-A2.7B-Chat-GGUF Qwen1.5-MoE-A2.7B-Chat-Q2_K.gguf --local-dir . && break || sleep 10; \
done'

# OpenMistral MoE
RUN bash -c 'set -e; for i in 1 2 3 4 5; do \
  hf download tensorblock/OpenMistral-MoE-GGUF OpenMistral-MoE-Q2_K.gguf --local-dir . && break || sleep 10; \
done'

# Verification of downloads.
RUN du -sh /models/*.gguf

WORKDIR /app

# Copy the bash script into the container
COPY ./run_llama_bench.sh /app/run_llama_bench.sh
RUN chmod +x /app/run_llama_bench.sh

ENTRYPOINT ["/app/run_llama_bench.sh"]

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Base deps
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    python3 \
    python3-pip \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install modern CMake (Kitware)
RUN curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor -o /usr/share/keyrings/kitware.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/kitware.gpg] https://apt.kitware.com/ubuntu/ jammy main" > /etc/apt/sources.list.d/kitware.list && \
    apt-get update && apt-get install -y cmake && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Clone ik_llama
RUN git clone https://github.com/ikawrakow/ik_llama.cpp.git

WORKDIR /opt/ik_llama.cpp

# TODO: needs a cuda arch per gpu that is going in !!

# Build with CUDA + NCCL
# ---- Build ik-llama with CUDA + NCCL ----
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_NCCL=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA_NO_WMMA=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_CUDA_ARCHITECTURES="50;60;70" \
    -DCMAKE_CXX_FLAGS="-mcmodel=large" \
    -DCMAKE_C_FLAGS="-mcmodel=large" \
    && cmake --build build -j$(nproc)

# Install huggingface_hub to download models.
RUN pip install --no-cache-dir -U huggingface_hub hf_transfer

# 3. Ensure the 'hf' binary is in the PATH and enable fast transfers
ENV PATH="/root/.local/bin:/usr/local/bin:$PATH"
ENV HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /models

# Qwen 1.5 MoE A2.7B
RUN hf download tensorblock/Qwen1.5-MoE-A2.7B-Chat-GGUF Qwen1.5-MoE-A2.7B-Chat-Q2_K.gguf --local-dir .

# 4. Verification
RUN du -sh /models/*.gguf

WORKDIR /app

# Move all binaries to /app
RUN cp /opt/ik_llama.cpp/build/bin/* /app/ && \
    chmod +x /app/*

# Copy benchmark script
COPY ./run_llama_bench.sh /app/run_llama_bench.sh
RUN chmod +x /app/run_llama_bench.sh

ENTRYPOINT ["/app/run_llama_bench.sh"]

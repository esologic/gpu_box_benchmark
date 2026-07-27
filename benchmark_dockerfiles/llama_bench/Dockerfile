FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    python3-pip \
    curl \
    gpg \
    jq \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor -o /usr/share/keyrings/kitware.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/kitware.gpg] https://apt.kitware.com/ubuntu/ focal main" > /etc/apt/sources.list.d/kitware.list && \
    apt-get update && \
    apt-get install -y cmake && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://github.com/ggml-org/llama.cpp.git .

# Compile llama.cpp with Kepler-specific flags
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="35;37;50;52;60;61;70;75" \
    -DGGML_CUDA_GRAPHS=OFF \
    && cmake --build build -j$(nproc)

# Install binaries and libraries
RUN cp build/bin/* /usr/local/bin/ && \
    cp build/bin/*.so* /usr/local/lib/ || true

# Install Hugging Face download tools
RUN pip install --no-cache-dir -U huggingface_hub hf_transfer
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Persistent model directory (bind mount at runtime)
RUN mkdir -p /models

# Copy benchmark runner
COPY run_llama_bench.sh /app/run_llama_bench.sh
RUN chmod +x /app/run_llama_bench.sh

WORKDIR /app

ENTRYPOINT ["/app/run_llama_bench.sh"]
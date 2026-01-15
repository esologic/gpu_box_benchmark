FROM ghcr.io/ggml-org/llama.cpp:full-cuda

# Install curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /models

# Qwen 2.5 1.5B (Requires ~1GB VRAM)
RUN curl -L -o tiny_model.gguf \
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf?download=true"

# Llama 3 8B Instruct (Requires ~5GB VRAM)
# This is the standard benchmark size.
RUN curl -L -o standard_model.gguf \
    "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf?download=true"

WORKDIR /app

# Copy the bash script into the container
COPY ./run_llama_bench.sh /app/run_llama_bench.sh
RUN chmod +x /app/run_llama_bench.sh

ENTRYPOINT ["/app/run_llama_bench.sh"]
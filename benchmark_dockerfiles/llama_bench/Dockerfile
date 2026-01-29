# See:
# * https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md
# * https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md

FROM ghcr.io/ggml-org/llama.cpp:full-cuda

ARG SESSION_ID=fallback

LABEL gpu_box_benchmark_session=$SESSION_ID

# Install system dependencies
RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*

# Install modern huggingface_hub to pull models
RUN pip install --no-cache-dir -U huggingface_hub hf_transfer

# 3. Ensure the 'hf' binary is in the PATH and enable fast transfers
ENV PATH="/root/.local/bin:/usr/local/bin:$PATH"
ENV HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /models

# Qwen 2.5 1.5B
RUN hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir .

# Llama 3 8B Instruct
RUN hf download bartowski/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --local-dir .

# Qwen 1.5 MoE A2.7B
RUN hf download tensorblock/Qwen1.5-MoE-A2.7B-Chat-GGUF Qwen1.5-MoE-A2.7B-Chat-Q2_K.gguf --local-dir .

# OpenMistral MoE
RUN hf download tensorblock/OpenMistral-MoE-GGUF OpenMistral-MoE-Q2_K.gguf --local-dir .

# 4. Verification
RUN du -sh /models/*.gguf

WORKDIR /app

# Copy the bash script into the container
COPY ./run_llama_bench.sh /app/run_llama_bench.sh
RUN chmod +x /app/run_llama_bench.sh

ENTRYPOINT ["/app/run_llama_bench.sh"]

#!/bin/bash
set -euo pipefail

# Environment variables
MODEL_DIR="${MODEL_DIR:-/models}"

MODEL_PATH="${MODEL_PATH:-${MODEL_DIR}/qwen2.5-1.5b-instruct-q4_k_m.gguf}"
NUM_PROMPT_TOKENS="${NUM_PROMPT_TOKENS:-512}"
NUM_GENERATION_TOKENS="${NUM_GENERATION_TOKENS:-128}"
PERFORMANCE_ONLY="${PERFORMANCE_ONLY:-0}"


download_model() {
    local repo="$1"
    local filename="$2"

    if [[ -f "${MODEL_DIR}/${filename}" ]]; then
        echo "Already exists: ${filename}"
        return
    fi

    echo "Downloading ${filename}"

    for attempt in {1..5}; do
        if hf download \
            "${repo}" \
            "${filename}" \
            --local-dir "${MODEL_DIR}"; then
            return
        fi

        echo "Download failed (attempt ${attempt}/5), retrying..."
        sleep 10
    done

    echo "Failed downloading ${filename}"
    exit 1
}


# Performance benchmark models
download_model "Qwen/Qwen2.5-1.5B-Instruct-GGUF" "qwen2.5-1.5b-instruct-q4_k_m.gguf"
download_model "bartowski/Meta-Llama-3-8B-Instruct-GGUF" "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
download_model "tensorblock/Qwen1.5-MoE-A2.7B-Chat-GGUF" "Qwen1.5-MoE-A2.7B-Chat-Q2_K.gguf"


if [[ "${PERFORMANCE_ONLY}" != "1" ]]; then
    # Capacity benchmark models
    download_model "bartowski/Qwen3-7B-GGUF" "Qwen3-7B-Q4_K_M.gguf"
    download_model "bartowski/Qwen3-14B-GGUF" "Qwen3-14B-Q4_K_M.gguf"
    download_model "bartowski/Qwen3-32B-GGUF" "Qwen3-32B-Q4_K_M.gguf"
    download_model "bartowski/Qwen3-72B-GGUF" "Qwen3-72B-Q4_K_M.gguf"
    download_model "bartowski/Qwen3-72B-GGUF" "Qwen3-72B-Q5_K_M.gguf"
    download_model "bartowski/Qwen3-72B-GGUF" "Qwen3-72B-Q6_K.gguf"
    download_model "bartowski/Qwen3-72B-GGUF" "Qwen3-72B-Q8_0.gguf"
fi


llama-bench \
    --output json \
    --repetitions 10 \
    --model "${MODEL_PATH}" \
    --n-prompt "${NUM_PROMPT_TOKENS}" \
    --n-gen "${NUM_GENERATION_TOKENS}" \
    --n-gpu-layers 99 \
    --flash-attn 1 \
    --n-cpu-moe 0
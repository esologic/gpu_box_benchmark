#!/bin/bash
set -euo pipefail

# Environment variables
MODEL_DIR="${MODEL_DIR:-/tmp/downloads/models}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${MODEL_DIR}/.cache}"
# MODEL_PATH accepts a space-separated list of models to try
MODEL_PATH="${MODEL_PATH:-${MODEL_DIR}/qwen2.5-1.5b-instruct-q4_k_m.gguf}"
NUM_PROMPT_TOKENS="${NUM_PROMPT_TOKENS:-512}"
NUM_GENERATION_TOKENS="${NUM_GENERATION_TOKENS:-128}"
NUM_REPETITIONS="${NUM_REPETITIONS:-10}"
PERFORMANCE_ONLY="${PERFORMANCE_ONLY:-1}"

download_model() {
    local repo="$1"
    local pattern="$2"

    # Enable nullglob (empty array if no match) AND globstar (recursive ** search)
    shopt -s nullglob globstar

    # The ** tells bash to look in MODEL_DIR and any folders inside it
    local existing_files=("${MODEL_DIR}"/**/${pattern})

    # Turn them back off to prevent unexpected behavior elsewhere in the script
    shopt -u nullglob globstar

    if [[ ${#existing_files[@]} -gt 0 ]]; then
        echo "Already exists: ${pattern} (${#existing_files[@]} file(s) found)"
        return
    fi

    echo "Downloading ${pattern} from ${repo}"

    for attempt in {1..5}; do
        if hf download \
            "${repo}" \
            --include "${pattern}" \
            --cache-dir "${HF_CACHE_DIR}" \
            --local-dir "${MODEL_DIR}"; then
            return
        fi

        echo "Download failed (attempt ${attempt}/5), retrying..."
        sleep 10
    done

    echo "Failed downloading ${pattern}"
    exit 1
}

mkdir -p "${HF_CACHE_DIR}"
mkdir -p "${MODEL_DIR}"

# Performance benchmark models
download_model "Qwen/Qwen2.5-1.5B-Instruct-GGUF" "qwen2.5-1.5b-instruct-q4_k_m*.gguf"
download_model "bartowski/Meta-Llama-3-8B-Instruct-GGUF" "Meta-Llama-3-8B-Instruct-Q4_K_M*.gguf"
download_model "tensorblock/Qwen1.5-MoE-A2.7B-Chat-GGUF" "Qwen1.5-MoE-A2.7B-Chat-Q2_K*.gguf"

if [[ "${PERFORMANCE_ONLY}" != 1 ]]; then
  download_model "Qwen/Qwen2.5-0.5B-Instruct-GGUF" "qwen2.5-0.5b-instruct-q4_k_m*.gguf"
  download_model "Qwen/Qwen2.5-1.5B-Instruct-GGUF" "qwen2.5-1.5b-instruct-q4_k_m*.gguf"
  download_model "Qwen/Qwen2.5-3B-Instruct-GGUF" "qwen2.5-3b-instruct-q4_k_m*.gguf"
  download_model "Qwen/Qwen2.5-7B-Instruct-GGUF" "qwen2.5-7b-instruct-q4_k_m*.gguf"
  download_model "Qwen/Qwen2.5-14B-Instruct-GGUF" "qwen2.5-14b-instruct-q4_k_m*.gguf"
  download_model "Qwen/Qwen2.5-32B-Instruct-GGUF" "qwen2.5-32b-instruct-q4_k_m*.gguf"
  download_model "Qwen/Qwen2.5-72B-Instruct-GGUF" "qwen2.5-72b-instruct-q4_k_m*.gguf"
  download_model "bartowski/Gigaberg-Mistral-Large-123B-GGUF" "Gigaberg-Mistral-Large-123B-Q4_K_M*.gguf"
fi

# Parse the space-separated string into an array
read -ra MODELS <<< "$MODEL_PATH"

ERROR_LOG=$(mktemp)
SUCCESS=0

for current_model in "${MODELS[@]}"; do
    echo "Attempting to benchmark: ${current_model}" >&2

    TEMP_OUT=$(mktemp)
    TEMP_ERR=$(mktemp)

    # We use if/then to gracefully handle llama-bench failing without triggering set -e
    if llama-bench \
        --output json \
        --repetitions "${NUM_REPETITIONS}" \
        --model "${current_model}" \
        --n-prompt "${NUM_PROMPT_TOKENS}" \
        --n-gen "${NUM_GENERATION_TOKENS}" \
        --n-gpu-layers 99 \
        --flash-attn 1 \
        --n-cpu-moe 0 > "${TEMP_OUT}" 2> "${TEMP_ERR}"; then

        # Success! Print JSON and set flag
        cat "${TEMP_OUT}"
        SUCCESS=1

        rm -f "${TEMP_OUT}" "${TEMP_ERR}"
        break
    else
        # Record the error for this specific model
        echo "Error for model: ${current_model}" >> "${ERROR_LOG}"
        cat "${TEMP_ERR}" >> "${ERROR_LOG}"
        echo -e "\n----------------------------------------\n" >> "${ERROR_LOG}"
    fi

    rm -f "${TEMP_OUT}" "${TEMP_ERR}"
done

# If we looped through everything and never flipped SUCCESS to 1
if [[ "${SUCCESS}" -eq 0 ]]; then
    echo "All models failed. Error log below:" >&2
    cat "${ERROR_LOG}" >&2
    rm -f "${ERROR_LOG}"
    exit 1
fi

rm -f "${ERROR_LOG}"
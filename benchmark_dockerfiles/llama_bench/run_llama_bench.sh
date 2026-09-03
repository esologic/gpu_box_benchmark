#!/bin/bash
set -euo pipefail

# Environment variables
MODEL_DIR="${MODEL_DIR:-/tmp/downloads/models}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${MODEL_DIR}/.cache}"

# MODEL_SPECS is a JSON array of objects of the form:
#
# [
#   {
#     "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
#     "download": "qwen2.5-7b-instruct-q4_k_m*.gguf",
#     "entrypoint": "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
#   },
#   ...
# ]
#
# Models are attempted in the order they appear in the array.
#
# entrypoint:
#   The GGUF file passed to llama-bench --model.
#   For multipart GGUF models, this must be the first shard
#   (for example: -00001-of-00012.gguf).

DEFAULT_MODEL_SPECS='[{"repo": "unsloth/Qwen3.6-35B-A3B-GGUF", "download": "Qwen3.6-35B-A3B-Q4_K_M*.gguf", "entrypoint": "Qwen3.6-35B-A3B-Q4_K_M.gguf"}]'
MODEL_SPECS="${MODEL_SPECS:-$DEFAULT_MODEL_SPECS}"

NUM_PROMPT_TOKENS="${NUM_PROMPT_TOKENS:-512}"
NUM_GENERATION_TOKENS="${NUM_GENERATION_TOKENS:-128}"
NUM_REPETITIONS="${NUM_REPETITIONS:-10}"
CONTEXT_DEPTH="${CONTEXT_DEPTH:-0}"

# This catches cases where the env var is empty, corrupted, or invalid JSON.
if ! echo "${MODEL_SPECS}" | jq empty >/dev/null 2>&1; then
    echo "Error: MODEL_SPECS contains invalid JSON." >&2
    echo "Raw content of MODEL_SPECS (full string):" >&2
    # Logging the full raw content as requested to diagnose injection issues
    echo "${MODEL_SPECS}" >&2
    echo "----------------------------------------" >&2
    exit 1
fi

mkdir -p "${HF_CACHE_DIR}"
mkdir -p "${MODEL_DIR}"

ensure_model() {
    local repo="$1"
    local download="$2"

    echo "Ensuring model '${download}' from '${repo}' is available..." >&2

    for attempt in {1..5}; do
        if hf download \
            "${repo}" \
            --include "${download}" \
            --cache-dir "${HF_CACHE_DIR}" \
            --local-dir "${MODEL_DIR}"; then
            return 0
        fi

        echo "Download failed (attempt ${attempt}/5), retrying..." >&2
        sleep 10
    done

    echo "Failed downloading '${download}' from '${repo}'." >&2
    exit 1
}

ERROR_LOG=$(mktemp)
SUCCESS=0

while IFS=$'\t' read -r repo download entrypoint; do
    ensure_model "${repo}" "${download}"

    current_model="${MODEL_DIR}/${entrypoint}"

    echo "Attempting to benchmark: ${current_model}" >&2

    TEMP_OUT=$(mktemp)
    TEMP_ERR=$(mktemp)

    # Gracefully handle llama-bench failures without triggering 'set -e'.
    if llama-bench \
        --output json \
        --repetitions "${NUM_REPETITIONS}" \
        --model "${current_model}" \
        --n-depth "${CONTEXT_DEPTH}" \
        --n-prompt "${NUM_PROMPT_TOKENS}" \
        --n-gen "${NUM_GENERATION_TOKENS}" \
        --n-gpu-layers -1 \
        --no-host \
        --no-kv-offload=false \
        --flash-attn 1 \
        --n-cpu-moe 0 \
        > "${TEMP_OUT}" \
        2> "${TEMP_ERR}"; then

        cat "${TEMP_OUT}"
        SUCCESS=1

        rm -f "${TEMP_OUT}" "${TEMP_ERR}"
        break
    else
        {
            echo "Error for model: ${current_model}"
            cat "${TEMP_ERR}"
            echo
            echo "----------------------------------------"
            echo
        } >> "${ERROR_LOG}"
    fi

    rm -f "${TEMP_OUT}" "${TEMP_ERR}"

# End of the while loop. The '< <(...)' process substitution feeds the loop
# with tab-separated repo/download/entrypoint triples generated from MODEL_SPECS.
done < <(
    jq -r '.[] | [.repo, .download, .entrypoint] | @tsv' <<< "${MODEL_SPECS}"
)

if [[ "${SUCCESS}" -eq 0 ]]; then
    echo "All models failed. Error log below:" >&2
    cat "${ERROR_LOG}" >&2
    rm -f "${ERROR_LOG}"
    exit 1
fi

rm -f "${ERROR_LOG}"
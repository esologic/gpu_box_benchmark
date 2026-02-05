#!/bin/bash
set -euo pipefail

# Environment variables
MODEL_PATH="${MODEL_PATH:-/models/qwen2.5-1.5b-instruct-q4_k_m.gguf}"
NUM_PROMPT_TOKENS="${NUM_PROMPT_TOKENS:-512}"
NUM_GENERATION_TOKENS="${NUM_GENERATION_TOKENS:-128}"

# Run llama-bench
/app/llama-bench \
--output json \
--repetitions 10 \
--model "${MODEL_PATH}" \
--n-prompt "${NUM_PROMPT_TOKENS}" \
--n-gen "${NUM_GENERATION_TOKENS}" \
--n-gpu-layers 99 \
--flash-attn 1 \
-sm graph

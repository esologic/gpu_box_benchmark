#!/bin/bash
set -euo pipefail

# Environment variables
MODEL_PATH="${MODEL_PATH:-'/models/tiny_model.gguf'}"
NUM_PROMPT_TOKENS="${NUM_PROMPT_TOKENS:-512}"
NUM_GENERATION_TOKENS="${NUM_GENERATION_TOKENS:-128}"

# Run llama-bench
/app/llama-bench \
--output json \
--repetitions 10 \
--model "${MODEL_PATH}" \
--n-prompt "${NUM_PROMPT_TOKENS}" \
--n-gen "${NUM_GENERATION_TOKENS}"


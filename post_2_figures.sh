#!/usr/bin/env bash

# 1️⃣ Define your input JSON files in an array
INPUTS=(
  "./post_2_files/24_Intel_Xeon_CPU_E5-2687W_v4_@_3.00GHz_1_Tesla_P100-PCIE-16GB_2026-09-15T17_27_15.json"
  "./post_2_files/24_Intel_Xeon_CPU_E5-2687W_v4_@_3.00GHz_2_Tesla_P100-PCIE-16GB_2026-09-14T20_47_32.json"
  "./post_2_files/24_Intel_Xeon_CPU_E5-2687W_v4_@_3.00GHz_3_Tesla_P100-PCIE-16GB_2026-09-13T18_57_41.json"
  "./post_2_files/24_Intel_Xeon_CPU_E5-2687W_v4_@_3.00GHz_1_Tesla_V100-PCIE-16GB_2026-09-13T10_47_18.json"
  "./post_2_files/24_Intel_Xeon_CPU_E5-2687W_v4_@_3.00GHz_2_Tesla_V100-PCIE-16GB_2026-09-12T12_17_15.json"
  "./post_2_files/24_Intel_Xeon_CPU_E5-2687W_v4_@_3.00GHz_3_Tesla_V100-PCIE-16GB_2026-09-11T13_31_26.json"
)

# 2️⃣ Build the command-line arguments array correctly.
# We iterate to create distinct entries: ["--input", "path1", "--input", "path2", ...]
INPUT_ARGS=()
for f in "${INPUTS[@]}"; do
    INPUT_ARGS+=("--input" "$f")
done

# 3️⃣ Run the commands using the constructed arguments
python bench_cli.py compare-systems \
  "${INPUT_ARGS[@]}" \
  --only-test blender_benchmark_monster_gpu \
  --only-test content_aware_timelapse_vit_attention \
  --only-test fah_bench_single \
  --only-test whisper_medium_fp16 \
  --only-test hashcat_sha256 \
  --only-test llama_bench_qwen_2_5_3b_instruct_f16_128k_ctx_generation \
  --only-test llama_bench_gemma_4_8b_f16_128k_ctx_generation \
  --only-test llama_bench_max_model_parameters_q4_4k_ctx \
  --only-test llama_bench_max_model_parameters_f16_32k_ctx \
  --n-cols 3 \
  --watermark "esologic.com/telsas-for-llms" \
  --font-family "Inconsolata" \
  --y-axis-pad-pct 0.3 \
  --output-path ./post_2_images/results_power.svg \
  --mode results_power \
  --title "Assorted System Tests"

python bench_cli.py compare-systems \
  "${INPUT_ARGS[@]}" \
  --only-test blender_benchmark_monster_gpu \
  --only-test content_aware_timelapse_vit_attention \
  --only-test fah_bench_single \
  --only-test whisper_medium_fp16 \
  --only-test hashcat_sha256 \
  --only-test llama_bench_qwen_2_5_3b_instruct_f16_128k_ctx_generation \
  --only-test llama_bench_gemma_4_8b_f16_128k_ctx_generation \
  --only-test llama_bench_max_model_parameters_q4_4k_ctx \
  --only-test llama_bench_max_model_parameters_f16_32k_ctx \
  --n-cols 3 \
  --watermark "esologic.com/telsas-for-llms" \
  --font-family "Inconsolata" \
  --output-path ./post_2_images/results.svg \
  --mode results \
  --title "Assorted System Tests"

python bench_cli.py compare-llm-families \
  "${INPUT_ARGS[@]}" \
  --watermark "esologic.com/telsas-for-llms" \
  --font-family "Inconsolata" \
  --output-dir ./post_2_images/llm_families


# "./post_2_files/8_Intel_Xeon_CPU_E5-1680_v4_@_3.40GHz_2_Tesla_M60_2026-09-11T17_31_49.json"
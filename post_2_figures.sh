#!/usr/bin/env bash

# Define shared configuration variables
WATERMARK="esologic.com/multi-gpu-scaling"
FONT_FAMILY="Inconsolata"

# Define your input file paths and their corresponding group names
# We map these to the inferred GPU types (K80, P100, V100) for color-coding
FILE_PATHS=(
  "./post_2_files/24_Intel_Xeon_CPU_E5-2687W_v4_@_3.00GHz_2_Tesla_K80_2026-09-28T15_17_03.json"
  "./post_2_files/24_Intel_Xeon_CPU_E5-2687W_v4_@_3.00GHz_4_Tesla_K80_2026-09-23T01_52_25.json"
  "./post_2_files/24_Intel_Xeon_CPU_E5-2687W_v4_@_3.00GHz_6_Tesla_K80_2026-09-18T17_16_05.json"
  "./post_2_files/24_Intel_Xeon_CPU_E5-2687W_v4_@_3.00GHz_1_Tesla_P100-PCIE-16GB_2026-09-15T17_27_15.json"
  "./post_2_files/24_Intel_Xeon_CPU_E5-2687W_v4_@_3.00GHz_2_Tesla_P100-PCIE-16GB_2026-09-16T01_33_06.json"
  "./post_2_files/24_Intel_Xeon_CPU_E5-2687W_v4_@_3.00GHz_3_Tesla_P100-PCIE-16GB_2026-09-17T09_09_02.json"
  "./post_2_files/24_Intel_Xeon_CPU_E5-2687W_v4_@_3.00GHz_1_Tesla_V100-PCIE-16GB_2026-09-13T10_47_18.json"
  "./post_2_files/24_Intel_Xeon_CPU_E5-2687W_v4_@_3.00GHz_2_Tesla_V100-PCIE-16GB_2026-09-12T12_17_15.json"
  "./post_2_files/24_Intel_Xeon_CPU_E5-2687W_v4_@_3.00GHz_3_Tesla_V100-PCIE-16GB_2026-09-11T13_31_26.json"
)

GROUP_NAMES=(
  "K80"
  "K80"
  "K80"
  "P100"
  "P100"
  "P100"
  "V100"
  "V100"
  "V100"
)

# Build the grouped-input command-line arguments
GROUPED_INPUT_ARGS=()
for i in "${!FILE_PATHS[@]}"; do
    GROUPED_INPUT_ARGS+=("--grouped-input" "${FILE_PATHS[$i]}" "${GROUP_NAMES[$i]}")
done

# Build the regular input command-line arguments
INPUT_ARGS=()
for file_path in "${FILE_PATHS[@]}"; do
    INPUT_ARGS+=("--input" "$file_path")
done

# Max Parameters
python bench_cli.py compare-systems \
  "${GROUPED_INPUT_ARGS[@]}" \
  --only-test llama_bench_max_model_parameters_q4_4k_ctx \
  --only-test llama_bench_max_model_parameters_q4_32k_ctx \
  --only-test llama_bench_max_model_parameters_q8_4k_ctx \
  --only-test llama_bench_max_model_parameters_q8_32k_ctx \
  --only-test llama_bench_max_model_parameters_f16_4k_ctx \
  --only-test llama_bench_max_model_parameters_f16_32k_ctx \
  --n-cols 2 \
  --watermark "${WATERMARK}" \
  --font-family "${FONT_FAMILY}" \
  --y-axis-pad-pct 0.3 \
  --output-path ./post_2_images/max_parameters.svg \
  --mode results \
  --title "Max Model Parameter Count" \
  --legend-columns 3

# Tensor Split Comparison (Split vs Non-Split)
python bench_cli.py compare-systems \
  "${GROUPED_INPUT_ARGS[@]}" \
  --only-test llama_bench_qwen_2_5_14b_instruct_q4_4k_ctx_generation \
  --only-test llama_bench_qwen_2_5_14b_instruct_q4_4k_ctx_generation_e_tensor_split \
  --only-test llama_bench_qwen_2_5_14b_instruct_q4_32k_ctx_generation \
  --only-test llama_bench_qwen_2_5_14b_instruct_q4_32k_ctx_generation_e_tensor_split \
  --only-test llama_bench_gemma_4_8b_q4_4k_ctx_generation \
  --only-test llama_bench_gemma_4_8b_q4_4k_ctx_generation_e_tensor_split \
  --only-test llama_bench_gemma_4_8b_f16_128k_ctx_generation \
  --only-test llama_bench_gemma_4_8b_f16_128k_ctx_generation_e_tensor_split \
  --n-cols 2 \
  --watermark "${WATERMARK}" \
  --font-family "${FONT_FAMILY}" \
  --y-axis-pad-pct 0.3 \
  --output-path ./post_2_images/tensor_split_comparison.svg \
  --mode results \
  --title "Split Mode Tensor Comparison" \
  --legend-columns 3

# 70 tok/s bar - results
python bench_cli.py compare-systems \
  "${GROUPED_INPUT_ARGS[@]}" \
  --only-test llama_bench_qwen_2_5_3b_instruct_q8_128k_ctx_generation \
  --only-test llama_bench_gemma_4_8b_q8_64k_ctx_generation \
  --only-test llama_bench_gemma_4_26b_q8_64k_ctx_generation \
  --only-test llama_bench_qwen_3_6_35b_q8_128k_ctx_generation \
  --n-cols 2 \
  --watermark "${WATERMARK}" \
  --font-family "${FONT_FAMILY}" \
  --y-axis-pad-pct 0.3 \
  --output-path ./post_2_images/across_models_results.svg \
  --mode results \
  --title "Biggest Model + CTX @ 70 Tok/s" \
  --legend-columns 3

# Grouped power plot
python bench_cli.py compare-systems \
  "${GROUPED_INPUT_ARGS[@]}" \
  --only-test blender_benchmark_monster_gpu \
  --only-test content_aware_timelapse_vit_attention \
  --only-test fah_bench_single \
  --only-test resnet50_infer_batch_256 \
  --only-test whisper_medium_fp16 \
  --only-test hashcat_sha256 \
  --n-cols 2 \
  --watermark "${WATERMARK}" \
  --font-family "${FONT_FAMILY}" \
  --y-axis-pad-pct 0.3 \
  --output-path ./post_2_images/non_llm_results_power.svg \
  --mode results_power \
  --title "Non-LLM Tests" \
  --legend-columns 3

# 70 tok/s bar - results and power
python bench_cli.py compare-systems \
  "${GROUPED_INPUT_ARGS[@]}" \
  --only-test llama_bench_qwen_2_5_3b_instruct_q8_128k_ctx_generation \
  --only-test llama_bench_gemma_4_8b_q8_64k_ctx_generation \
  --only-test llama_bench_gemma_4_26b_q8_64k_ctx_generation \
  --only-test llama_bench_qwen_3_6_35b_q8_128k_ctx_generation \
  --n-cols 2 \
  --watermark "${WATERMARK}" \
  --font-family "${FONT_FAMILY}" \
  --y-axis-pad-pct 0.3 \
  --output-path ./post_2_images/across_models_results_power.svg \
  --mode results_power \
  --title "Biggest Model + CTX @ 70 Tok/s" \
  --legend-columns 3

# Grouped LLM families
python bench_cli.py compare-llm-families \
  "${GROUPED_INPUT_ARGS[@]}" \
  --watermark "${WATERMARK}" \
  --font-family "${FONT_FAMILY}" \
  --output-dir ./post_2_images/llm_families \
  --ext .svg \
  --legend-columns 3

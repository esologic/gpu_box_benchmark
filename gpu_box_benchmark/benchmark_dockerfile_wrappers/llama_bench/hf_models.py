"""
Describes where the models are located on Huggingface, adds relevant info to be able to download
and use them.
"""

from gpu_box_benchmark.benchmark_dockerfile_wrappers.llama_bench.llama_bench_types import (
    HFModelSpec,
    ModelQuantization,
)

QWEN_2_5_1_5B_MODEL = HFModelSpec(
    repo="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    download="qwen2.5-1.5b-instruct-q4_k_m*.gguf",
    entrypoint="qwen2.5-1.5b-instruct-q4_k_m.gguf",
)

META_LLAMA_3_8B_MODEL = HFModelSpec(
    repo="bartowski/Meta-Llama-3-8B-Instruct-GGUF",
    download="Meta-Llama-3-8B-Instruct-Q4_K_M*.gguf",
    entrypoint="Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
)

QWEN_1_5_MOE_MODEL = HFModelSpec(
    repo="tensorblock/Qwen1.5-MoE-A2.7B-Chat-GGUF",
    download="Qwen1.5-MoE-A2.7B-Chat-Q2_K*.gguf",
    entrypoint="Qwen1.5-MoE-A2.7B-Chat-Q2_K.gguf",
)

QWEN_2_5_3B_INSTRUCT = ModelQuantization(
    q4=HFModelSpec(
        repo="Qwen/Qwen2.5-3B-Instruct-GGUF",
        download="qwen2.5-3b-instruct-q4_k_m*.gguf",
        entrypoint="qwen2.5-3b-instruct-q4_k_m.gguf",
    ),
    q8=HFModelSpec(
        repo="Qwen/Qwen2.5-3B-Instruct-GGUF",
        download="qwen2.5-3b-instruct-q8_0*.gguf",
        entrypoint="qwen2.5-3b-instruct-q8_0.gguf",
    ),
    f16=HFModelSpec(
        repo="Qwen/Qwen2.5-3B-Instruct-GGUF",
        download="qwen2.5-3b-instruct-fp16*.gguf",
        entrypoint="qwen2.5-3b-instruct-fp16.gguf",
    ),
)

QWEN_2_5_14B_INSTRUCT = ModelQuantization(
    q4=HFModelSpec(
        repo="Qwen/Qwen2.5-14B-Instruct-GGUF",
        download="qwen2.5-14b-instruct-q4_k_m*.gguf",
        entrypoint="qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf",
    ),
    q8=HFModelSpec(
        repo="Qwen/Qwen2.5-14B-Instruct-GGUF",
        download="qwen2.5-14b-instruct-q8_0*.gguf",
        entrypoint="qwen2.5-14b-instruct-q8_0.gguf",
    ),
    f16=HFModelSpec(
        repo="Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
        download="qwen2.5-coder-14b-instruct-fp16-*.gguf",
        entrypoint="qwen2.5-coder-14b-instruct-fp16-00001-of-00004.gguf",
    ),
)

QWEN_2_5_32B_INSTRUCT = ModelQuantization(
    q4=HFModelSpec(
        repo="Qwen/Qwen2.5-32B-Instruct-GGUF",
        download="qwen2.5-32b-instruct-q4_k_m*.gguf",
        entrypoint="qwen2.5-32b-instruct-q4_k_m-00001-of-00005.gguf",
    ),
    q8=HFModelSpec(
        repo="Qwen/Qwen2.5-32B-Instruct-GGUF",
        download="qwen2.5-32b-instruct-q8_0*.gguf",
        entrypoint="qwen2.5-32b-instruct-q8_0-00001-of-00009.gguf",
    ),
    f16=HFModelSpec(
        repo="Qwen/Qwen2.5-32B-Instruct-GGUF",
        download="qwen2.5-32b-instruct-fp16*.gguf",
        entrypoint="qwen2.5-32b-instruct-fp16-00001-of-00015.gguf",
    ),
)

QWEN_2_5_72B_INSTRUCT = ModelQuantization(
    q4=HFModelSpec(
        repo="Qwen/Qwen2.5-72B-Instruct-GGUF",
        download="qwen2.5-72b-instruct-q4_k_m*.gguf",
        entrypoint="qwen2.5-72b-instruct-q4_k_m-00001-of-00012.gguf",
    ),
    q8=HFModelSpec(
        repo="Qwen/Qwen2.5-72B-Instruct-GGUF",
        download="qwen2.5-72b-instruct-q8_0*.gguf",
        entrypoint="qwen2.5-72b-instruct-q8_0-00001-of-00021.gguf",
    ),
    f16=HFModelSpec(
        repo="Qwen/Qwen2.5-72B-Instruct-GGUF",
        download="qwen2.5-72b-instruct-fp16*.gguf",
        entrypoint="qwen2.5-72b-instruct-fp16-00001-of-00030.gguf",
    ),
)

# Should be 16B
DEEPSEEK_CODER_V2_LITE = ModelQuantization(
    q4=HFModelSpec(
        repo="lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        download="DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M*.gguf",
        entrypoint="DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
    ),
    q8=HFModelSpec(
        repo="lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        download="DeepSeek-Coder-V2-Lite-Instruct-Q8*.gguf",
        entrypoint="DeepSeek-Coder-V2-Lite-Instruct-Q8_0.gguf",
    ),
    f16=HFModelSpec(
        repo="lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        download="DeepSeek-Coder-V2-Lite-Instruct-F16*.gguf",
        entrypoint="DeepSeek-Coder-V2-Lite-Instruct-F16.gguf",
    ),
)

GEMMA_4_E4B_8B = ModelQuantization(
    q4=HFModelSpec(
        repo="unsloth/gemma-4-E4B-it-GGUF",
        download="gemma-4-E4B-it-Q4_K_M*.gguf",
        entrypoint="gemma-4-E4B-it-Q4_K_M.gguf",
    ),
    q8=HFModelSpec(
        repo="unsloth/gemma-4-E4B-it-GGUF",
        download="gemma-4-E4B-it-Q8_0*.gguf",
        entrypoint="gemma-4-E4B-it-Q8_0.gguf",
    ),
    f16=HFModelSpec(
        repo="unsloth/gemma-4-E4B-it-GGUF",
        download="gemma-4-E4B-it-BF16*.gguf",
        entrypoint="gemma-4-E4B-it-BF16.gguf",
    ),
)

GEMMA_4_12B = ModelQuantization(
    q4=HFModelSpec(
        repo="unsloth/gemma-4-12b-it-GGUF",
        download="gemma-4-12b-it-Q4_K_M*.gguf",
        entrypoint="gemma-4-12b-it-Q4_K_M.gguf",
    ),
    q8=HFModelSpec(
        repo="unsloth/gemma-4-12b-it-GGUF",
        download="gemma-4-12b-it-Q8_0*.gguf",
        entrypoint="gemma-4-12b-it-Q8_0.gguf",
    ),
    f16=HFModelSpec(
        repo="unsloth/gemma-4-12b-it-GGUF",
        download="gemma-4-12b-it-BF16*.gguf",
        entrypoint="gemma-4-12b-it-BF16.gguf",
    ),
)

GEMMA_4_26B = ModelQuantization(
    q4=HFModelSpec(
        repo="unsloth/gemma-4-26B-A4B-it-GGUF",
        download="gemma-4-26B-A4B-it-Q4_K_M*.gguf",
        entrypoint="gemma-4-26B-A4B-it-Q4_K_M.gguf",
    ),
    q8=HFModelSpec(
        repo="unsloth/gemma-4-26B-A4B-it-GGUF",
        download="gemma-4-26B-A4B-it-Q8_0*.gguf",
        entrypoint="gemma-4-26B-A4B-it-Q8_0.gguf",
    ),
    f16=HFModelSpec(
        repo="unsloth/gemma-4-26B-A4B-it-GGUF",
        download="BF16/gemma-4-26B-A4B-it-BF16-*.gguf",
        entrypoint="BF16/gemma-4-26B-A4B-it-BF16-00001-of-00002.gguf",
    ),
)

GEMMA_4_31B = ModelQuantization(
    q4=HFModelSpec(
        repo="unsloth/gemma-4-31B-it-GGUF",
        download="gemma-4-31B-it-Q4_K_M*.gguf",
        entrypoint="gemma-4-31B-it-Q4_K_M.gguf",
    ),
    q8=HFModelSpec(
        repo="unsloth/gemma-4-31B-it-GGUF",
        download="gemma-4-31B-it-Q8*.gguf",
        entrypoint="gemma-4-31B-it-Q8_0.gguf",
    ),
    f16=HFModelSpec(
        repo="unsloth/gemma-4-31B-it-GGUF",
        download="BF16/gemma-4-31B-it-BF16-*.gguf",
        entrypoint="BF16/gemma-4-31B-it-BF16-00001-of-00002.gguf",
    ),
)

QWEN_3_6_27B = ModelQuantization(
    q4=HFModelSpec(
        repo="unsloth/Qwen3.6-27B-GGUF",
        download="Qwen3.6-27B-Q4_K_M*.gguf",
        entrypoint="Qwen3.6-27B-Q4_K_M.gguf",
    ),
    q8=HFModelSpec(
        repo="unsloth/Qwen3.6-27B-GGUF",
        download="Qwen3.6-27B-Q8_0*.gguf",
        entrypoint="Qwen3.6-27B-Q8_0.gguf",
    ),
    f16=HFModelSpec(
        repo="unsloth/Qwen3.6-27B-GGUF",
        download="BF16/Qwen3.6-27B-BF16-*.gguf",
        entrypoint="BF16/Qwen3.6-27B-BF16-00001-of-00002.gguf",
    ),
)

QWEN_3_6_35B = ModelQuantization(
    q4=HFModelSpec(
        repo="unsloth/Qwen3.6-35B-A3B-GGUF",
        download="Qwen3.6-35B-A3B-Q4_K_M*.gguf",
        entrypoint="Qwen3.6-35B-A3B-Q4_K_M.gguf",
    ),
    q8=HFModelSpec(
        repo="unsloth/Qwen3.6-35B-A3B-GGUF",
        download="Qwen3.6-35B-A3B-Q8_0*.gguf",
        entrypoint="Qwen3.6-35B-A3B-Q8_0.gguf",
    ),
    f16=HFModelSpec(
        repo="unsloth/Qwen3.6-35B-A3B-GGUF",
        download="BF16/Qwen3.6-35B-A3B-BF16-*.gguf",
        entrypoint="BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf",
    ),
)

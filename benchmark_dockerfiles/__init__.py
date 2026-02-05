"""Allows us to easily locate assets within this directory from python code."""

from pathlib import Path

_ASSETS_DIRECTORY = Path(__file__).parent.resolve()

RESNET50_DOCKERFILE = _ASSETS_DIRECTORY / "resnet50" / "Dockerfile"
LLAMA_BENCH_DOCKERFILE = _ASSETS_DIRECTORY / "llama_bench" / "Dockerfile"
BLENDER_BENCHMARK_DOCKERFILE = _ASSETS_DIRECTORY / "blender_benchmark" / "Dockerfile"
FAHBENCH_BENCHMARK_DOCKERFILE = _ASSETS_DIRECTORY / "fah_bench" / "Dockerfile"
AI_BENCHMARK_DOCKERFILE = _ASSETS_DIRECTORY / "ai_benchmark" / "Dockerfile"
WHISPER_DOCKERFILE = _ASSETS_DIRECTORY / "whisper" / "Dockerfile"
CONTENT_AWARE_TIMELAPSE_DOCKERFILE = _ASSETS_DIRECTORY / "content_aware_timelapse" / "Dockerfile"
NVIDIA_GDS_DOCKERFILE = _ASSETS_DIRECTORY / "nvidia_gds" / "Dockerfile"
HASHCAT_DOCKERFILE = _ASSETS_DIRECTORY / "hashcat" / "Dockerfile"

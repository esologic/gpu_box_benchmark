# See: https://opendata.blender.org/
# Tested to work on K80 - T40

FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV BLENDER_VERSION=3.6.0
ENV BENCHMARK_LAUNCHER_VERSION=3.3.0
ENV BENCHMARK_LAUNCHER_URL=https://download.blender.org/release/BlenderBenchmark2.0/launcher/benchmark-launcher-cli-${BENCHMARK_LAUNCHER_VERSION}-linux.tar.gz

RUN apt-get update && apt-get install -y \
    xz-utils libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxi6 libxrender1 libxxf86vm1 libxkbcommon0 \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN curl -fsSL "${BENCHMARK_LAUNCHER_URL}" -o benchmark-launcher.tar.gz \
    && tar -xzf benchmark-launcher.tar.gz \
    && chmod +x benchmark-launcher-cli \
    && rm benchmark-launcher.tar.gz

RUN ./benchmark-launcher-cli blender download ${BLENDER_VERSION}
RUN ./benchmark-launcher-cli devices --blender-version ${BLENDER_VERSION}  # sanity check to make sure we can actually run benchmarks.
RUN ./benchmark-launcher-cli scenes download monster --blender-version ${BLENDER_VERSION}

COPY ./run_blender_benchmark.sh /app/run_blender_benchmark.sh
RUN chmod +x /app/run_blender_benchmark.sh

ENTRYPOINT ["/app/run_blender_benchmark.sh"]

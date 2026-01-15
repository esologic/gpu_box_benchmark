FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    wget xz-utils libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxi6 libxrender1 libxxf86vm1 libxkbcommon0 \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

ENV BLENDER_VERSION=4.5.0
ENV DEVICE_TYPE=CUDA

WORKDIR /app

RUN wget https://download.blender.org/release/BlenderBenchmark2.0/launcher/benchmark-launcher-cli-3.1.0-linux.tar.gz \
    && tar -xzf benchmark-launcher-cli-3.1.0-linux.tar.gz \
    && chmod +x benchmark-launcher-cli \
    && rm benchmark-launcher-cli-3.1.0-linux.tar.gz

RUN ./benchmark-launcher-cli blender download ${BLENDER_VERSION}
RUN ./benchmark-launcher-cli devices --blender-version ${BLENDER_VERSION}  # sanity check to make sure we can actually run benchmarks.
RUN ./benchmark-launcher-cli scenes download monster --blender-version ${BLENDER_VERSION}

# Copy the bash script into the container
COPY ./run_blender_benchmark.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# ---------------------------
# Entrypoint
# ---------------------------
ENTRYPOINT ["/app/entrypoint.sh"]

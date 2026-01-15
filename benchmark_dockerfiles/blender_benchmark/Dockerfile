FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    wget xz-utils libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxi6 libxrender1 libxxf86vm1 libxkbcommon0 \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

ENV BLENDER_VERSION=4.5.0
ENV TEST_DEVICE=CUDA

WORKDIR /app

RUN wget https://download.blender.org/release/BlenderBenchmark2.0/launcher/benchmark-launcher-cli-3.1.0-linux.tar.gz \
    && tar -xzf benchmark-launcher-cli-3.1.0-linux.tar.gz \
    && chmod +x benchmark-launcher-cli \
    && rm benchmark-launcher-cli-3.1.0-linux.tar.gz

RUN ./benchmark-launcher-cli blender download ${BLENDER_VERSION}
RUN ./benchmark-launcher-cli devices --blender-version ${BLENDER_VERSION}  # sanity check to make sure we can actually run benchmarks.
RUN ./benchmark-launcher-cli scenes download monster --blender-version ${BLENDER_VERSION}

RUN echo '#!/bin/bash' > /app/entrypoint.sh \
 && echo 'set -e' >> /app/entrypoint.sh \
 && echo '' >> /app/entrypoint.sh \
 && echo '# Use default env vars if not provided' >> /app/entrypoint.sh \
 && echo 'BLENDER_VERSION="${BLENDER_VERSION}"' >> /app/entrypoint.sh \
 && echo 'TEST_DEVICE="${TEST_DEVICE}"' >> /app/entrypoint.sh \
 && echo '' >> /app/entrypoint.sh \
 && echo 'exec ./benchmark-launcher-cli benchmark --blender-version "$BLENDER_VERSION" --device-type "$TEST_DEVICE" --json monster' >> /app/entrypoint.sh \
 && chmod +x /app/entrypoint.sh

# ---------------------------
# Entrypoint
# ---------------------------
ENTRYPOINT ["/app/entrypoint.sh"]

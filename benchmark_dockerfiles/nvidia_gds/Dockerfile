FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install GDS user-space tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcufile-12-4 \
    libcufile-dev-12-4 \
    gds-tools-12-4 \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Add GDS tools to PATH
ENV PATH="/usr/local/cuda/gds/tools:${PATH}"

WORKDIR /app

# Test file that will be loaded/unloaded to the GPUs. Built into the image here so it can be re-used
RUN truncate -s 5G /app/model_benchmark.bin

# Copy script to /app
COPY run_gds_bench.sh /app/run_gds_bench.sh
RUN chmod +x /app/run_gds_bench.sh

ENTRYPOINT ["/app/run_gds_bench.sh"]
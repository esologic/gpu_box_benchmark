FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG SESSION_ID=fallback

LABEL gpu_box_benchmark_session=$SESSION_ID

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    clinfo \
    ocl-icd-opencl-dev \
    build-essential \
    libnvidia-compute-550 \
    clinfo \
    cmake \
    git \
    fftw3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN git clone https://github.com/fahbench/fahbench.git

WORKDIR /opt/fahbench

RUN git submodule update --init --recursive

RUN mkdir build
WORKDIR /opt/fahbench/build

# Disable GUI, allow OpenCL autodetection
RUN cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_GUI=OFF

RUN make -j$(nproc)
RUN make install

# Copy the wrapper script
COPY run_fah_bench.sh /usr/local/bin/run_fah_bench.sh
RUN chmod +x /usr/local/bin/run_fah_bench.sh

# Default working dir
WORKDIR /opt/fahbench

# Set the entrypoint to the script
ENTRYPOINT ["/usr/local/bin/run_fah_bench.sh"]

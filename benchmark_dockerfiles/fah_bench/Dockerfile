FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies and the ICD loader
RUN apt-get update && apt-get install -y \
    ocl-icd-libopencl1 \
    ocl-icd-opencl-dev \
    build-essential \
    cmake \
    git \
    fftw3-dev \
    && rm -rf /var/lib/apt/lists/*

# Manually configure the NVIDIA OpenCL vendor file
# This is crucial for the GPU to be recognized in older environments
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

WORKDIR /opt

# Clone FAHBench
RUN git clone --recursive https://github.com/fahbench/fahbench.git /opt/fahbench

RUN mkdir build
WORKDIR /opt/fahbench/build

# Disable GUI, allow OpenCL autodetection
RUN cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_GUI=OFF

RUN make -j$(nproc)
RUN make install

WORKDIR /app
COPY run_fah_bench.sh /app/run_fah_bench.sh
RUN chmod +x /app/run_fah_bench.sh

ENTRYPOINT ["/app/run_fah_bench.sh"]

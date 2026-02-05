# See: https://hashcat.net/hashcat/
# Tested to work on K80 - V100

FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    hashcat \
    ocl-icd-libopencl1 \
    clinfo \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

WORKDIR /app

COPY run_hashcat_benchmark.sh /app/run_hashcat_benchmark.sh
RUN chmod +x /app/run_hashcat_benchmark.sh

ENTRYPOINT ["/app/run_hashcat_benchmark.sh"]
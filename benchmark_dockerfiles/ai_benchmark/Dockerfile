# See: https://ai-benchmark.com/
# Good info can be found on their forum: https://ai-benchmark.net/index.php

FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        make \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install pyenv to get python3.7
ENV PYENV_ROOT=/opt/pyenv
ENV PATH="$PYENV_ROOT/bin:$PATH"

RUN curl -fsSL https://pyenv.run | bash

# Install Python 3.7.x via pyenv
ENV PYTHON_VERSION=3.7.17
RUN pyenv install ${PYTHON_VERSION}

# Create venv using the pyenv interpreter directly
RUN /opt/pyenv/versions/${PYTHON_VERSION}/bin/python -m venv /opt/venv

# Activate venv via PATH (no shell activation)
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Standard upgrade of tooling inside venv
RUN /opt/venv/bin/python -m pip install --upgrade \
        pip \
        setuptools \
        wheel

# Install compatible TF + AI Benchmark in the venv
RUN /opt/venv/bin/python -m pip install --no-cache-dir \
        tensorflow==2.10.1 \
        ai-benchmark

COPY ./run_ai_benchmark.py /app/run_ai_benchmark.py

# Run the benchmark python file
ENTRYPOINT ["/opt/venv/bin/python", "/app/run_ai_benchmark.py"]

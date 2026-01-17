# syntax=docker/dockerfile:1

FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# ---- system deps for pyenv + Python build ----
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

# ---- install pyenv ----
ENV PYENV_ROOT=/opt/pyenv
ENV PATH="$PYENV_ROOT/bin:$PATH"

RUN curl -fsSL https://pyenv.run | bash

# ---- install Python 3.7.x via pyenv ----
ENV PYTHON_VERSION=3.7.17
RUN pyenv install ${PYTHON_VERSION}

# ---- create venv using the pyenv interpreter directly ----
RUN /opt/pyenv/versions/${PYTHON_VERSION}/bin/python -m venv /opt/venv

# ---- activate venv via PATH (no shell activation) ----
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ---- upgrade tooling inside venv ----
RUN /opt/venv/bin/python -m pip install --upgrade \
        pip \
        setuptools \
        wheel

# ---- install compatible TF + AI Benchmark (INSIDE venv) ----
RUN /opt/venv/bin/python -m pip install --no-cache-dir \
        tensorflow==2.10.1 \
        ai-benchmark

COPY ./run_ai_benchmark.py /app/run_ai_benchmark.py

# ---- run benchmark (robust, no console-script dependency) ----
ENTRYPOINT ["/opt/venv/bin/python", "/app/run_ai_benchmark.py"]

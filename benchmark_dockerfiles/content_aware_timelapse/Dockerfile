FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

SESSION_ID=fallback

LABEL gpu_box_benchmark_session=$SESSION_ID

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        ffmpeg \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libffi-dev \
        liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# pyenv
ENV PYENV_ROOT=/opt/pyenv
ENV PATH="${PYENV_ROOT}/bin:${PYENV_ROOT}/shims:${PATH}"

RUN curl -fsSL https://pyenv.run | bash

ENV PYTHON_VERSION=3.10.13
RUN pyenv install ${PYTHON_VERSION} \
    && pyenv global ${PYTHON_VERSION}

# Poetry (installed via official installer)
ENV POETRY_HOME=/opt/poetry
ENV PATH="${POETRY_HOME}/bin:${PATH}"

RUN curl -fsSL https://install.python-poetry.org | python -

# Make Poetry non-interactive + venv-in-project (important for reproducibility)
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

# Clone repo
RUN git clone https://github.com/esologic/content_aware_timelapse.git /app/content_aware_timelapse \
    && cd /app/content_aware_timelapse \
    && git checkout tags/v0.17.4

WORKDIR /app/content_aware_timelapse

# Create venv via project tooling
RUN chmod +x ./tools/create_venv.sh \
    && ./tools/create_venv.sh

ENV VIRTUAL_ENV=/app/content_aware_timelapse/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Copy the bash script into the container
COPY ./run_content_aware_timelapse.sh /app/content_aware_timelapse/run_content_aware_timelapse.sh
RUN chmod +x /app/content_aware_timelapse/run_content_aware_timelapse.sh

ENTRYPOINT ["/app/content_aware_timelapse/run_content_aware_timelapse.sh"]

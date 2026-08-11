ARG USE_GPU=true

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base-true
FROM ubuntu:22.04 AS base-false
FROM base-${USE_GPU} AS base

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl bzip2 ca-certificates \
    libsndfile1 libhdf5-dev \
    procps \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniforge.sh \
    && conda clean -afy

# ── Capa 1: entorno conda sin PyTorch ────────────────────────
# Esta capa se cachea — solo se invalida si cambian los yml
ARG USE_GPU=true
COPY environment.gpu.yml environment.cpu.yml ./
RUN if [ "$USE_GPU" = "true" ]; then \
        mamba env create -f environment.gpu.yml; \
    else \
        mamba env create -f environment.cpu.yml; \
    fi \
    && conda clean -afy

ENV PATH=${CONDA_DIR}/envs/janus-bci/bin:${PATH}
ENV CONDA_DEFAULT_ENV=janus-bci

# ── Capa 2: PyTorch con versiones exactas desde índice oficial ─
# Separado del yml para poder usar --index-url correctamente
ARG USE_GPU=true
RUN if [ "$USE_GPU" = "true" ]; then \
        pip install --no-cache-dir \
            torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
            --index-url https://download.pytorch.org/whl/cu118; \
    else \
        pip install --no-cache-dir \
            torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi

RUN git config --global --add safe.directory /workspace

ENV PYTHONPATH=/workspace
COPY . .
CMD ["bash"]

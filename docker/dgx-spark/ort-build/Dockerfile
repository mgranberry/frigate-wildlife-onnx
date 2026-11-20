FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential git curl wget ca-certificates \
    python3 python3-pip python3-dev python3-setuptools python3-wheel \
    libssl-dev zlib1g-dev \
    software-properties-common gnupg && \
    pip install --no-cache-dir --break-system-packages cmake && \
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/sbsa/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install --no-install-recommends -y \
    cuda-nvcc-13-1 \
    cuda-cudart-dev-13-1 \
    cuda-nvrtc-dev-13-1 \
    cuda-libraries-dev-13-1 \
    libcublas-dev-13-1 \
    libcurand-dev-13-1 \
    libcufft-dev-13-1 \
    libcudnn9-dev-cuda-13 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/local/cuda-13.1/bin:${PATH}"
ENV CUDA_HOME="/usr/local/cuda-13.1"
ENV CUDNN_HOME="/usr/lib/aarch64-linux-gnu"

WORKDIR /workspace

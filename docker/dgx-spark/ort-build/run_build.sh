#!/bin/bash
set -e

git config --global --add safe.directory /workspace/onnxruntime

mkdir -p /usr/local/cuda-13.1/include
cp -s /usr/include/aarch64-linux-gnu/cudnn*.h /usr/local/cuda-13.1/include/ 2>/dev/null || true

pip install --no-cache-dir --break-system-packages \
    pytest packaging wheel setuptools numpy flatbuffers sympy

cd /workspace/onnxruntime

/bin/bash ./build.sh \
    --config Release \
    --update \
    --build \
    --build_wheel \
    --build_shared_lib \
    --parallel "${ORT_BUILD_JOBS:-18}" \
    --use_cuda \
    --cuda_home /usr/local/cuda-13.1 \
    --cudnn_home /usr \
    --cuda_version 13.1 \
    --skip_tests \
    --no_kleidiai \
    --compile_no_warning_as_error \
    --allow_running_as_root \
    --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="120;121" onnxruntime_USE_KLEIDIAI=OFF onnxruntime_USE_FP4_QMOE=OFF CMAKE_CXX_FLAGS="-Wno-error=restrict -Wno-restrict"

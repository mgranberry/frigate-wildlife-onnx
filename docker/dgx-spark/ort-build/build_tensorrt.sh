#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
src="${ORT_TRT_SOURCE_DIR:-$root/artifacts/fp16-tensorrt/onnxruntime-source}"
expected=28225b8583474cf6a6e7f97222227ae668f80e8b
[[ -d "$src/.git" ]] || { echo "Missing isolated ONNX Runtime checkout: $src" >&2; exit 1; }
[[ "$(git -C "$src" rev-parse HEAD)" == "$expected" ]] || { echo 'Unexpected ONNX Runtime revision' >&2; exit 1; }
docker build -f "$root/ort-build/Dockerfile.tensorrt" -t frigate-ort-builder:tensorrt13 "$root/ort-build"
docker run --rm --cpus="${ORT_BUILD_CPUS:-20}" --memory="${ORT_BUILD_MEMORY:-80g}" \
 -e "ORT_BUILD_JOBS=${ORT_BUILD_JOBS:-20}" \
 -v "$src:/workspace/onnxruntime" -v "$root/ort-build/run_build_tensorrt.sh:/run_build_tensorrt.sh:ro" \
 frigate-ort-builder:tensorrt13 /run_build_tensorrt.sh
wheel="$src/build/Linux/Release/dist/onnxruntime_gpu-1.32.0-cp311-cp311-linux_aarch64.whl"
[[ -f "$wheel" ]] || { echo "Wheel not found: $wheel" >&2; exit 1; }
mkdir -p "$root/artifacts/fp16-tensorrt/wheel"
cp "$wheel" "$root/artifacts/fp16-tensorrt/wheel/"
docker build --build-arg BASE_IMAGE=frigate:custom-cuda13 -f "$root/ort-build/Dockerfile.tensorrt-runtime" -t frigate:custom-cuda13-trt "$root/artifacts/fp16-tensorrt/wheel"
echo 'Built frigate:custom-cuda13-trt; run `docker compose build frigate` to rebuild the wildlife image.'

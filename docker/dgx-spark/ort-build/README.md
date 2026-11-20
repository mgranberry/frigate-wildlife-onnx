# Rebuilding the ARM64 CUDA ONNX Runtime wheel

The deployed `frigate:custom-cuda13` image contains a locally compiled
`onnxruntime-gpu` 1.32.0 wheel for CPython 3.11, Linux ARM64 and CUDA 13.1.
It targets CUDA architectures `120;121`, including the Spark GB10. The root
Dockerfile installs this wheel before the wildlife layer adds the classifier.

This directory captures the builder Dockerfile and build script previously in
`../build_ort/`. The source checkout used for the historical build was clean at:

```text
https://github.com/microsoft/onnxruntime.git
28225b8583474cf6a6e7f97222227ae668f80e8b
```

Its build disabled KleidiAI and FP4 QMoE, targeted `120;121`, and relaxed
compiler warnings. See `run_build.sh` for the full invocation. The existing
wheel `onnxruntime_gpu-1.32.0-cp311-cp311-linux_aarch64.whl` has SHA-256
`f2d90dc870b3e801ba5ceab213093e01913d6fa25f4a7ec6827f1047e554b6d3`.
The wheel is intentionally excluded from Git. Preserve it in a separate backup
if exact recovery of the deployed build is important.

## Prepare a source checkout

Run on the ARM64 Spark with Docker, NVIDIA container access, network access,
and ample free disk space. Building the wheel is expensive.

```bash
mkdir -p artifacts
git clone https://github.com/microsoft/onnxruntime.git artifacts/onnxruntime
git -C artifacts/onnxruntime checkout 28225b8583474cf6a6e7f97222227ae668f80e8b
git -C artifacts/onnxruntime submodule update --init --recursive
```

Alternatively point `ORT_SOURCE_DIR` at an existing clean checkout of that
commit, with submodules initialized. The build script verifies the checkout.

## Build and install the wheel in a base image

```bash
./ort-build/build.sh
# Optional: ORT_SOURCE_DIR=/path/to/onnxruntime ORT_BUILD_JOBS=12 ./ort-build/build.sh

# Inspect the wheel emitted under artifacts/onnxruntime/build/Linux/Release/dist/.
# Copy the selected CPython 3.11 ARM64 wheel to this directory under the name
# expected by Dockerfile; back up the existing wheel before replacing it.
docker build -f Dockerfile -t frigate:custom-cuda13 .
docker compose build frigate
```

These commands build images; they do not restart Frigate. `docker compose up -d
--no-build` would deploy the newly built wildlife image. Test a rebuilt image in
isolation before deploying: the wildlife Dockerfile intentionally refuses an
upstream classifier source hash it does not recognize.

## TensorRT-enabled base (`frigate:custom-cuda13-trt`)

To add `TensorrtExecutionProvider` (TensorRT 10.15.1.29 + CUDA 13.1) on top of
`frigate:custom-cuda13` using the same pinned ONNX Runtime commit
(`28225b8583474cf6a6e7f97222227ae668f80e8b`), clone the checkout into
`artifacts/fp16-tensorrt/onnxruntime-source` (or set `ORT_TRT_SOURCE_DIR`) and
run:

```bash
./ort-build/build_tensorrt.sh
python3 wildlife/build_classifier_patch.py
docker compose build frigate
```

`build_tensorrt.sh` builds `frigate-ort-builder:tensorrt13`
(`Dockerfile.tensorrt`), compiles the ARM64 wheel with `--use_tensorrt`
(`run_build_tensorrt.sh`, defaulting to 20 jobs / 80 GiB), copies it to
`artifacts/fp16-tensorrt/wheel/`, and packages `frigate:custom-cuda13-trt`
(`Dockerfile.tensorrt-runtime`). Keep `Dockerfile.tensorrt` and
`run_build_tensorrt.sh` aligned if TensorRT or CUDA versions change.

The checked-in Dockerfile pins the upstream Frigate image digest used by the
current base. CUDA apt packages, Debian packages and Python build dependencies
are not version-locked, so a future clean build may differ or fail. A complete
clean wheel build was **not** rerun while packaging this repository. The
historical wheel checksum above identifies the known working artifact; it is not
a promise that a new build will be byte-identical.

`historical-moe-stubs.c` is preserved from the old build directory as context.
The captured `run_build.sh` does not compile or load it, and it is not part of
the documented wheel build or live image.

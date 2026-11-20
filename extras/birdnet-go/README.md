# BirdNET-Go CUDA 13 + TensorRT FP32 Overlay (DGX Spark / ARM64 Blackwell)

Reproducible patch, build script, and Dockerfile for running [BirdNET-Go](https://github.com/tphakala/birdnet-go) (`20261031`) with `TensorrtExecutionProvider` (FP32 + persistent engine cache) and `CUDAExecutionProvider` fallback on NVIDIA DGX Spark (`aarch64`, CUDA 13.1, TensorRT 10.15, ORT 1.32.0).

## Why TensorRT FP32 (Not FP16 / BF16)?

Unlike vision models, BirdNET v2.4, BirdNET v3.0, and Perch v2 include in-graph audio frontends (STFT/DFT, mel filterbanks, `log`/PCEN) that require FP32 precision. Measured on a real bird audio clip on DGX Spark (`sm_121`):

| Model | CUDA FP32 | TRT FP32 (default) | TRT FP16 | TRT BF16 |
| --- | ---: | ---: | ---: | ---: |
| `BirdNET_V2.4` | `1.85 ms` | **`1.11 ms`** (`1.67x`, top-5 match) | `0.67 ms` (diff `0.97`, top-5 drift) | `0.77 ms` (diff `9.46`, top-1 flip) |
| `BirdNET_V3.0` | `7.71 ms` | **`5.39 ms`** (`1.43x`, top-5 match) | `3.10 ms` (diff `0.44`, top-5 drift) | `4.77 ms` (diff `0.09`, top-5 match) |
| `Perch_V2` | `6.94 ms` | `7.71 ms` (`0.90x`, top-5 match) | `2.34 ms` (diff `5.95`, top-5 drift) | `2.48 ms` (diff `0.42`, top-5 drift) |

`BIRDNET_TENSORRT=1` defaults to lossless **FP32** with persistent engine caching (`trt-cache/`). Set `BIRDNET_TRT_FP16=1` only if you want to experiment with FP16.

## Build & Run

1. Build `frigate:dev-cuda13-trt` first (see `docker/dgx-spark/ort-build/README.md`) so the ARM64 CUDA 13.1 + TensorRT 10.15 + ORT 1.32.0 shared libraries are available locally.
2. Build the patched `birdnet-go-cuda` binary and image:
   ```bash
   # Place build-birdnet.sh, birdnet-cuda.patch, and Dockerfile in a cuda/ directory
   # or run from extras/birdnet-go/:
   BIRDNET_BUILD_DIR=/tmp/birdnet-build-work ./extras/birdnet-go/build-birdnet.sh
   docker build \
     --build-arg CUDA_RUNTIME_IMAGE=frigate:dev-cuda13-trt \
     -f extras/birdnet-go/Dockerfile \
     -t birdnet-go:20261031-trt-ort132 \
     extras/birdnet-go
   ```
3. Start with `extras/birdnet-go/docker-compose.example.yml`. First startup compiles and caches the `.engine` files under `./config/trt-cache/` (~40s total); subsequent restarts load from cache.

## Attribution & License

Derived from [tphakala/birdnet-go](https://github.com/tphakala/birdnet-go) (`MIT` license).

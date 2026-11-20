# Frigate Wildlife ONNX Model Preparation & Verification Scripts

Scripts and Docker overlays for preparing, adapting, baking NHWC `uint8` preprocessing into, and verifying **MegaDetector v6** (`YOLOv10-c` compact and `YOLOv10-e` extra) and **SpeciesNet v4.0.3a (`always_crop`)** ONNX models for [Frigate NVR](https://frigate.video/).

## Pre-Built Hugging Face Models

- **Classifier (Apache-2.0)**: [`mgranberry/speciesnet-v4.0.3a-nhwc-uint8-onnx`](https://huggingface.co/mgranberry/speciesnet-v4.0.3a-nhwc-uint8-onnx)
  - `[1, 480, 480, 3]` `uint8` NHWC input -> `[1, 2498]` `float32` logits (`cast-first` preprocessing baked in for TensorRT / CUDA / OpenVINO).
- **Detector, Compact (AGPL-3.0)**: [`mgranberry/megadetector-v6-yolov10c-320-nhwc-fp16-onnx`](https://huggingface.co/mgranberry/megadetector-v6-yolov10c-320-nhwc-fp16-onnx)
  - `[1, 320, 320, 3]` `uint8` NHWC FP16 (`4.8 MB`, ~2.7M params) adapted to Frigate's `yolo-generic` output (`[1, 7, 300]`, classes `animal`, `person`, `car`). Best for Intel N95/N100, CPU-only, Jetson, or busy multi-camera setups.
- **Detector, Extra (AGPL-3.0)**: [`mgranberry/megadetector-v6-yolov10e-320-nhwc-fp16-onnx`](https://huggingface.co/mgranberry/megadetector-v6-yolov10e-320-nhwc-fp16-onnx)
  - `[1, 320, 320, 3]` `uint8` NHWC FP16 (~31M params) adapted to Frigate's `yolo-generic` output (`[1, 7, 300]`, classes `animal`, `person`, `car`). Best for dedicated NVIDIA or AMD GPUs.

## Why These Graph & Tensor Shapes?

Both models intentionally deviate from standard PyTorch/ONNX conventions (`NCHW float32` inputs and MegaDetector's `[1, 300, 6]` output) to match Frigate's runtime pipeline and let TensorRT/OpenVINO fuse operations:

1. **Baked `NHWC uint8` input with `cast-first` preprocessing (`bake_nhwc_uint8.py`)**:
   - Frigate's frame and crop pipeline natively produces `uint8` NHWC RGB arrays (`[1, H, W, 3]`).
   - Prepending `Cast(uint8 -> float32) -> Div(255.0) -> Transpose(NHWC -> NCHW)` inside the ONNX graph avoids CPU `float32` conversion, `/ 255.0` scaling, and `NHWC -> NCHW` transposes, and lets TensorRT/OpenVINO fuse preprocessing directly into the compiled engine.
   - The `cast-first` order (casting to `float32` before transposing) is deliberate: TensorRT rejects intermediate `uint8` tensors produced by `Transpose(uint8)`.
2. **MegaDetector `[1, 7, 300]` output (`adapt_megadetector.py`)**:
   - Native MegaDetector YOLOv10 outputs `[1, 300, 6]` (`[x1, y1, x2, y2, score, class_id]`).
   - Frigate's built-in `yolo-generic` postprocessor (`post_process_yolo`) expects `[1, 4 + num_classes, N]`, i.e. `[1, 7, 300]` (`[cx, cy, w, h, animal_score, person_score, car_score]` transposed to channels-first).
   - Appending that reshape inside the ONNX graph using `Equal` + `Cast` (instead of `OneHot`, which breaks TensorRT single-engine fusion) lets stock Frigate run MegaDetector with zero custom detector code and no CPU fallback.

## Major Hardware Paths & Setup Instructions

The MegaDetector `yolo-generic` detectors work with stock Frigate. The `classification.custom` ONNX classifier options (`width`, `height`, `input_tensor`, `input_dtype`, `crop_strategy`, `crop_scale`) are not yet in stock Frigate:
- [`mgranberry/frigate` (`feat/custom-cls-multi-backend`)](https://github.com/mgranberry/frigate/tree/feat/custom-cls-multi-backend) provides the upstream-targeted multi-backend ONNX classifier and `crop_strategy` changes.
- `docker/Dockerfile.frigate-wildlife` in this repo layers all 5 patches (`00` through `04`, additionally including YUV subcrop, SpeciesNet geographic rollup, and `reject_labels`) onto Frigate `0.19.0-dev` (`1c30924`).

### Path 1: NVIDIA GPU (x86_64 RTX 20/30/40/50, CUDA / TensorRT)

The default `BASE_IMAGE` in `docker/Dockerfile.frigate-wildlife` (`sha256:11378010...`) is the **standard** Frigate image (CPU / OpenVINO on `amd64`), not the `-tensorrt` variant. For NVIDIA GPUs, pass the matching `1c30924` `-tensorrt` base image (or your own CUDA/TensorRT Frigate base image):

```bash
docker build \
  --build-arg BASE_IMAGE=<frigate-1c30924-tensorrt-image> \
  -f docker/Dockerfile.frigate-wildlife \
  -t frigate:wildlife-trt .
```

- **Recommended models**: `MDV6-yolov10-e` (`320x320`) + `SpeciesNet v4.0.3a` (`480x480`).
- **Classifier device**: Set `device: Tensorrt` (or `AUTO`) under `classification.custom.wildlife`.

### Path 2: Intel iGPU / Arc / N95 / N100 (OpenVINO, `amd64`)

The default `docker/Dockerfile.frigate-wildlife` build (`sha256:11378010...`) already ships OpenVINO on `amd64`:

```bash
docker build -f docker/Dockerfile.frigate-wildlife -t frigate:wildlife .
```

- Pass `/dev/dri:/dev/dri` into the Frigate container and set `device: GPU` (or `AUTO`) under `classification.custom.wildlife`.
- **Recommended models**: **`MDV6-yolov10-c` (`320x320`)** + `SpeciesNet v4.0.3a` (`480x480`). On N95/N100-class hardware, `MDV6-yolov10-e` is ~23x slower on CPU and significantly heavier on the 16-24 EU iGPU.
- **Caveats**: First startup with OpenVINO GPU takes ~15-45s to compile/cache kernels, and the iGPU is shared with VAAPI/QSV hardware video decode.

### Path 3: AMD GPU (ROCm)

Pass the matching `1c30924` `-rocm` Frigate base image:

```bash
docker build \
  --build-arg BASE_IMAGE=<frigate-1c30924-rocm-image> \
  -f docker/Dockerfile.frigate-wildlife \
  -t frigate:wildlife-rocm .
```

### Path 4: CPU-Only (Low-Traffic Wildlife Cameras)

- Use **`MDV6-yolov10-c` (`320x320`)**: runs at **16.6 ms p50** (~60 FPS) even on a 4-core Intel N95 CPU with ONNX Runtime.
- `SpeciesNet v4.0.3a` (`480x480`) runs at ~1.2 s/crop on an N95 CPU (OpenVINO CPU). Because classification only runs on detected `animal` crops rather than every motion frame, this is still workable for quiet cameras with occasional wildlife visits, though a GPU/iGPU is strongly recommended for busier scenes.


### Path 5: NVIDIA DGX Spark (ARM64 Blackwell `sm_121`, CUDA 13 + TensorRT)

Upstream Frigate does not yet publish an `aarch64` CUDA 13 / Blackwell `sm_121` TensorRT image. The `docker/dgx-spark/` directory contains the full builder and runtime Dockerfiles to reproduce the DGX Spark stack:

1. Build the ARM64 CUDA 13.1 + TensorRT 10.15 `onnxruntime-gpu` wheel (`sm_120;121`) and base image (`frigate:dev-cuda13` -> `frigate:dev-cuda13-trt`) following `docker/dgx-spark/ort-build/README.md` and `docker/dgx-spark/Dockerfile.cuda13-base`.
2. Layer the wildlife patches onto `frigate:dev-cuda13-trt`:
   ```bash
   docker build \
     --build-arg BASE_IMAGE=frigate:dev-cuda13-trt \
     -f docker/Dockerfile.frigate-wildlife \
     -t frigate:wildlife-cuda13 .
   ```
3. See `examples/docker-compose.dgx-spark.yml` and `examples/frigate-config.dgx-spark.yaml` for a complete working Compose + Frigate `0.19-0` configuration.

## Optional Extra: BioCAP Wildlife Semantic Search (`docker/biocap/`)

`docker/biocap/` provides an optional OpenAI/`llama.cpp`-compatible `/v1/embeddings` sidecar running [`imageomics/biocap`](https://huggingface.co/imageomics/biocap) (a biology/wildlife-tuned CLIP model) for Frigate's `semantic_search`:

1. Export the ONNX vision/text encoders and tokenizer vocabulary (`export_biocap.py` requires `open_clip_torch`, `torch`, `onnx`):
   ```bash
   python3 docker/biocap/export_biocap.py
   ```
2. Build the sidecar image:
   - **DGX Spark (`TensorrtExecutionProvider`)**:
     ```bash
     docker build --build-arg TRT_LIBS_IMAGE=frigate:dev-cuda13-trt \
       -f docker/biocap/Dockerfile -t frigate-biocap:trt13 docker/biocap
     ```
   - **Generic x86_64 CUDA or CPU**:
     ```bash
     docker build -f docker/biocap/Dockerfile.generic -t frigate-biocap:generic docker/biocap
     ```
3. Wire it into Frigate using the `genai:` and `semantic_search:` blocks in `examples/frigate-config.dgx-spark.yaml` and `examples/docker-compose.dgx-spark.yml`.

### Measured Reference Latencies

Single-stream sequential inference (`batch=1`, warm):

| Hardware / Backend | `MDV6-yolov10-c` 320 FP16 | `MDV6-yolov10-e` 320 FP16 | `SpeciesNet v4.0.3a` 480 |
| --- | ---: | ---: | ---: |
| Intel N95 (4-core Alder Lake-N), ORT CPU (4T) | **16.6 ms** p50 | 1110.3 ms p50 | 1512.0 ms p50 |
| Intel N95 (4-core Alder Lake-N), OpenVINO CPU | **19.8 ms** p50 | 462.2 ms p50 | 1202.2 ms p50 |
| Intel N95 iGPU (16 EU, `0x46d2`), OpenVINO GPU | **21.9 ms** p50 | 182.0 ms p50 | **202.9 ms** p50 |
| NVIDIA GPU (TensorRT FP16) | <2 ms | ~2-4 ms | ~3.4 ms p50 (Blackwell) |

## Example Frigate Configuration

Download the Hugging Face files into:
- `/config/model_cache/megadetector-v6-yolov10c-320/{model.onnx,labels.txt}` (or `megadetector-v6-yolov10e-320/`)
- `/config/model_cache/wildlife/{model.onnx,labelmap.txt}`

```yaml
detectors:
  onnx:
    type: onnx

model:
  model_type: yolo-generic
  path: /config/model_cache/megadetector-v6-yolov10c-320/model.onnx
  labelmap_path: /config/model_cache/megadetector-v6-yolov10c-320/labels.txt
  width: 320
  height: 320
  input_tensor: nhwc
  input_dtype: int
  input_pixel_format: rgb

classification:
  custom:
    wildlife:
      enabled: true
      name: wildlife
      device: AUTO # Use Tensorrt on NVIDIA, GPU on Intel OpenVINO, or CPU
      width: 480
      height: 480
      input_tensor: nhwc
      input_dtype: int
      input_pixel_format: rgb
      threshold: 0.6
      object_config:
        objects:
          - animal
        classification_type: sub_label
        crop_strategy: tight
        crop_scale: 1.0
        # Optional (patch 04 in docker/Dockerfile.frigate-wildlife):
        # reject_labels:
        #   - blank
```

### Optional Patch `04` Features (Geofencing & Blank Rejection)

When running the `docker/Dockerfile.frigate-wildlife` overlay image:
- **Geographic rollup**: Set `FRIGATE_SPECIESNET_COUNTRY` (ISO-3166 alpha-3, e.g. `USA`) and optionally `FRIGATE_SPECIESNET_STATE` (e.g. `TX`) in the Frigate container environment to enable SpeciesNet geofence rollup (out-of-range species roll up to genus/family/order instead of mislabeling).
- **Blank rejection**: Add `reject_labels: [blank]` under `classification.custom.wildlife.object_config` to reject false-positive `animal` tracks that SpeciesNet classifies as `blank`.


## Optional Extra: GPU-Accelerated BirdNET-Go (`extras/birdnet-go/`)

`extras/birdnet-go/` contains the patch, build script, Dockerfile, and sample Compose file for running [BirdNET-Go](https://github.com/tphakala/birdnet-go) (`BirdNET_V2.4`, `BirdNET_V3.0`, `Perch_V2`) with lossless **TensorRT FP32** (persistent engine cache) + **CUDA** fallback on DGX Spark (`extras/birdnet-go/README.md`).

## Repository Contents

- `requirements.txt`: Host Python dependencies (`numpy`, `onnx`, `onnxruntime`) for running scripts outside Docker.
- `bake_nhwc_uint8.py`: Prepends `Cast(uint8 -> float32) -> Div(255.0) -> Transpose(NHWC -> NCHW)` (`--order cast-first`) to an NCHW `float32` ONNX model.
- `adapt_megadetector.py`: Appends ONNX graph ops converting MegaDetector `[1, 300, 6]` (`xyxy`, `score`, `class_id`) output to Frigate `yolo-generic` `[1, 7, 300]` (`cx, cy, w, h, class_scores` transposed).
- `convert_fp16.py`: Converts an ONNX model's weights/activations to FP16 while preserving I/O types (`keep_io_types=True`) and topologically sorting nodes.
- `prepare_models.py`: Downloads and verifies SHA-256 hashes of SPARROW `MDV6-yolov10-c.zip` + `SpeciesNet-Crop.zip` (`v4.0.3a`), adapts MegaDetector, and bakes NHWC `uint8` into SpeciesNet.
- `prepare_extra_model.py`: Exports `MDV6-yolov10-e.pt` at `320x320` (or `640x640`), adapts it with `adapt_megadetector.py`, converts to FP16, bakes NHWC `uint8`, and validates parity against the unadapted ONNX model.
- `prepare_speciesnet_labels.py`: Generates the 2,498-line readable `speciesnet-labelmap.txt` matching classifier output order.
- `wildlife_speciesnet/` & `test_geofence.py`: Isolated SpeciesNet geographic rollup & geofence postprocessing + unit tests (`python3 test_geofence.py -v`).
- `test_adapter.py`: Validates adapted MegaDetector ONNX outputs against the unadapted model inside a Frigate container.
- `docker/`: Reproducible `Dockerfile.frigate-wildlife` (with `patches/` and `apply_patches.py`), `Dockerfile.export`, `docker/dgx-spark/` (ARM64 Blackwell CUDA 13 + TensorRT ORT wheel builder & base Dockerfile), and `docker/biocap/` (optional BioCAP embeddings sidecar).
- `examples/`: Sample `docker-compose.dgx-spark.yml` and `frigate-config.dgx-spark.yaml`.
- `extras/birdnet-go/`: Optional BirdNET-Go TensorRT FP32 / CUDA 13 build files and sample Compose.

## Quick Usage (Reproducing Models)

Install host dependencies (`pip install -r requirements.txt`), then:

```bash
python3 bake_nhwc_uint8.py \
  SpeciesNet-Crop/model.onnx \
  model.onnx \
  --order cast-first
```

To build the export container for `prepare_extra_model.py`:

```bash
docker build -f docker/Dockerfile.export -t frigate-wildlife-export .
```

## Attributions & Licenses

- **SpeciesNet (`model.onnx`, `wildlife_speciesnet/`)**: Derived from [Google Camera Trap AI (`google/cameratrapai`)](https://github.com/google/cameratrapai) via the [SPARROW Zenodo bundle (`SpeciesNet-Crop.zip`, v4.0.3a `always_crop`)](https://zenodo.org/records/20563673), with taxonomy/geofence assets from [Addax-Data-Science/SPECIESNET-v4-0-2-A](https://huggingface.co/Addax-Data-Science/SPECIESNET-v4-0-2-A). Licensed under **Apache-2.0** (`LICENSE-APACHE`).
- **MegaDetector v6 YOLOv10-c / YOLOv10-e (`adapt_megadetector.py`, `prepare_extra_model.py`, exported weights)**: Derived from [MegaDetector (`agentmorris/MegaDetector`)](https://github.com/agentmorris/MegaDetector) / [SPARROW (`MDV6-yolov10-c.pt`, `MDV6-yolov10-e.pt`)](https://zenodo.org/records/18177050) (Ultralytics YOLOv10). Licensed under **AGPL-3.0** (`LICENSE-AGPL-3.0`).
- **Frigate NVR**: Designed for [blakeblackshear/frigate](https://github.com/blakeblackshear/frigate).

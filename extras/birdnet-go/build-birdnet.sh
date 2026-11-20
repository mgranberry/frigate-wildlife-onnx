#!/usr/bin/env bash
set -euo pipefail
# Rebuild the local CUDA-patched BirdNET-Go application. Does not deploy.
script_dir="$(cd "$(dirname "$0")" project_dir="$(cd "$(dirname "$0")/.." && pwd)"project_dir="$(cd "$(dirname "$0")/.." && pwd)" pwd)"
work_dir="${BIRDNET_BUILD_DIR:-$script_dir/build-work}"
mkdir -p "$work_dir"
if [[ ! -d "$work_dir/birdnet/.git" ]]; then
  git clone --branch 20261031 --depth 1 https://github.com/tphakala/birdnet-go.git "$work_dir/birdnet"
fi
[[ "$(git -C "$work_dir/birdnet" rev-parse HEAD)" == bc1e0de67d9a2292b4742a79f868a7720f8e6a12 ]] || { echo 'Unexpected BirdNET-Go revision' >&2; exit 1; }
if git -C "$work_dir/birdnet" diff --quiet; then
  git -C "$work_dir/birdnet" apply "$script_dir/birdnet-cuda.patch"
else
  git -C "$work_dir/birdnet" apply --reverse --check "$script_dir/birdnet-cuda.patch" || { echo 'Unexpected source changes' >&2; exit 1; }
fi
if [[ ! -f "$work_dir/tensorflow/tensorflow/lite/c/c_api.h" ]]; then
  git clone --branch v2.17.1 --filter=blob:none --no-checkout --depth 1 https://github.com/tensorflow/tensorflow.git "$work_dir/tensorflow"
  git -C "$work_dir/tensorflow" sparse-checkout set --no-cone '**/*.h'
  git -C "$work_dir/tensorflow" checkout
fi
if [[ ! -f "$work_dir/openvino/include/openvino/c/openvino.h" ]]; then
  mkdir -p "$work_dir/openvino"
  archive="$work_dir/openvino/openvino.tgz"
  curl -fsSL https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.2/linux/openvino_toolkit_ubuntu22_2026.2.0.21903.52ddc073857_arm64.tgz -o "$archive"
  echo "8ce45467967e22fddb83a6b72a8bd1f9bfa6f43351e1ca2eaf5251064fe17767  $archive" | sha256sum -c -
  tar -xzf "$archive" -C "$work_dir/openvino" --strip-components=2 openvino_toolkit_ubuntu22_2026.2.0.21903.52ddc073857_arm64/runtime/include
fi
(cd "$work_dir/birdnet/frontend" && npm ci --no-audit --no-fund && npm run build)
mkdir -p "$work_dir/lib" "$work_dir/go-cache" "$work_dir/go-build-cache"
# Use the identical TFLite library shipped in the pinned BirdNET-Go image.
container_id="$(docker create ghcr.io/tphakala/birdnet-go@sha256:5eebc826c137a78aa7f91f0a4f3a02d9775f1a19d720f63b309ec5ae608b8698)"
trap 'docker rm -f "$container_id" >/dev/null 2>&1 || true' EXIT
docker cp -L "$container_id:/usr/lib/libtensorflowlite_c.so.2.17.1" "$work_dir/lib/libtensorflowlite_c.so"
docker run --rm --cpus "${BIRDNET_BUILD_CPUS:-8}" -v "$work_dir/birdnet:/src" -v "$work_dir/go-cache:/go" -v "$work_dir/go-build-cache:/root/.cache/go-build" -v "$work_dir/tensorflow:/tf:ro" -v "$work_dir/openvino/include:/ov:ro" -v "$work_dir/lib:/liblocal:ro" -w /src golang:1.27-trixie bash -c 'CGO_CFLAGS="-I/tf -I/ov" CGO_LDFLAGS="-L/liblocal -ltensorflowlite_c" CGO_ENABLED=1 go build -buildvcs=false -tags noembed,openvino -ldflags "-s -w -X main.version=20261031-cuda-local" -o /src/birdnet-go-cuda .'
cp "$work_dir/birdnet/birdnet-go-cuda" "$script_dir/birdnet-go-cuda"

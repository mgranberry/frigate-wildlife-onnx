#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ort_source="${ORT_SOURCE_DIR:-$repo_dir/artifacts/onnxruntime}"
expected=28225b8583474cf6a6e7f97222227ae668f80e8b
if [[ ! -d "$ort_source/.git" ]]; then
  printf 'ONNX Runtime checkout not found: %s\nSee ort-build/README.md for setup.\n' "$ort_source" >&2
  exit 1
fi
ort_source="$(cd -- "$ort_source" && pwd)"
actual="$(git -C "$ort_source" rev-parse HEAD)"
if [[ "$actual" != "$expected" ]]; then
  printf 'Unexpected ONNX Runtime revision: %s (expected %s)\n' "$actual" "$expected" >&2
  exit 1
fi
if [[ -n "$(git -C "$ort_source" status --porcelain --untracked-files=no)" ]]; then
  printf 'Tracked source changes detected; use a clean pinned checkout.\n' >&2
  exit 1
fi
if git -C "$ort_source" submodule status --recursive | grep -qE '^[-+U]'; then
  printf 'Submodules are missing, conflicted or at unexpected revisions.\n' >&2
  exit 1
fi
docker build -t frigate-ort-builder:cuda13 -f "$repo_dir/ort-build/Dockerfile" "$repo_dir/ort-build"
docker run --rm --gpus all \
  -e "ORT_BUILD_JOBS=${ORT_BUILD_JOBS:-18}" \
  -v "$ort_source:/workspace/onnxruntime" \
  -v "$repo_dir/ort-build/run_build.sh:/run_build.sh:ro" \
  frigate-ort-builder:cuda13 /run_build.sh
printf 'Build complete. Wheel output: %s/build/Linux/Release/dist/\n' "$ort_source"

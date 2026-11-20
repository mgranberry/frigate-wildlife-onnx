#!/usr/bin/env python3
"""Rebuild the MDV6 YOLOv10-e ONNX (default 320x320 NHWC uint8 FP16) and Frigate adapter."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL = "https://zenodo.org/api/records/18177050/files/MDV6-yolov10-e.pt/content"
SHA = "062576a5808a021154db095e6808fc3d96af4388cd394bb7654b9cc3759a0690"
IMAGE = "frigate-wildlife-export"
PACKAGES = [
    "ultralytics==8.3.0",
    "onnx==1.23.2",
    "onnxscript==0.7.2",
    "onnx-ir==1.1.0",
    "numpy==2.5.5",
    "matplotlib==3.11.2",
    "opencv-python-headless==5.0.0.93",
    "scipy==1.18.1",
    "pandas==3.1.1",
    "seaborn==0.13.2",
    "psutil==7.2.2",
    "requests==2.34.2",
    "PyYAML==6.0.3",
]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--size", type=int, default=320, help="Export square input size (default: 320)"
    )
    p.add_argument(
        "--weights",
        type=Path,
        help="Existing verified .pt file; otherwise download from Zenodo",
    )
    p.add_argument(
        "--deps",
        type=Path,
        help="Existing preinstalled export-deps dir for offline export",
    )
    p.add_argument(
        "--test-image",
        required=True,
        type=Path,
        help="Image containing an animal for adapter validation",
    )
    p.add_argument(
        "--export-image",
        default=IMAGE,
        help="Container image with PyTorch, ultralytics, onnx, and onnxruntime (see docker/Dockerfile.export)",
    )
    p.add_argument(
        "--frigate-image",
        default="ghcr.io/blakeblackshear/frigate:stable-tensorrt",
        help="Frigate container image with onnxruntime/CUDA for test_adapter.py validation",
    )
    a = p.parse_args()

    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    test = a.test_image.resolve(strict=True)
    sz = a.size
    names = [
        "MDV6-yolov10-e.pt",
        "MDV6-yolov10-e.onnx",
        f"megadetector-v6-extra-{sz}-frigate.onnx",
        f"megadetector-v6-extra-{sz}-nchw-fp16.onnx",
        f"megadetector-v6-extra-{sz}-nhwc-fp16.onnx",
        "manifest.json",
    ]
    if any((out / n).exists() for n in names):
        raise FileExistsError("Output files already exist; choose a new directory")

    with tempfile.TemporaryDirectory(prefix="frigate-extra-export-") as td:
        tmp = Path(td)
        weights = tmp / names[0]
        if a.weights:
            shutil.copyfile(a.weights, weights)
        else:
            with urllib.request.urlopen(URL, timeout=180) as r, weights.open("wb") as f:
                shutil.copyfileobj(r, f)
        if digest(weights) != SHA:
            raise ValueError("Upstream weights SHA256 mismatch")

        deps_mount = []
        pythonpath_env = []
        if a.deps:
            deps_mount = ["-v", f"{a.deps.resolve(strict=True)}:/work/deps:ro"]
            pythonpath_env = ["-e", "PYTHONPATH=/work/deps"]
        else:
            probe = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--user",
                    f"{os.getuid()}:{os.getgid()}",
                    "-e",
                    "HOME=/tmp",
                    "--entrypoint",
                    "python3",
                    a.export_image,
                    "-c",
                    "import torch, ultralytics, onnx, onnxruntime",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if probe.returncode != 0:
                # Export image lacks required packages; install pinned toolchain into /work/deps.
                base = [
                    "docker",
                    "run",
                    "--rm",
                    "--user",
                    f"{os.getuid()}:{os.getgid()}",
                    "-e",
                    "HOME=/tmp",
                    "-v",
                    f"{tmp}:/work",
                    "--entrypoint",
                    "python3",
                    a.export_image,
                ]
                subprocess.run(
                    base
                    + [
                        "-m",
                        "pip",
                        "install",
                        "--quiet",
                        "--target",
                        "/work/deps",
                        "--no-deps",
                        *PACKAGES[:4],
                    ],
                    check=True,
                )
                subprocess.run(
                    base
                    + [
                        "-m",
                        "pip",
                        "install",
                        "--quiet",
                        "--target",
                        "/work/deps",
                        *PACKAGES[4:],
                    ],
                    check=True,
                )
                pythonpath_env = ["-e", "PYTHONPATH=/work/deps"]

        code = (
            "import torch; print('torch', torch.__version__, flush=True); "
            "from ultralytics import YOLO; "
            f"YOLO('/work/MDV6-yolov10-e.pt').export(format='onnx', imgsz={sz}, opset=18, simplify=False, device='cpu')"
        )
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--network",
                "none",
                "-e",
                "HOME=/tmp",
                *pythonpath_env,
                "-e",
                "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1",
                "-e",
                "YOLO_CONFIG_DIR=/tmp/ultralytics",
                "-v",
                f"{tmp}:/work",
                *deps_mount,
                "--entrypoint",
                "python3",
                a.export_image,
                "-c",
                code,
            ],
            check=True,
        )

        # Run graph-editing steps in the export image (stock Frigate images do not ship the onnx package).
        run = [
            "docker",
            "run",
            "--rm",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--network",
            "none",
            "-e",
            "HOME=/tmp",
            *pythonpath_env,
            "-v",
            f"{tmp}:/work",
            *deps_mount,
            "-v",
            f"{HERE}:/integration:ro",
            "--entrypoint",
            "python3",
            a.export_image,
        ]
        subprocess.run(
            run
            + [
                "/integration/adapt_megadetector.py",
                "/work/" + names[1],
                "/work/" + names[2],
            ],
            check=True,
        )
        subprocess.run(
            run
            + [
                "/integration/convert_fp16.py",
                "/work/" + names[2],
                "/work/" + names[3],
            ],
            check=True,
        )
        subprocess.run(
            run
            + [
                "/integration/bake_nhwc_uint8.py",
                "--order",
                "cast-first",
                "/work/" + names[3],
                "/work/" + names[4],
            ],
            check=True,
        )

        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--network",
                "none",
                "--gpus",
                "all",
                "-v",
                f"{tmp}:/work:ro",
                "-v",
                f"{HERE}:/integration:ro",
                "-v",
                f"{test}:/test/image:ro",
                "--entrypoint",
                "python3",
                a.frigate_image,
                "/integration/test_adapter.py",
                "--original",
                "/work/" + names[1],
                "--adapted",
                "/work/" + names[4],
                "--images",
                "/test/image",
            ],
            check=True,
        )

        manifest = {
            "weight_url": URL,
            "weight_sha256": SHA,
            "export_image": a.export_image,
            "export_packages": PACKAGES,
            "size": sz,
            "input_shape": [1, sz, sz, 3],
            "input_dtype": "uint8",
            "opset": 18,
            "files": {n: digest(tmp / n) for n in names[:5]},
            "validation_image_sha256": digest(test),
            "note": "ONNX metadata can vary between builds; validate predictions, not file identity.",
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for n in names:
            shutil.copyfile(tmp / n, out / n)
    print("Prepared and validated:", out)


if __name__ == "__main__":
    main()

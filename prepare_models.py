#!/usr/bin/env python3
"""Fetch verified wildlife weights and produce a local Frigate model directory.

This creates artifacts only; it does not change a running Frigate instance.
"""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
ARCHIVES = {
    "MDV6-yolov10-c.zip": (
        "https://zenodo.org/records/20563673/files/MDV6-yolov10-c.zip?download=1",
        "e92436614c7e21650961527ca320ca2f3ae2d06f8f8ce1397777ea407062dd81",
    ),
    "SpeciesNet-Crop.zip": (
        "https://zenodo.org/records/20563673/files/SpeciesNet-Crop.zip?download=1",
        "55ba1b862c494abc8f46ced18d373383b1e3bba6d84fd4e4eec8175c6b1f90a0",
    ),
}
EXPECTED = {
    "megadetector-v6-frigate.onnx": "4ace6be6dacec8c83530aa265e798e4daa4f2137367aeaf03e1ee09a2ec383e8",
    "wildlife/model.onnx": "72e37a16f4f3c6239cba741191ca59a127fe4f38fb26bfb44e5b277e86ac6899",
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New or existing local model-cache directory",
    )
    p.add_argument(
        "--cache",
        type=Path,
        help="Directory of previously downloaded archives, if available",
    )
    p.add_argument(
        "--image",
        default=None,
        help="Optional Docker image containing onnx; runs with local python3 if omitted",
    )
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="frigate-wildlife-models-") as tmpname:
        tmp = Path(tmpname)
        inputs = {}
        for name, (url, checksum) in ARCHIVES.items():
            candidate = args.cache / name if args.cache else None
            path = candidate if candidate and candidate.is_file() else tmp / name
            if path == tmp / name:
                print("Downloading", name, flush=True)
                with (
                    urllib.request.urlopen(url, timeout=120) as response,
                    path.open("wb") as dest,
                ):
                    shutil.copyfileobj(response, dest)
            if sha(path) != checksum:
                raise ValueError(f"Checksum mismatch for {name}")
            inputs[name] = path
        original = tmp / "detector-original.onnx"
        species_raw = tmp / "speciesnet-raw.onnx"
        species = tmp / "speciesnet-nhwc.onnx"
        for archive, member, destination in [
            ("MDV6-yolov10-c.zip", "MDV6-yolov10-c/1/model.onnx", original),
            ("SpeciesNet-Crop.zip", "SpeciesNet-Crop/model.onnx", species_raw),
        ]:
            with (
                zipfile.ZipFile(inputs[archive]) as z,
                z.open(member) as src,
                destination.open("wb") as dst,
            ):
                shutil.copyfileobj(src, dst)
        if sha(species_raw) != EXPECTED["wildlife/model.onnx"]:
            raise ValueError(
                "Extracted SpeciesNet ONNX does not match expected checksum"
            )
        adapted = tmp / "megadetector-v6-frigate.onnx"

        def run_tool(script_name, *script_args):
            if args.image:
                cmd = [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--user",
                    f"{os.getuid()}:{os.getgid()}",
                    "-v",
                    f"{tmp}:/work",
                    "-v",
                    f"{HERE}:/integration:ro",
                    "--entrypoint",
                    "python3",
                    args.image,
                    f"/integration/{script_name}",
                    *script_args,
                ]
            else:
                mapped = [a.replace("/work/", str(tmp) + "/") for a in script_args]
                cmd = [sys.executable, str(HERE / script_name), *mapped]
            subprocess.run(cmd, check=True)

        run_tool(
            "adapt_megadetector.py",
            "/work/detector-original.onnx",
            "/work/megadetector-v6-frigate.onnx",
        )
        run_tool(
            "bake_nhwc_uint8.py",
            "--order",
            "cast-first",
            "/work/speciesnet-raw.onnx",
            "/work/speciesnet-nhwc.onnx",
        )
        items = [
            (
                adapted,
                output / "megadetector-v6-frigate.onnx",
                EXPECTED["megadetector-v6-frigate.onnx"],
            ),
            (species, output / "wildlife/model.onnx", sha(species)),
            (
                HERE / "labels.txt",
                output / "megadetector-labels.txt",
                sha(HERE / "labels.txt"),
            ),
            (
                HERE / "speciesnet-labelmap.txt",
                output / "wildlife/labelmap.txt",
                sha(HERE / "speciesnet-labelmap.txt"),
            ),
        ]
        for source, dest, checksum in items:
            if sha(source) != checksum:
                raise ValueError(
                    f"Generated model does not match expected checksum: {source.name}"
                )
            if dest.exists() and sha(dest) != checksum:
                raise FileExistsError(f"Refusing to overwrite different file: {dest}")
        for source, dest, _ in items:
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, dest)
            print("Ready:", dest)


if __name__ == "__main__":
    main()

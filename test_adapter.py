"""Compare native MegaDetector output with Frigate's real YOLO postprocessor."""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, "/opt/frigate")
from frigate.util.model import post_process_yolo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    original = ort.InferenceSession(
        str(args.original), sess_options=opts, providers=providers
    )
    adapted = ort.InferenceSession(
        str(args.adapted), sess_options=opts, providers=providers
    )
    if "CUDAExecutionProvider" not in adapted.get_providers():
        raise RuntimeError("CUDA not available")

    orig_in = original.get_inputs()[0]
    adap_in = adapted.get_inputs()[0]
    sz = int(orig_in.shape[2])
    is_nhwc_uint8 = adap_in.type == "tensor(uint8)"
    expected_shape = [1, sz, sz, 3] if is_nhwc_uint8 else [1, 3, sz, sz]
    if list(adap_in.shape) != expected_shape:
        raise AssertionError(
            f"Unexpected adapted input shape {adap_in.shape}, expected {expected_shape}"
        )

    counts = {}
    start = time.monotonic()
    for image_path in args.images:
        image = Image.open(image_path).convert("RGB")
        scale = min(sz / image.width, sz / image.height)
        size = (round(image.width * scale), round(image.height * scale))
        canvas = Image.new("RGB", (sz, sz), (114, 114, 114))
        canvas.paste(
            image.resize(size, Image.Resampling.BILINEAR),
            ((sz - size[0]) // 2, (sz - size[1]) // 2),
        )
        x_u8 = np.asarray(canvas, dtype=np.uint8)[None]
        x_f32 = x_u8.astype(np.float32).transpose(0, 3, 1, 2) / 255
        raw = original.run(None, {orig_in.name: x_f32})[0][0]
        transformed = adapted.run(
            None, {adap_in.name: x_u8 if is_nhwc_uint8 else x_f32}
        )[0]
        predictions = post_process_yolo([transformed], sz, sz)
        positives = raw[np.where(raw[:, 5] == 0, raw[:, 4] > 0.15, raw[:, 4] > 0.4)]
        if len(positives):
            boxes = positives[:, :4].copy()
            boxes[:, 2:] -= boxes[:, :2]
            indices = cv2.dnn.NMSBoxes(boxes, positives[:, 4], 0.15, 0.4).flatten()[:20]
            selected = positives[indices]
        else:
            selected = np.zeros((0, 6), dtype=np.float32)
        actual = predictions[predictions[:, 1] > 0]
        if len(selected) != len(actual):
            raise AssertionError(
                f"Detection count mismatch on {image_path}: {len(selected)} vs {len(actual)}"
            )
        for i, (x1, y1, x2, y2, score, cls) in enumerate(selected):
            expected = np.array([cls, score, y1 / sz, x1 / sz, y2 / sz, x2 / sz])
            if not np.allclose(actual[i], expected, atol=args.atol):
                raise AssertionError(
                    f"Mismatch on {image_path}: {actual[i]} vs {expected}"
                )
            counts[int(cls)] = counts.get(int(cls), 0) + 1
        print(image_path.name, len(actual), "matches", flush=True)
    if not counts:
        raise AssertionError("No detections tested")
    print(
        json.dumps(
            {
                "size": sz,
                "adapted_input": {"shape": list(adap_in.shape), "type": adap_in.type},
                "frames": len(args.images),
                "matched_detections": counts,
                "seconds": round(time.monotonic() - start, 2),
                "providers": adapted.get_providers(),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

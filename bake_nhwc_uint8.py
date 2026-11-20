#!/usr/bin/env python3
"""Bake NHWC -> NCHW Transpose and uint8 -> float32 / 255.0 normalization into an ONNX graph."""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--order",
        choices=["cast-first", "transpose-first"],
        default="cast-first",
        help="cast-first avoids intermediate uint8 tensors rejected by TensorRT",
    )
    args = parser.parse_args()

    model = onnx.load(str(args.source))
    graph = model.graph
    if len(graph.input) != 1:
        raise ValueError(f"Expected 1 input, got {len(graph.input)}")

    orig_input = graph.input[0]
    orig_name = orig_input.name
    dims = [d.dim_value for d in orig_input.type.tensor_type.shape.dim]
    if len(dims) != 4 or dims[0] != 1 or dims[1] != 3:
        raise ValueError(f"Expected NCHW [1, 3, H, W] input, got {dims}")
    if orig_input.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise ValueError(
            f"Expected FLOAT input, got {orig_input.type.tensor_type.elem_type}"
        )

    height, width = dims[2], dims[3]
    prefix = "frigate_nhwc_prep_"
    if any(n.name.startswith(prefix) for n in graph.node):
        raise ValueError("Graph already has NHWC preprocessing baked in")

    nchw_float = prefix + "nchw_float"
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp == orig_name:
                node.input[i] = nchw_float

    scale_name = prefix + "scale_255"
    graph.initializer.append(
        numpy_helper.from_array(np.array(255.0, dtype=np.float32), scale_name)
    )

    if args.order == "cast-first":
        nhwc_cast = prefix + "nhwc_cast"
        nhwc_norm = prefix + "nhwc_norm"
        prep_nodes = [
            helper.make_node(
                "Cast",
                [orig_name],
                [nhwc_cast],
                name=prefix + "cast",
                to=TensorProto.FLOAT,
            ),
            helper.make_node(
                "Div", [nhwc_cast, scale_name], [nhwc_norm], name=prefix + "div255"
            ),
            helper.make_node(
                "Transpose",
                [nhwc_norm],
                [nchw_float],
                name=prefix + "transpose",
                perm=[0, 3, 1, 2],
            ),
        ]
    else:
        nchw_uint8 = prefix + "nchw_uint8"
        nchw_cast = prefix + "nchw_cast"
        prep_nodes = [
            helper.make_node(
                "Transpose",
                [orig_name],
                [nchw_uint8],
                name=prefix + "transpose",
                perm=[0, 3, 1, 2],
            ),
            helper.make_node(
                "Cast",
                [nchw_uint8],
                [nchw_cast],
                name=prefix + "cast",
                to=TensorProto.FLOAT,
            ),
            helper.make_node(
                "Div", [nchw_cast, scale_name], [nchw_float], name=prefix + "div255"
            ),
        ]

    for idx, node in enumerate(prep_nodes):
        graph.node.insert(idx, node)

    new_input = helper.make_tensor_value_info(
        orig_name, TensorProto.UINT8, [1, height, width, 3]
    )
    del graph.input[:]
    graph.input.append(new_input)

    onnx.checker.check_model(model)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.destination))
    print(
        f"Saved {args.destination} ({args.order}) with input [1, {height}, {width}, 3] uint8",
        flush=True,
    )


if __name__ == "__main__":
    main()

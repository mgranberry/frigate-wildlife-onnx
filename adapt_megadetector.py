"""Adapt MegaDetector v6 end-to-end ONNX outputs for Frigate's yolo-generic parser.

No model weights are changed. Original [x1,y1,x2,y2,score,class] predictions
become [cx,cy,w,h,animal_score,person_score,vehicle_score] per candidate, using
Equal + Cast instead of OneHot so TensorRT compiles the entire graph into one engine.
"""

import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    args = p.parse_args()

    model = onnx.load(str(args.source))
    graph = model.graph
    if len(graph.output) != 1:
        raise ValueError("Expected one detector output")
    original = graph.output[0].name
    expected = [d.dim_value for d in graph.output[0].type.tensor_type.shape.dim]
    if expected != [1, 300, 6]:
        raise ValueError(f"Unexpected MegaDetector output shape: {expected}")
    prefix = "frigate_adapter_"
    if any(n.name.startswith(prefix) for n in graph.node):
        raise ValueError("Already adapted")

    def const(name, value):
        graph.initializer.append(
            numpy_helper.from_array(np.asarray(value), prefix + name)
        )
        return prefix + name

    ix = const("x_indices", np.array([0, 1], np.int64))
    iy = const("y_indices", np.array([2, 3], np.int64))
    iscore = const("score_index", np.array([4], np.int64))
    iclass = const("class_index", np.array([5], np.int64))
    half = const("half", np.array(0.5, np.float32))
    class_ids = const("class_ids", np.array([[[0, 1, 2]]], np.int64))
    nodes = []

    def add(op, inputs, output, **kw):
        nodes.append(
            helper.make_node(op, inputs, [prefix + output], name=prefix + output, **kw)
        )
        return prefix + output

    p1 = add("Gather", [original, ix], "p1", axis=2)
    p2 = add("Gather", [original, iy], "p2", axis=2)
    center = add("Mul", [add("Add", [p1, p2], "sum"), half], "center")
    size = add("Sub", [p2, p1], "size")
    score = add("Gather", [original, iscore], "score", axis=2)
    cls = add(
        "Cast",
        [add("Gather", [original, iclass], "class_float", axis=2)],
        "class_int",
        to=TensorProto.INT64,
    )
    class_eq = add("Equal", [cls, class_ids], "class_eq")
    onehot = add("Cast", [class_eq], "onehot", to=TensorProto.FLOAT)
    scores = add("Mul", [onehot, score], "class_scores")
    combined = add("Concat", [center, size, scores], "candidates", axis=2)
    output = add("Transpose", [combined], "output", perm=[0, 2, 1])

    graph.node.extend(nodes)
    del graph.output[:]
    graph.output.append(
        helper.make_tensor_value_info(output, TensorProto.FLOAT, [1, 7, 300])
    )
    onnx.checker.check_model(model)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.destination))
    print(f"Saved {args.destination}", flush=True)


if __name__ == "__main__":
    main()

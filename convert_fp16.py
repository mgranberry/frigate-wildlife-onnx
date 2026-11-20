#!/usr/bin/env python3
"""Convert an ONNX graph to FP16 while preserving FP32 model I/O."""

import argparse, heapq
import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("input")
p.add_argument("output")
a = p.parse_args()
m = convert_float_to_float16(onnx.load(a.input), keep_io_types=True)
g = m.graph
nodes = list(g.node)
# ORT's converter can append graph-input casts after their consumers.
producer = {o: i for i, n in enumerate(nodes) for o in n.output if o}
deps = [set(producer[x] for x in n.input if x in producer) for n in nodes]
children = [[] for _ in nodes]
for i, ds in enumerate(deps):
    for d in ds:
        children[d].append(i)
queue = [i for i, ds in enumerate(deps) if not ds]
heapq.heapify(queue)
ordered = []
while queue:
    i = heapq.heappop(queue)
    ordered.append(nodes[i])
    for child in children[i]:
        deps[child].remove(i)
        if not deps[child]:
            heapq.heappush(queue, child)
if len(ordered) != len(nodes):
    raise RuntimeError("Cannot sort graph; possible cycle")
del g.node[:]
g.node.extend(ordered)
onnx.checker.check_model(m)
onnx.save(m, a.output)
print("converted", len(ordered), "nodes, preserving input/output types")

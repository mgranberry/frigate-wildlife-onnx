ARG TRT_LIBS_IMAGE=frigate:dev-cuda13-trt
FROM ${TRT_LIBS_IMAGE} AS trt-libs

FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update -qq \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Copy CUDA 13.1, cuDNN 9, and TensorRT 10.15 runtime shared libraries (sm_120/ptx only)
COPY --from=trt-libs /usr/local/cuda-13.1/targets/sbsa-linux/lib/libcudart.so.13* /usr/local/lib/
COPY --from=trt-libs /usr/local/cuda-13.1/targets/sbsa-linux/lib/libcublas.so.13* /usr/local/lib/
COPY --from=trt-libs /usr/local/cuda-13.1/targets/sbsa-linux/lib/libcublasLt.so.13* /usr/local/lib/
COPY --from=trt-libs /usr/lib/aarch64-linux-gnu/libcudnn*.so.9* /usr/lib/aarch64-linux-gnu/
COPY --from=trt-libs /usr/lib/aarch64-linux-gnu/libnvinfer.so.10* /usr/lib/aarch64-linux-gnu/
COPY --from=trt-libs /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.10* /usr/lib/aarch64-linux-gnu/
COPY --from=trt-libs /usr/lib/aarch64-linux-gnu/libnvinfer_builder_resource_sm120.so.10.15.1 /usr/lib/aarch64-linux-gnu/
COPY --from=trt-libs /usr/lib/aarch64-linux-gnu/libnvinfer_builder_resource_ptx.so.10.15.1 /usr/lib/aarch64-linux-gnu/
COPY --from=trt-libs /usr/lib/aarch64-linux-gnu/libnvonnxparser.so.10* /usr/lib/aarch64-linux-gnu/

# Copy the locally compiled ARM64 Blackwell (sm_120;121) ONNX Runtime GPU + TensorRT package
COPY --from=trt-libs /usr/local/lib/python3.11/dist-packages/onnxruntime /usr/local/lib/python3.11/site-packages/onnxruntime
COPY --from=trt-libs /usr/local/lib/python3.11/dist-packages/onnxruntime_gpu-1.32.0.dist-info /usr/local/lib/python3.11/site-packages/onnxruntime_gpu-1.32.0.dist-info

RUN ldconfig \
 && python3 -m pip install --no-cache-dir \
      numpy pillow ftfy regex fastapi uvicorn packaging flatbuffers protobuf sympy coloredlogs

WORKDIR /app
COPY server.py /app/server.py

EXPOSE 8090
CMD ["python3", "/app/server.py"]

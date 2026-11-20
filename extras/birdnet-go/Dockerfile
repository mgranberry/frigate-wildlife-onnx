# Local experimental build for the DGX Spark (CUDA 13 + TensorRT 10.15 + ORT 1.32.0).
ARG CUDA_RUNTIME_IMAGE=frigate:dev-cuda13-trt
FROM ${CUDA_RUNTIME_IMAGE} AS cuda-runtime
FROM ghcr.io/tphakala/birdnet-go@sha256:5eebc826c137a78aa7f91f0a4f3a02d9775f1a19d720f63b309ec5ae608b8698
USER root
COPY --from=cuda-runtime /usr/local/cuda-13.1/targets/sbsa-linux/lib/ /opt/cuda/lib/
COPY --from=cuda-runtime /usr/lib/aarch64-linux-gnu/libcudnn*.so* /opt/cuda/lib/
COPY --from=cuda-runtime /usr/lib/aarch64-linux-gnu/libnvinfer*.so* /opt/cuda/lib/
COPY --from=cuda-runtime /usr/lib/aarch64-linux-gnu/libnvonnxparser*.so* /opt/cuda/lib/
COPY --from=cuda-runtime /usr/local/lib/python3.11/dist-packages/onnxruntime/capi/libonnxruntime.so.1.32.0 /opt/cuda/lib/libonnxruntime.so.1.32.0
COPY --from=cuda-runtime /usr/local/lib/python3.11/dist-packages/onnxruntime/capi/libonnxruntime_providers*.so /opt/cuda/lib/
RUN ln -sf libonnxruntime.so.1.32.0 /opt/cuda/lib/libonnxruntime.so && \
    ln -sf /opt/cuda/lib/libonnxruntime.so.1.32.0 /usr/lib/libonnxruntime.so && \
    ln -sf /opt/cuda/lib/libonnxruntime_providers_cuda.so /usr/lib/libonnxruntime_providers_cuda.so && \
    ln -sf /opt/cuda/lib/libonnxruntime_providers_tensorrt.so /usr/lib/libonnxruntime_providers_tensorrt.so && \
    ln -sf /opt/cuda/lib/libonnxruntime_providers_shared.so /usr/lib/libonnxruntime_providers_shared.so
COPY birdnet-go-cuda /usr/bin/birdnet-go
ENV LD_LIBRARY_PATH=/opt/cuda/lib BIRDNET_CUDA=1 BIRDNET_TENSORRT=1

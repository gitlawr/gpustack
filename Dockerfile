ARG CUDA_VERSION=12.5.1

FROM nvidia/cuda:$CUDA_VERSION-runtime-ubuntu22.04 AS cuda-12-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY dist/*.whl /tmp/

RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
    pip3 install /tmp/*_x86_64.whl; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
    pip3 install /tmp/*_aarch64.whl; \
    fi \
    && rm /tmp/*.whl

ENTRYPOINT [ "gpustack", "start" ]

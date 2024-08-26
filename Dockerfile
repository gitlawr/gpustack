ARG CUDA_VERSION=12.5.1

FROM nvidia/cuda:$CUDA_VERSION-runtime-ubuntu22.04 AS cuda-12-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


FROM cuda-12-base AS cuda-amd64

COPY dist/*_x86_64.whl /tmp/

RUN pip3 install /tmp/*_x86_64.whl && rm /tmp/*_x86_64.whl

ENTRYPOINT [ "gpustack", "start" ]


FROM cuda-12-base as cuda-arm64

COPY dist/*_aarch64.whl /tmp/

RUN pip3 install /tmp/*_aarch64.whl && rm /tmp/*_aarch64.whl

ENTRYPOINT [ "gpustack", "start" ]

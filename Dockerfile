ARG CUDA_VERSION=12.5.1

FROM --platform=$TARGETPLATFORM nvidia/cuda:$CUDA_VERSION-runtime-ubuntu22.04 as cuda-12-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


FROM --platform=linux/amd64 cuda-12-base

COPY dist/*_x86_64.whl /tmp/

RUN pip3 install /tmp/*_x86_64.whl && rm /tmp/*_x86_64.whl

ENTRYPOINT [ "gpustack", "start" ]


FROM --platform=linux/arm64 cuda-12-base

COPY dist/*_aarch64.whl /tmp/

RUN pip3 install /tmp/*_aarch64.whl && rm /tmp/*_aarch64.whl

ENTRYPOINT [ "gpustack", "start" ]

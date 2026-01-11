FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y build-essential cmake git && \
    rm -rf /var/lib/apt/lists/*
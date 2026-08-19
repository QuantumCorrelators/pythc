FROM ubuntu:24.04

# Install python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/* \

WORKDIR pythc
RUN python3 -m venv .venv
RUN .venv/bin/python -m pip install "git+https://github.com/QuantumCorrelators/pythc.git"

CMD [".venv/bin/python"]

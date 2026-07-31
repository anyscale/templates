FROM anyscale/ray:2.56.0-py312

RUN mkdir -p /home/ray/default

RUN sudo apt-get update && \
    sudo apt-get install -y podman && \
    sudo rm -rf /var/lib/apt/lists/*

COPY python_depset.lock /tmp/python_depset.lock
RUN uv pip install --system --no-deps --no-cache-dir --index-strategy unsafe-best-match \
    -r /tmp/python_depset.lock

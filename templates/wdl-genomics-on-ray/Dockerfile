# Cluster image. Under --container-runtime none every tool must be on every node at boot, and
# a job's requirements: resolves before working_dir is staged, so the toolchain is baked in.
#   anyscale image build -n my-wdl-tools --containerfile Dockerfile
FROM anyscale/ray:2.58.0-py312

# zlib1g-dev: without it Flye's make fails behind an unrelated setup.py TypeError. bzip2, lzma
# and libcurl are htslib's (CRAM, https); jq is for tools/stage-demo-data.sh.
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
        build-essential \
        zlib1g-dev \
        libbz2-dev \
        liblzma-dev \
        libcurl4-openssl-dev \
        jq \
    && sudo rm -rf /var/lib/apt/lists/*

# Copied before the package so an engine change keeps this, the most expensive layer, cached.
# /opt is root-owned and the image runs as ray, hence the sudo mkdir.
COPY --chown=ray:users tools/ /tmp/wdl-tools-build/
RUN sudo mkdir -p /opt/wdl-tools && sudo chown ray:users /opt/wdl-tools
RUN bash /tmp/wdl-tools-build/build_tools.sh /opt/wdl-tools \
    && rm -rf /tmp/wdl-tools-build

ENV PATH="/opt/wdl-tools/bin:${PATH}"

# Installed, not just uploaded, so the container_backend entry point resolves for plain miniwdl too.
COPY --chown=ray:users pyproject.toml /tmp/wdl-on-ray/
COPY --chown=ray:users wdl_on_ray/ /tmp/wdl-on-ray/wdl_on_ray/
RUN pip install --no-cache-dir /tmp/wdl-on-ray && rm -rf /tmp/wdl-on-ray

# Fail the build, not the run: a tool that cannot execute, or a QUAST without its aligner.
RUN set -eux; \
    minimap2 --version; \
    paftools.js version; \
    samtools --version | head -1; \
    bcftools --version | head -1; \
    flye --version; \
    quast --version; \
    miniwdl --version; \
    wdl-on-ray --version; \
    python -c "import quast_libs.qutils as q, shutil, sys; \
p = shutil.which('minimap2'); print('quast will use minimap2 at', p); sys.exit(0 if p else 1)"

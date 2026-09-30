# Cluster image. Ray workers have no container runtime, so the bioconda toolchain and a JRE are
# baked in; the build asserts Python 3.12 and Ray 2.58.0, which driver and workers must share.

# The nf-ray plugin. JDK 21 for the Gradle plugin's toolchain; the bytecode targets Java 17.
FROM --platform=$BUILDPLATFORM gradle:8.14-jdk21 AS plugin-build

WORKDIR /src
COPY nf-ray-plugin/ /src/

ENV NXF_PLUGINS_DIR=/out/plugins
RUN gradle --no-daemon installPlugin \
    && ls -la /out/plugins/*/ \
    && test -f /out/plugins/nf-ray-0.1.0/classes/ai/anyscale/nfray/RayExecutor.class \
         || test -n "$(find /out/plugins/nf-ray-0.1.0 -name '*.jar' -print -quit)" \
         || (echo 'plugin built but produced no classes or jar' >&2; exit 1)

FROM anyscale/ray:2.58.0-py312-cu128

USER root
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        bzip2 xz-utils curl ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*
USER ray

RUN sudo mkdir -p /opt/nf-tools && sudo chown -R ray:users /opt/nf-tools

RUN curl -sSL https://micro.mamba.pm/api/micromamba/linux-64/2.0.5 \
      | tar -xvj -C /opt/nf-tools bin/micromamba \
    && /opt/nf-tools/bin/micromamba --version

ENV MAMBA_ROOT_PREFIX=/opt/nf-tools/mamba
ENV PATH=/opt/nf-tools/bin:${PATH}

# Named files rather than tools/, so editing the other scripts there doesn't rebuild the envs.
COPY --chown=ray:users tools/build_tools.sh tools/manifest.toml tools/env.* \
     /opt/nf-tools/src/tools/
RUN bash /opt/nf-tools/src/tools/build_tools.sh /opt/nf-tools

# Appended, not prepended: the env carries its own python, which must not shadow the image's.
ENV PATH=${PATH}:/opt/nf-tools/envs/main/bin

ENV NXF_PLUGINS_DIR=/opt/nf-tools/nextflow-plugins
COPY --from=plugin-build --chown=ray:users /out/plugins/ ${NXF_PLUGINS_DIR}/

ENV NXF_OPTS="-Xms512m -Xmx4g"
ENV NXF_HOME=/opt/nf-tools/nextflow
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Two installs: the torch file's --index-url must not apply to anything else.
COPY --chown=ray:users tools/requirements.gpu.txt /tmp/requirements.gpu.txt
COPY --chown=ray:users requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.gpu.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# GATK ships `scorevariants` only in its source tree. Its setup.py declares no dependencies.
ARG GATK_VERSION=4.6.2.0
ARG GATK_SRC_SHA256=ad5df0e5a390aea483336a4895ce67d687142ee448ae59b1c41ed9c7f523acee
RUN curl -fsSL -o /tmp/gatk-src.tar.gz \
        "https://github.com/broadinstitute/gatk/archive/refs/tags/${GATK_VERSION}.tar.gz" \
    && echo "${GATK_SRC_SHA256}  /tmp/gatk-src.tar.gz" | sha256sum -c - \
    && mkdir /tmp/gatk-python \
    && tar -xzf /tmp/gatk-src.tar.gz -C /tmp/gatk-python --strip-components=7 \
        "gatk-${GATK_VERSION}/src/main/python/org/broadinstitute/hellbender" \
    && pip install --no-cache-dir --no-deps --no-build-isolation /tmp/gatk-python \
    && rm -rf /tmp/gatk-src.tar.gz /tmp/gatk-python

ENV NF_GATK_PYTHON=/home/ray/anaconda3/bin/python

COPY --chown=ray:users pyproject.toml /tmp/nf-ray/pyproject.toml
COPY --chown=ray:users nf_ray/ /tmp/nf-ray/nf_ray/
RUN pip install --no-cache-dir /tmp/nf-ray && rm -rf /tmp/nf-ray

# /bin/sh is dash, where `tool --version | head -1` passes whatever the tool does, hence pipefail;
# sed -n rather than head, so a chatty tool can't die of SIGPIPE.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN set -eux; \
    test "$(command -v python)" = /home/ray/anaconda3/bin/python; \
    test "$(command -v samtools)" = /opt/nf-tools/envs/main/bin/samtools; \
    python -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version"; \
    python -c "import ray; assert ray.__version__ == '2.58.0', ray.__version__"; \
    python -c "import numpy; assert numpy.__version__.startswith('2.2.'), numpy.__version__"; \
    python -c "import torch; print('torch', torch.__version__); assert torch.version.cuda.startswith('12.8'), torch.version.cuda"; \
    python -c "import pytorch_lightning, pysam, Bio, scorevariants.models.gatk_1d; print('lightning', pytorch_lightning.__version__, 'pysam', pysam.__version__, 'biopython', Bio.__version__)"; \
    java -version; \
    nextflow -version; \
    nf-ray --version; \
    bwa-mem2 version; \
    samtools --version | sed -n 1p; \
    bcftools --version | sed -n 1p; \
    tabix --version | sed -n 1p; \
    bgzip --version | sed -n 1p; \
    fastp --version 2>&1 | sed -n 1p; \
    gatk --version; \
    rtg version | sed -n 1,2p; \
    multiqc --version; \
    jq --version; \
    test "$(command -v aws)" = /usr/local/bin/aws; \
    aws --version

RUN set -eux; \
    ls "${NXF_PLUGINS_DIR}"; \
    grep -rq "ray" "${NXF_PLUGINS_DIR}"/nf-ray-*/ || (echo 'nf-ray plugin looks empty' >&2; exit 1)

# NVScoreVariants on two records. Kept in the image: run with `gpu` on an L4 node to check CUDA.
COPY --chown=ray:users tools/smoke-nvscorevariants.sh /opt/nf-tools/src/tools/
RUN bash /opt/nf-tools/src/tools/smoke-nvscorevariants.sh cpu

WORKDIR /home/ray

# Cluster image for the nextflow-genomics-on-ray template.
#
#   ../../.claude/skills/template/scripts/push-custom-image-to-gcp.sh \
#       . nextflow-genomics-on-ray 2.58.0
#
# Or, to build one you own:  anyscale image build -n my-nf-tools --containerfile Dockerfile
#
# --- why a custom image rather than a stock one plus `requirements:` ---------
#
# This template runs Nextflow processes with no container runtime. A Ray worker
# is already inside a container and cannot start another, so a process that
# declares `container 'quay.io/biocontainers/...'` has nothing to run it, and the
# toolchain has to be on every node before the first task dispatches.
#
# An Anyscale job's `requirements:` installs pip packages, and the toolchain is
# bioconda packages plus a JRE, so it is baked in here. Baking also keeps every
# node on the same tool versions.
#
# --- the version contract ---------------------------------------------------
#
# Python 3.12 and Ray 2.58.0, everywhere, asserted at build time below. A Ray
# driver and its workers must agree on both to the patch, and `ext.image` fails
# outright on a mismatch.

# =============================================================================
# Stage 1: build the Nextflow executor plugin.
#
# Gradle, the JDK compiler and ~500 MB of dependency cache stay here. The shipped
# image gets one extracted directory.
#
# JDK 21, although the image runs Java 17. The io.nextflow.nextflow-plugin Gradle
# plugin asks for a Java 21 toolchain, and in gradle:8.14-jdk17 compileGroovy
# fails with "Cannot find a Java installation ... matching: {languageVersion=21}".
# It still emits Java 17 bytecode (class-file major version 61, checked on the
# built RayExecutor.class), which is what the image's openjdk-17 loads.
#
# $BUILDPLATFORM, because the output is JVM bytecode and runs anywhere. On an
# amd64 builder this changes nothing; on an arm64 one it keeps Gradle out of
# emulation.
# =============================================================================
FROM --platform=$BUILDPLATFORM gradle:8.14-jdk21 AS plugin-build

WORKDIR /src
COPY nf-ray-plugin/ /src/

# installPlugin extracts to $NXF_PLUGINS_DIR/<name>-<version>, which is exactly
# the layout Nextflow expects to find offline. Setting the variable here means the
# next stage can copy a single predictable path.
ENV NXF_PLUGINS_DIR=/out/plugins
RUN gradle --no-daemon installPlugin \
    && ls -la /out/plugins/*/ \
    && test -f /out/plugins/nf-ray-0.1.0/classes/ai/anyscale/nfray/RayExecutor.class \
         || test -n "$(find /out/plugins/nf-ray-0.1.0 -name '*.jar' -print -quit)" \
         || (echo 'plugin built but produced no classes or jar' >&2; exit 1)

# =============================================================================
# Stage 2: the cluster image.
# =============================================================================
FROM anyscale/ray:2.58.0-py312-cu128

USER root
# openjdk: Nextflow's own runtime, and GATK's and rtg's. One JRE serves all three.
# bzip2/xz/curl: htslib reads CRAM and remote references.
# procps: `nproc`, which the smoke pipeline reports and several tools call.
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        bzip2 xz-utils curl ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*
USER ray

# /opt is root-owned in the base image, so the directory has to be made and
# handed over before an unprivileged build step can write into it.
RUN sudo mkdir -p /opt/nf-tools && sudo chown -R ray:users /opt/nf-tools

# micromamba, for the conda environments. A single static binary; no base
# environment, no shell hook, nothing that could shadow the image's python.
RUN curl -sSL https://micro.mamba.pm/api/micromamba/linux-64/2.0.5 \
      | tar -xvj -C /opt/nf-tools bin/micromamba \
    && /opt/nf-tools/bin/micromamba --version

ENV MAMBA_ROOT_PREFIX=/opt/nf-tools/mamba
ENV PATH=/opt/nf-tools/bin:${PATH}

# The toolchain. Copied before the rest of the template so that editing the
# executor or the pipeline does not invalidate this layer -- it is by far the
# most expensive one.
#
# Named files, not `tools/`. The directory also holds scripts that are not build
# inputs (stage-demo-data.sh, lock-envs.sh), and copying all of it meant editing
# one of those rebuilt the conda envs -- and baked whatever else happened to be
# sitting in a working tree into the image. `env.*` picks up a lock file too.
COPY --chown=ray:users tools/build_tools.sh tools/manifest.toml tools/env.* \
     /opt/nf-tools/src/tools/
RUN bash /opt/nf-tools/src/tools/build_tools.sh /opt/nf-tools

# The main environment goes on PATH, *last*; any other environment stays off it
# (see build_tools.sh). This must be ENV rather than a shell profile, because
# process scripts run from Ray worker processes that never source one.
#
# Last, not first, because the env carries its own python (conda-forge 3.12.x,
# which multiqc needs). Put first, it shadowed the image's: `python` became the
# env's interpreter, `import ray` failed in the verification step below, and
# `nf-ray` and GATK's NVScoreVariants would have picked it up at run time too.
# Appended, `python` stays /home/ray/anaconda3/bin/python, and the tools still
# resolve because nothing earlier on PATH carries their names -- the
# verification step checks both.
ENV PATH=${PATH}:/opt/nf-tools/envs/main/bin

# The executor plugin, from stage 1. NXF_PLUGINS_DIR points Nextflow here, so it
# resolves nf-ray offline -- a cluster with no route to the plugin registry still
# works, and no pipeline run pays a download.
ENV NXF_PLUGINS_DIR=/opt/nf-tools/nextflow-plugins
COPY --from=plugin-build --chown=ray:users /out/plugins/ ${NXF_PLUGINS_DIR}/

# Nextflow's own JVM heap. It runs on the head node, which this template's compute
# configs give `CPU: 0` so that no task competes with it -- but the default heap
# is a fraction of host RAM and a 72-task DAG with full trace reporting is not
# small.
ENV NXF_OPTS="-Xms512m -Xmx4g"
ENV NXF_HOME=/opt/nf-tools/nextflow
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# The GPU arm, GATK's NVScoreVariants: a Java front end that runs GATK's Python
# package, `scorevariants`, on PyTorch, in the image's Python rather than a second
# environment. torch first, from the CUDA-12.8 index, then Lightning, pysam and
# Biopython from PyPI. Two commands rather than one because the torch requirements
# file carries an --index-url that must not apply to anything else.
COPY --chown=ray:users tools/requirements.gpu.txt /tmp/requirements.gpu.txt
COPY --chown=ray:users requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.gpu.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# `scorevariants` itself, from the GATK source at the version of the jar in the main
# env. GATK publishes it only in its source tree and inside the 704 MB release zip;
# the source tarball is the smaller download, and it is pinned by checksum. Its
# setup.py declares no dependencies, so --no-deps; --no-build-isolation so pip builds
# it with the image's setuptools instead of fetching an unpinned one.
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

# The interpreter NVSCOREVARIANTS points GATK's `python` and `python3` lookups at.
ENV NF_GATK_PYTHON=/home/ray/anaconda3/bin/python

# The executor's Ray side. Installed (not merely staged) so the `nf-ray` console
# script is on PATH for the plugin to call.
COPY --chown=ray:users pyproject.toml /tmp/nf-ray/pyproject.toml
COPY --chown=ray:users nf_ray/ /tmp/nf-ray/nf_ray/
RUN pip install --no-cache-dir /tmp/nf-ray && rm -rf /tmp/nf-ray

# --- verification ------------------------------------------------------------
#
# Every declared tool actually runs, and the version contract actually holds. A
# build that gets this far cannot be the one where a stray dependency upgraded
# Ray or numpy -- which is a failure that otherwise appears as a worker that will
# not start, on a cluster, twenty minutes later.
#
# bash with pipefail, because the default /bin/sh here is dash, where
# `tool --version | head -1` succeeds whatever the tool does. And `sed -n`, not
# `head`, to take the first lines: sed reads to the end, so a tool that writes a
# lot cannot die of SIGPIPE and fail the build for the wrong reason.
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

# The plugin is resolvable offline and names itself `ray`. Checked here because
# the alternative place to discover it is a pipeline that has already staged its
# inputs.
RUN set -eux; \
    ls "${NXF_PLUGINS_DIR}"; \
    grep -rq "ray" "${NXF_PLUGINS_DIR}"/nf-ray-*/ || (echo 'nf-ray plugin looks empty' >&2; exit 1)

# NVScoreVariants end to end, on the CPU, on two records: the Java front end, the
# Python lookup, scorevariants, Lightning, and the model inside the jar, run the way
# NVSCOREVARIANTS runs them. GATK exits 0 when its Python fails, so the script checks
# that both records come back scored; a broken environment fails the build here
# rather than on an L4 node.
# Kept in the image, so `bash /opt/nf-tools/src/tools/smoke-nvscorevariants.sh gpu`
# checks the CUDA path on a GPU node.
COPY --chown=ray:users tools/smoke-nvscorevariants.sh /opt/nf-tools/src/tools/
RUN bash /opt/nf-tools/src/tools/smoke-nvscorevariants.sh cpu

WORKDIR /home/ray

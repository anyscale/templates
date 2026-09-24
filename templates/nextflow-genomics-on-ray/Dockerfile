# Cluster image for the nextflow-genomics-on-ray template.
#
#   ../../.claude/skills/template/scripts/push-custom-image-to-gcp.sh \
#       . nextflow-genomics-on-ray 2.58.0
#
# Or, on a machine with podman and no docker daemon:  bash tools/push-image-podman.sh
# Or, to build one you own:  anyscale image build -n my-nf-tools --containerfile Dockerfile
#
# --- why a custom image rather than a stock one plus `requirements:` ---------
#
# This template runs Nextflow processes with no container runtime. A Ray worker
# is already inside a container and cannot nest another, so a process that
# declares `container 'quay.io/biocontainers/...'` has nothing to run it -- the
# toolchain has to be present on every node before the first task dispatches.
#
# An Anyscale job's `requirements:` cannot do that job here: it resolves *before*
# working_dir is staged, so it cannot see a wheelhouse shipped with the code, and
# a published index is fetched anonymously. Pre-staging wheels on
# /mnt/user_storage works but is a separate job to keep in sync. Baking is
# simpler and is what the WDL sibling template settled on for the same reason.
#
# --- the version contract ---------------------------------------------------
#
# Python 3.12 and Ray 2.58.0, everywhere, asserted at build time below. A Ray
# driver and its workers must agree on both to the patch, and `ext.image` fails
# outright on a mismatch. The one tool that cannot meet that (DeepVariant, which
# pins python <3.11) gets a quarantined environment and is never on PATH.

# =============================================================================
# Stage 1: build the Nextflow executor plugin.
#
# Gradle, the JDK compiler and ~500 MB of dependency cache stay here. The shipped
# image gets one extracted directory.
# =============================================================================
FROM gradle:8.14-jdk17 AS plugin-build

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
COPY --chown=ray:users tools/ /opt/nf-tools/src/tools/
RUN bash /opt/nf-tools/src/tools/build_tools.sh /opt/nf-tools

# The main environment goes on PATH, *last*. The others deliberately do not; see
# build_tools.sh. This must be ENV rather than a shell profile, because process
# scripts run from Ray worker processes that never source one.
#
# Last, not first, because the env carries its own python (conda-forge 3.12.x,
# which multiqc needs). Put first, it shadowed the image's: `python` became the
# env's interpreter, `import ray` failed in the verification step below, and
# `nf-ray` and score_variants.py would have picked it up at run time too.
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

# GPU leg. torch first, from the CUDA-12.8 index, then everything else from PyPI.
# Two commands rather than one because the torch requirements file carries an
# --index-url that must not apply to anything else.
COPY --chown=ray:users tools/requirements.gpu.txt /tmp/requirements.gpu.txt
COPY --chown=ray:users requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.gpu.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# The executor's Ray side. Installed (not merely staged) so the `nf-ray` console
# script is on PATH for the plugin to call.
COPY --chown=ray:users pyproject.toml /tmp/nf-ray/pyproject.toml
COPY --chown=ray:users nf_ray/ /tmp/nf-ray/nf_ray/
RUN pip install --no-cache-dir /tmp/nf-ray && rm -rf /tmp/nf-ray

# The model the GPU process scores with, baked in. Downloading it per task would
# have every node in a fan-out pull the same 200 MB from Hugging Face at once, and
# would make the pipeline need egress it otherwise does not.
#
# `*.py` is not optional: the model runs its own modelling code (esm_config.py,
# modeling_esm.py) via trust_remote_code, and ANNOTATE_VARIANTS sets
# HF_HUB_OFFLINE=1, so code missing here cannot be fetched later. The revision is
# pinned for the same reason -- remote code should be code someone read. The
# second command loads it offline and runs one forward pass, so a transformers
# upgrade that breaks the model's code fails here rather than on a GPU node.
ENV HF_HOME=/opt/nf-tools/hf
ENV NF_RAY_SCORER_MODEL=InstaDeepAI/nucleotide-transformer-v2-50m-multi-species
ENV NF_RAY_SCORER_REVISION=81b29e5786726d891dbf929404ef20adca5b36f1
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${NF_RAY_SCORER_MODEL}', revision='${NF_RAY_SCORER_REVISION}', \
                  allow_patterns=['*.json', '*.safetensors', '*.txt', '*.py'], \
                  ignore_patterns=['jax_model/*'])" \
    && HF_HUB_OFFLINE=1 python -c "\
import os; \
from transformers import AutoModelForMaskedLM, AutoTokenizer; \
name, rev = os.environ['NF_RAY_SCORER_MODEL'], os.environ['NF_RAY_SCORER_REVISION']; \
tok = AutoTokenizer.from_pretrained(name, revision=rev, trust_remote_code=True); \
model = AutoModelForMaskedLM.from_pretrained(name, revision=rev, trust_remote_code=True).eval(); \
out = model(**tok(['ACGTACGTACGTAAAAAA'], return_tensors='pt'), output_hidden_states=True); \
print('scorer model loads offline; last hidden state', tuple(out.hidden_states[-1].shape))" \
    && du -sh ${HF_HOME}

# --- verification ------------------------------------------------------------
#
# Every declared tool actually runs, and the version contract actually holds. A
# build that gets this far cannot be the one where a stray dependency upgraded
# Ray or numpy -- which is a failure that otherwise appears as a worker that will
# not start, on a cluster, twenty minutes later.
RUN set -eux; \
    test "$(command -v python)" = /home/ray/anaconda3/bin/python; \
    test "$(command -v samtools)" = /opt/nf-tools/envs/main/bin/samtools; \
    python -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version"; \
    python -c "import ray; assert ray.__version__ == '2.58.0', ray.__version__"; \
    python -c "import numpy; assert numpy.__version__.startswith('2.2.'), numpy.__version__"; \
    python -c "import torch; print('torch', torch.__version__); assert torch.version.cuda.startswith('12.8'), torch.version.cuda"; \
    python -c "import transformers; print('transformers', transformers.__version__)"; \
    java -version; \
    nextflow -version; \
    nf-ray --version; \
    bwa-mem2 version || true; \
    samtools --version | head -1; \
    bcftools --version | head -1; \
    fastp --version 2>&1 | head -1; \
    gatk --list 2>&1 | head -3; \
    rtg version | head -2; \
    multiqc --version; \
    test "$(command -v aws)" = /usr/local/bin/aws; \
    aws --version; \
    /opt/nf-tools/envs/deepvariant/bin/python --version; \
    test ! -e /opt/nf-tools/envs/deepvariant/bin/run_deepvariant || echo "deepvariant present"; \
    python -c "import sys; assert '3.10' not in sys.version, 'deepvariant env leaked onto PATH'"

# The plugin is resolvable offline and names itself `ray`. Checked here because
# the alternative place to discover it is a pipeline that has already staged its
# inputs.
RUN set -eux; \
    ls "${NXF_PLUGINS_DIR}"; \
    grep -rq "ray" "${NXF_PLUGINS_DIR}"/nf-ray-*/ || (echo 'nf-ray plugin looks empty' >&2; exit 1)

WORKDIR /home/ray

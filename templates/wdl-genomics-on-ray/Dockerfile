# Cluster image for the wdl-genomics-on-ray template.
#
# Build and push:
#   ../../.claude/skills/template/scripts/push-custom-image-to-gcp.sh . wdl-genomics-on-ray 2.56.0
#
# Or build your own copy under any name (see the README section "Build your own image"):
#   anyscale image build -n my-wdl-tools --containerfile Dockerfile
#
# Why a custom image rather than the stock anyscale/ray one plus `requirements:`.
#
# The workflow runs with `--container-runtime none`, which executes each WDL task's command
# directly in the Ray worker's environment. There is no per-task container, so *every tool has
# to be on every node before the first task dispatches*. Three routes were tried; only this
# one holds:
#
#   requirements: + a published index    a job's requirements: is installed at cluster startup,
#                                        and pip fetches an https:// index anonymously, so a
#                                        private bucket cannot be reached by the node's IAM role
#   requirements: + ./local/wheelhouse   requirements: resolves *before* working_dir is staged,
#                                        so wheels shipped with the submission are not there yet
#   pre-staged wheels on /mnt/user_storage   works, but it is a separate job the reader has to
#                                        run first, and it leaves state behind between runs
#
# Baking the toolchain in has none of those failure modes: the tools are present at node boot,
# nothing resolves at run time, and the image is the same on every node by construction.
# tools/BUILDING.md documents the wheel routes for anyone who wants them anyway.
FROM anyscale/ray:2.56.0-py312

# Build dependencies for the two `kind = "source"` tools in the manifest. zlib1g-dev is the
# one that actually bites: without it Flye's make fails, and its setup.py error handler raises
# `TypeError: exit expected at most 1 argument, got 2` on the way out, hiding the real cause.
# bzip2/liblzma are htslib's, for samtools' CRAM support.
# libcurl-dev is for htslib's https/S3 file handlers: tools/stage-demo-data.sh reads the
# public GIAB CRAMs by region over https, and neither this image's samtools (before this
# line) nor Ubuntu's apt samtools is built with them ("Protocol not supported"). jq is the
# same script's manifest writer. The pipeline itself needs neither.
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
        build-essential \
        zlib1g-dev \
        libbz2-dev \
        liblzma-dev \
        libcurl4-openssl-dev \
        jq \
    && sudo rm -rf /var/lib/apt/lists/*

# The toolchain, from tools/manifest.toml. Copied before the package so a change to the
# workflow engine doesn't invalidate this layer -- it is by far the most expensive one.
#
# The prefix is created with sudo first: the base image runs as `ray`, /opt is root-owned,
# and without this build_tools.sh dies at its first mkdir. (Found the first time this file
# was actually built -- see the git history for why that took a while.)
COPY --chown=ray:users tools/ /tmp/wdl-tools-build/
RUN sudo mkdir -p /opt/wdl-tools && sudo chown ray:users /opt/wdl-tools
RUN bash /tmp/wdl-tools-build/build_tools.sh /opt/wdl-tools \
    && rm -rf /tmp/wdl-tools-build

# /opt/wdl-tools/bin has to be on PATH for *Ray workers*, not just for an interactive shell,
# because that is where the WDL task commands run. ENV covers both.
ENV PATH="/opt/wdl-tools/bin:${PATH}"

# The engine. Installing it (rather than relying on the uploaded working_dir) is what
# registers the `miniwdl.plugin.container_backend` entry point, so `container_backend = ray`
# resolves for anything that reads miniwdl's configuration -- including miniwdl invoked
# directly, which is the documented escape hatch.
COPY --chown=ray:users pyproject.toml /tmp/wdl-on-ray/
COPY --chown=ray:users wdl_on_ray/ /tmp/wdl-on-ray/wdl_on_ray/
RUN pip install --no-cache-dir /tmp/wdl-on-ray && rm -rf /tmp/wdl-on-ray

# Fail the build rather than the run. Each of these has failed silently at least once in this
# project's history: a tool that installs but cannot execute, and a QUAST whose aligner is
# missing so it exits 0 with a contiguity-only report.
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

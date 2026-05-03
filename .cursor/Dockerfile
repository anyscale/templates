# Cursor Cloud Agent base image for anyscale/templates.
# Provides the minimum prerequisites .cursor/install.sh needs to bootstrap
# at every VM startup (curl, sudo, apt, python/pip, git). Heavier deps
# (gh, anyscale, gcloud, docker) are installed by install.sh idempotently.
#
# This Dockerfile's primary job is to put environment.json into "Manual
# Dockerfile" mode so Cursor honors the install command instead of falling
# back to its agent-driven setup which ignores it.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl wget ca-certificates sudo gnupg lsb-release \
      git python3 python3-pip python-is-python3 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

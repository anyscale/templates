#!/usr/bin/env bash
# Start the Docker daemon for the byod custom-image bump flow — the
# `push-custom-image-to-gcp.sh` step needs `docker build` + `docker push`.
#
# The previous env start command, `sudo service docker start`, fails on this base
# image with "docker: unrecognized service": docker.io ships no init-service entry
# and the container runs no systemd/sysvinit, so the daemon never came up and every
# custom-image bump then failed its `docker info` check. Launch dockerd directly
# and block until the socket is ready instead.
#
# Base-image templates (image_uri only) don't need docker; for them this is just a
# short readiness wait, and a failed start is deliberately non-fatal (exit 0) so
# those bumps still proceed. Custom-image bumps that truly need docker will
# stop-and-report cleanly at their own `docker info` check if it isn't up.
set -uo pipefail

docker info >/dev/null 2>&1 && { echo "docker: already running"; exit 0; }

# fuse-overlayfs: the storage driver that works in an unprivileged container
# (the Dockerfile installs the fuse-overlayfs + iptables packages for exactly this).
sudo sh -c 'dockerd --storage-driver=fuse-overlayfs >/var/log/dockerd.log 2>&1 &'

for _ in $(seq 1 30); do
  docker info >/dev/null 2>&1 && { echo "docker: ready"; exit 0; }
  sleep 1
done

echo "docker: dockerd failed to start after 30s — last log lines:" >&2
sudo tail -n 40 /var/log/dockerd.log 1>&2 || true
# If dockerd genuinely can't run here (e.g. an unprivileged container needing
# --iptables=false or more capabilities), the log above says which. Non-fatal.
exit 0

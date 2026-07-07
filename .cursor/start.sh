#!/usr/bin/env bash
# Start dockerd for the custom-image bump flow (push-custom-image-to-gcp.sh needs
# `docker build`/`push`). Cursor runs this as environment.json's `start`, every boot.
# `service docker start` fails on this base ("docker: unrecognized service"), so run
# dockerd directly. Non-fatal (exit 0): base-image bumps don't need docker.
set -uo pipefail

# dockerd runs as root; agents run as non-root `ubuntu` and call plain `docker`.
open_socket() { sudo chmod 666 /var/run/docker.sock 2>/dev/null || true; }

# sudo, so readiness can't false-negative before the socket is opened.
sudo docker info >/dev/null 2>&1 && { open_socket; echo "docker: already running"; exit 0; }

# fuse-overlayfs: the storage driver for this unprivileged container.
sudo sh -c 'dockerd --storage-driver=fuse-overlayfs >/var/log/dockerd.log 2>&1 &'
for _ in $(seq 1 30); do
  sudo docker info >/dev/null 2>&1 && { open_socket; echo "docker: ready"; exit 0; }
  sleep 1
done

echo "docker: dockerd failed to start after 30s — last log lines:" >&2
sudo tail -n 40 /var/log/dockerd.log 1>&2 || true
exit 0

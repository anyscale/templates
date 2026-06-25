#!/usr/bin/env bash
# Persist Claude Code across ephemeral Anyscale clusters.
#
# Why this exists: /home (and so ~/.claude — settings, memory, auth) is on an
# overlay filesystem that is DESTROYED when the cluster is torn down.
# /mnt/user_storage is persistent NFS (your per-user share) and survives.
# This script syncs the whole ~/.claude to/from there, and reinstalls the CLI.
#
# Usage:
#   ./setup_claude.sh backup     # run BEFORE shutting down: ~/.claude -> persistent NFS
#   ./setup_claude.sh restore    # run on a FRESH cluster: persistent NFS -> ~/.claude (+ install CLI)
#   ./setup_claude.sh            # same as restore
#
# Point it somewhere else (e.g. shared storage) by overriding the location:
#   CLAUDE_BACKUP_DIR=/mnt/shared_storage/zach/.claude ./setup_claude.sh backup
#
# Tip: add `bash setup_claude.sh restore` to your Anyscale workspace startup
# command so a new cluster comes up with your settings + memory already in place.

set -euo pipefail

BACKUP_DIR="${CLAUDE_BACKUP_DIR:-/mnt/user_storage/claude-backup/.claude}"
CLAUDE_DIR="$HOME/.claude"

backup() {
    if [ ! -d "$CLAUDE_DIR" ]; then
        echo "[setup_claude] nothing to back up — $CLAUDE_DIR does not exist"
        return 0
    fi
    mkdir -p "$BACKUP_DIR"
    # --delete keeps the backup an exact mirror (drops files removed locally).
    rsync -a --delete "$CLAUDE_DIR"/ "$BACKUP_DIR"/
    echo "[setup_claude] backed up $CLAUDE_DIR -> $BACKUP_DIR ($(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1))"
}

restore() {
    if [ ! -d "$BACKUP_DIR" ]; then
        echo "[setup_claude] no backup at $BACKUP_DIR yet — run './setup_claude.sh backup' first."
        return 0
    fi
    mkdir -p "$CLAUDE_DIR"
    # No --delete: overlay the backup onto whatever a fresh ~/.claude already has.
    rsync -a "$BACKUP_DIR"/ "$CLAUDE_DIR"/
    echo "[setup_claude] restored $BACKUP_DIR -> $CLAUDE_DIR"
}

install_cli() {
    if ! command -v claude >/dev/null 2>&1; then
        echo "[setup_claude] installing Claude CLI..."
        curl -fsSL https://claude.ai/install.sh | bash
    fi
}

case "${1:-restore}" in
    backup)  backup ;;
    restore) install_cli; restore ;;
    *)       echo "usage: $0 [backup|restore]"; exit 2 ;;
esac

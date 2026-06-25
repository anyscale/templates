#!/usr/bin/env bash
# Persist Claude Code config + memory across this EPHEMERAL DEMO cluster.
#
# Everything on the box is disposable here — the overlay home (~/.claude) AND
# the /mnt NFS mounts can be wiped. The ONLY durable store is the git remote
# (GitHub). So this stashes the small, NON-SECRET parts of ~/.claude into this
# repo (which gets pushed) and restores them on a fresh cluster:
#     settings.json            — your Claude Code settings
#     projects/*/memory/*.md    — cross-session memory notes
# Credentials, caches, sessions and history are deliberately EXCLUDED: re-auth
# on a new cluster, and the rest regenerates. Nothing secret goes into git.
#
#   ./setup_claude.sh backup    # ~/.claude essentials -> repo, then commit + push
#   ./setup_claude.sh restore   # repo -> ~/.claude  (run this on a fresh cluster)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="$HERE/.claude-state"
CLAUDE="$HOME/.claude"

backup() {
    rm -rf "$STATE"
    mkdir -p "$STATE/projects"
    [ -f "$CLAUDE/settings.json" ] && cp "$CLAUDE/settings.json" "$STATE/settings.json"
    # Copy only the memory dirs (not the whole projects tree), preserving the
    # project-key path so restore can slot them back.
    if [ -d "$CLAUDE/projects" ]; then
        ( cd "$CLAUDE/projects" && find . -type d -name memory -print0 2>/dev/null \
            | while IFS= read -r -d '' d; do
                mkdir -p "$STATE/projects/$d"
                cp "$d"/*.md "$STATE/projects/$d/" 2>/dev/null || true
              done )
    fi
    if ( cd "$HERE" && git add .claude-state \
         && git commit -q -m "claude-state: settings + memory snapshot" \
         && git push origin HEAD ); then
        echo "[setup_claude] settings + memory committed and pushed to git."
    else
        echo "[setup_claude] nothing new to push (or push failed — check git)."
    fi
}

restore() {
    [ -d "$STATE" ] || { echo "[setup_claude] no $STATE found — run backup first (or git pull)."; return 0; }
    mkdir -p "$CLAUDE"
    [ -f "$STATE/settings.json" ] && cp "$STATE/settings.json" "$CLAUDE/settings.json"
    [ -d "$STATE/projects" ] && { mkdir -p "$CLAUDE/projects"; cp -a "$STATE/projects/." "$CLAUDE/projects/"; }
    echo "[setup_claude] restored settings + memory into $CLAUDE"
}

case "${1:-restore}" in
    backup)  backup ;;
    restore) restore ;;
    *) echo "usage: $0 [backup|restore]"; exit 2 ;;
esac

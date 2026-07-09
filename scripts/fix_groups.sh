#!/bin/bash
# ── TRIZ: Resolve all runtime-injected supplementary GIDs ──────────────────
# Principle 10 (Preliminary Action): add group entries BEFORE they're needed.
# Principle 5 (Merging): handle ALL injected GIDs in one pass, not just 109.
# IFR: self-healing — no manual per-GID work, handles any future GID.
#
# HuggingFace containers inject supplementary GIDs at runtime that the
# base image's /etc/group doesn't know. This causes:
#   groups: cannot find name for group ID 109
# Run this at container startup before any application code.
for gid in $(id -G 2>/dev/null); do
    if [ "$gid" != "0" ] && ! getent group "$gid" >/dev/null 2>&1; then
        echo "g${gid}:x:${gid}:" >> /etc/group
    fi
done

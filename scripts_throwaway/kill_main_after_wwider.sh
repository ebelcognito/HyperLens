#!/usr/bin/env bash
# Watcher: when main recovery (PID 1842362) finishes W_wider_bd refine,
# kill it so it doesn't run a redundant D_long_lbfgs (handled separately on alloc 49255630).
set -uo pipefail

MAIN_PID=1842362
MAIN_LOG=/tmp/octuplet_recovery_main.log

echo "[$(date +%H:%M:%S)] watcher: polling for W_wider_bd refine completion"
while true; do
    if grep -q "\[refine\] W_wider_bd done" "$MAIN_LOG" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] watcher: W_wider_bd refine done — killing main recovery PID=$MAIN_PID"
        kill -9 "$MAIN_PID" 2>/dev/null && echo "[$(date +%H:%M:%S)] watcher: killed" || echo "[$(date +%H:%M:%S)] watcher: kill failed (already dead?)"
        break
    fi
    if ! kill -0 "$MAIN_PID" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] watcher: main recovery already dead, exiting watcher"
        break
    fi
    sleep 5
done

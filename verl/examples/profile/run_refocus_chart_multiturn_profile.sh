#!/usr/bin/env bash
# Backward-compatible entrypoint -> workloads/chart/run_rollout_profile.sh
exec "$(dirname "$0")/workloads/chart/run_rollout_profile.sh" "$@"

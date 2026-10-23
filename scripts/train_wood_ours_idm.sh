#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
shift || true
"$(dirname "$0")/_method_benchmark_loop.sh" ours_idm "${GPU}" "$@"

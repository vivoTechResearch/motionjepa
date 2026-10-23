#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/_post_train_probe_loop.sh" lewm_flat "${1:-0}" "${@:2}"

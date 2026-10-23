#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/_post_train_probe_loop.sh" ours_idm "${1:-0}" "${@:2}"

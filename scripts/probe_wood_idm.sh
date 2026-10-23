#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/_post_train_probe_loop.sh" idm "${1:-0}" "${@:2}"

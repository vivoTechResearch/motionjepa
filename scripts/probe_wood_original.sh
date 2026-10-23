#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/_post_train_probe_loop.sh" original_lewm "${1:-0}" "${@:2}"

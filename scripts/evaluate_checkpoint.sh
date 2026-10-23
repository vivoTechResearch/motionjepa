#!/usr/bin/env bash
# Evaluate a released checkpoint with the paper's 25- or 50-step protocol.
set -euo pipefail
if [[ $# -lt 2 ]]; then echo "usage: $0 {cube|pusht|reacher|tworoom} POLICY [25|50] [num_eval]" >&2; exit 2; fi
task="$1"; policy="$2"; horizon="${3:-25}"; num_eval="${4:-5}"
case "${horizon}" in 25) offset=25; budget=50 ;; 50) offset=50; budget=100 ;; *) echo "horizon must be 25 or 50" >&2; exit 2 ;; esac
export STABLEWM_HOME="${STABLEWM_HOME:?Set STABLEWM_HOME to the directory containing datasets and checkpoints.}"
cd "$(dirname "$0")/.."
python eval.py --config-name="${task}" "policy=${policy}" "policy_cache_dir=${STABLEWM_HOME}" "eval.num_eval=${num_eval}" "eval.goal_offset_steps=${offset}" "eval.eval_budget=${budget}" wood.enabled=true wood.eval=true

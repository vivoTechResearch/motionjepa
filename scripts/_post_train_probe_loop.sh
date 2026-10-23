#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?missing probe method}"
GPU="${2:?missing GPU id}"
shift 2
LOG_ROOT="${MOTION_JEPA_LOG_ROOT:-outputs}"
OUTPUT_ROOT="${MOTION_JEPA_PROBE_ROOT:-${LOG_ROOT}/post_train_probes}"
export STABLEWM_HOME="${STABLEWM_HOME:?Set STABLEWM_HOME to the directory containing the LeWM HDF5 datasets.}"
cd "$(dirname "$0")/.."

for benchmark in pusht ogcube tworoom reacher; do
  # The train loop uses ogcube/reacher in names but ogb/dmc in data overrides.
  run_dir="$(find "${LOG_ROOT}" -maxdepth 1 -type d -name "*_${benchmark}_${METHOD}_wood*" 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -z "${run_dir}" ]]; then
    echo "No ${benchmark}/${METHOD} wood run under ${LOG_ROOT}; skipping." >&2
    continue
  fi
  export_dir="$(find "${run_dir}/checkpoints/checkpoints" -maxdepth 1 -type d -name 'pretrained_epoch_*' 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -z "${export_dir}" || ! -f "${export_dir}/weights.pt" ]]; then
    echo "No exported checkpoint in ${run_dir}; skipping." >&2
    continue
  fi
  name="${benchmark}_${METHOD}_$(basename "${export_dir}")"
  CUDA_VISIBLE_DEVICES="${GPU}" python post_train_probe.py \
    --benchmark "${benchmark}" --checkpoint "${export_dir}" \
    --output-root "${OUTPUT_ROOT}" --run-name "${name}" "$@"
done

#!/usr/bin/env bash
# Train one released Motion-JEPA variant on all four wood-texture benchmarks.
set -euo pipefail

METHOD="${1:?usage: $0 METHOD GPU [Hydra overrides...]}"
GPU="${2:?usage: $0 METHOD GPU [Hydra overrides...]}"
shift 2

export STABLEWM_HOME="${STABLEWM_HOME:?Set STABLEWM_HOME to the directory containing LeWM HDF5 datasets.}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$(dirname "$0")/.."

COMMON=(
  "hardware.gpus=[${GPU}]"
  "wood.enabled=true" "wood.train=true" "wood.val=true" "wood.eval=true"
  "wood.white_threshold=245" "wood.background_tolerance=8"
  "loss.weights.image_reconstruction=1.0" "loss.weights.pose_probe=1.0"
  "inverse_action_raw.enabled=false" "diff_image.enabled=false"
)

case "${METHOD}" in
  original_lewm) SETTINGS=("loss.weights.prediction=1.0" "loss.weights.sigreg=0.09") ;;
  lewm_flat) SETTINGS=("loss.weights.prediction=1.0" "loss.weights.sigreg=1.0" "sigreg_flat.enabled=true") ;;
  idm) SETTINGS=("loss.weights.prediction=1.0" "loss.weights.sigreg=0.0" "inverse_action_raw.enabled=true" "loss.weights.inverse_action_raw_prediction=1.0") ;;
  ours|ours_idm)
    SETTINGS=("loader.batch_size=64" "loss.weights.prediction=1.0" "diff_image.enabled=true" "diff_image.detach_z_for_prediction=false")
    SETTINGS+=("loss.weights.sigreg=0.25" "loss.weights.sigreg_diff_image=2.0" "loss.weights.diff_image_latent_prediction=0.5")
    if [[ "${METHOD}" == "ours_idm" ]]; then SETTINGS+=("inverse_action_raw.enabled=true" "loss.weights.inverse_action_raw_prediction=1.0"); fi
    ;;
  *) echo "Unknown released method: ${METHOD}" >&2; exit 2 ;;
esac

BENCHMARKS=(
  "pusht:pusht:white:"
  "tworoom:tworoom:white:"
  "ogb:ogcube:static:assets/cube_static_background.npy"
  "dmc:reacher:static:assets/reacher_static_background.npy"
)
for item in "${BENCHMARKS[@]}"; do
  IFS=: read -r data label mask static_background <<< "${item}"
  mask_overrides=("wood.mask_mode=${mask}")
  if [[ "${mask}" == static ]]; then mask_overrides+=("wood.static_background_path=${static_background}"); else mask_overrides+=("wood.static_background_path=null"); fi
  python train.py "data=${data}" "experiment.name=${label}_${METHOD}" "${COMMON[@]}" "${SETTINGS[@]}" "${mask_overrides[@]}" "$@"
done

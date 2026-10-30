#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python train.py \
  --logs-dir "${PTC_LOG_DIR:-runs/ptc}" \
  --lambda_topo 0.1 --topo_temperature 0.05 \
  --lambda_cm_ptc 0.05 --tau_cm 0.2 \
  "$@"

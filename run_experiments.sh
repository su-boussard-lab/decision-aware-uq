#!/usr/bin/env bash
#
# Reproduce the manuscript results.
#
#   ./run_experiments.sh                    # DPR: bootstrap ensemble and Bayesian LR
#   ./run_experiments.sh pou                # POU (requires the restricted data)
#   ./run_experiments.sh ensemble_dpr       # one or more named configs
#
# Outputs: results/<ensemble|blr>/<dpr|pou>/, logs in results/logs/.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
CONFIG_DIR="configs"
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

# Extract the preprocessed DPR splits on first use.
DPR_DIR="data/DPR_Data/preprocessed_data"
if [[ ! -f "$DPR_DIR/dpr_preprocessed/train.csv" ]]; then
  log "Extracting $DPR_DIR/dpr_preprocessed.zip"
  unzip -q -o "$DPR_DIR/dpr_preprocessed.zip" -d "$DPR_DIR/dpr_preprocessed"
fi

run_config() {
  local name="$1"
  local cfg="$CONFIG_DIR/$name.yaml"
  [[ -f "$cfg" ]] || { log "No config named '$name' in $CONFIG_DIR"; exit 1; }
  log "Running $name"
  "$PYTHON" run_experiments.py --config "$cfg" --log-file "$LOG_DIR/$name.log"
}

MODE="${1:-dpr}"
case "$MODE" in
  dpr) for name in ensemble_dpr blr_dpr; do run_config "$name"; done ;;
  pou) for name in ensemble_pou blr_pou; do run_config "$name"; done ;;
  *)   for name in "$@"; do run_config "$name"; done ;;
esac

log "Done. Outputs under results/"

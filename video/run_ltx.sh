#!/usr/bin/env bash
# Run run_v61.py (or any python script) in the separate LTX environment.
# Environment (see README.md / requirements_video.txt): torch 2.8.0+cu128, diffusers from source at commit e0abab83 (0.41.0.dev0),
# transformers 5.17.0; model Lightricks/LTX-2.5-Diffusers (distilled transformer).
#   LTX_PYTHON=/path/to/ltx-env/bin/python bash run_ltx.sh run_v61.py --stage root --scene park_typed --seed 2
# Optional env: CUDA_VISIBLE_DEVICES (GPU), HF_HOME (model cache), HF_TOKEN (if the model requires it),
#   LH_THREADS (CPU thread cap, default 24), LH_CPUS (CPU range 'a-b' to pin), LH_RSS_GB (RSS watchdog limit, default 55).
set -euo pipefail
cd "$(dirname "$0")"
PY=${LTX_PYTHON:-python}
mkdir -p logs
LOG="logs/$(basename "$1" .py)_g${CUDA_VISIBLE_DEVICES:-0}_$(date +%Y%m%d_%H%M%S).log"
echo "run(ltx): $* | gpu ${CUDA_VISIBLE_DEVICES:-0} | log $LOG"
"$PY" "$@" 2>&1 | tee "$LOG"

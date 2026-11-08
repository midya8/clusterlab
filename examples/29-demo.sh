#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m clusterlab demo
python -m clusterlab predict examples/predict.json >/tmp/clusterlab-predict.json
python examples/export_browser.py
python examples/render_audit.py

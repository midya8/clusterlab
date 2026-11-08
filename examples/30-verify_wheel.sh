#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH="$1"
source_root=$2
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
cd "$out"
python - <<'PY'
import clusterlab,clusterlab.__main__,clusterlab.io,clusterlab.kmeans,clusterlab.preprocess,clusterlab.metrics,clusterlab.data,clusterlab.experiment
print('eight installed modules imported')
PY
python -m clusterlab demo
python -m clusterlab predict "$source_root/examples/predict.json" >predict.json
python -m clusterlab train --out refit
python -m clusterlab evaluate --out refit
python - "$source_root" <<'PY'
import json,sys,numpy as np
from pathlib import Path
source=Path(sys.argv[1])/'clusterlab/artifacts'
a=json.load(open('refit/report.json'));b=json.load(open(source/'report.json'))
assert a['rows']==b['rows']==1797
for arm in a['arms']:
 x,y=a['arms'][arm],b['arms'][arm]
 assert x['labels']==y['labels']
 np.testing.assert_allclose(x['squared_distances'],y['squared_distances'],rtol=1e-8,atol=1e-10)
 np.testing.assert_allclose(x['silhouettes'],y['silhouettes'],rtol=1e-8,atol=1e-10)
 for k in ['ari','nmi','silhouette']:assert abs(x[k]-y[k])<1e-10
p=json.load(open('predict.json'));assert len(p)==4
print('all four installed CLIs, full refit and held-out evaluation pass')
PY

"""Reproduce label-free restart witnesses from frozen training artifacts only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from clusterlab.experiment import ROOT
from clusterlab.stability import restart_audit
from clusterlab.io import read,save,dumps
from clusterlab.preprocess import transform
import numpy as np
m=ROOT/'artifacts/models.json'
r=restart_audit(read(m))
save(ROOT/'artifacts/stability.json',r)
# No fitting assignment vectors or digit-label metadata are sent to this page.
bundle=dict(r)
bundle['examples']=[{'row':e['row'],'pixels':e['pixels']} for e in read(ROOT/'artifacts/report.json')['arms']['raw_plus']['examples']]
bundle['reference']={}
for name,a in r['arms'].items():
 z=transform([e['pixels'] for e in bundle['examples']],a['representation'])
 bundle['reference'][name]=[]
 for run in a['runs']:
  d=np.sum((z[:,None,:]-np.asarray(run['centers'])[None,:,:])**2,axis=2)
  bundle['reference'][name].append([{'cluster':int(row.argmin()),'squared_distances':row.tolist()} for row in d])
Path('docs/stability-data.js').write_text('globalThis.CLUSTER_STABILITY='+dumps(bundle)+';\n')

"""Export frozen inference models without restart label vectors."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from clusterlab.experiment import ROOT
from clusterlab.io import read,dumps
m=read(ROOT/'artifacts/models.json');r=read(ROOT/'artifacts/report.json');plan=read(ROOT/'artifacts/evaluation-plan.json')
models={k:{name:v[name] for name in ['centers','representation','pixel_centers','chosen']} for k,v in m.items()}
summary={k:{name:v[name] for name in ['ari','nmi','silhouette','mean_squared_distance','counts','composition_cluster_by_digit','fit_best_inertia','fit_worst_inertia','fit_selected_restart','fit_ari_to_selected_by_restart','examples']} for k,v in r['arms'].items()}
histories={k:[{'seed':run['seed'],'inertia':[h['inertia'] for h in run['history']],'reason':run['reason'],'repairs':run['repairs']} for run in v['runs']] for k,v in m.items()}
bundle={'version':1,'models_sha256':plan['models_sha256'],'models':models,'audit':summary,'histories':histories,'rows':1797,'fitting_rows':3823}
Path('docs/data.js').write_text('globalThis.CLUSTER_DATA='+dumps(bundle)+';\n')
Path('examples/predict.json').write_text(dumps({'pixels':r['arms']['raw_plus']['examples'][0]['pixels']}))

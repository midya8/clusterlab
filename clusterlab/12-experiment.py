"""Freeze choices before test access; serialize full witnesses, not just winners."""
from pathlib import Path
import hashlib
import numpy as np
from . import data
from .io import read,save as write
from .kmeans import fit,distances
from .preprocess import fit_transform,transform,inverse
from .metrics import ari,nmi,contingency,silhouette
ROOT=Path(__file__).parent
ARMS={'raw_plus':('raw','plus'),'raw_uniform':('raw','uniform'),'pca16_plus':('pca','plus'),'white16_plus':('white','plus')}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def train(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    protocol={'version':1,'k':10,'rank':16,'seeds':list(range(100,110)),'max_iter':200,'selection':'minimum fitting inertia within each arm, first run on ties','arms':ARMS,'data_manifest':read(ROOT/'data/manifest.json'),'test_usage':'not loaded during training; no test-based selection','labels_usage':'ignored by training; external evaluation only','fitting_rows':3823}
    write(out/'protocol.json',protocol)
    x,_=data.load('train');models={}
    for arm,(mode,init) in ARMS.items():
        p=fit_transform(x,mode);z=transform(x,p);m=fit(z,10,tuple(protocol['seeds']),init,200)
        m['representation']=p;m['pixel_centers']=inverse(m['centers'],p).tolist();models[arm]=m
    write(out/'models.json',models)
    write(out/'evaluation-plan.json',{'version':1,'protocol_sha256':sha(out/'protocol.json'),'models_sha256':sha(out/'models.json'),'cases':['original_official_test'],'metrics':['assigned_squared_distance_in_arm_geometry','ari','arithmetic_nmi','all_row_euclidean_silhouette_in_arm_geometry'],'test_rows':1797,'test_row_examples':list(range(20)),'cross_geometry_warning':'do not compare inertia or silhouette as if metric were unchanged'})
    return models


def evaluate(out):
    out=Path(out);plan=read(out/'evaluation-plan.json')
    for name in ('protocol','models'):
        if sha(out/(name+'.json'))!=plan[name+'_sha256']:raise ValueError('frozen artifact mismatch before test load')
    models=read(out/'models.json');x,y=data.load('test');report={'plan':plan,'rows':len(x),'arms':{}}
    for name,m in models.items():
        z=transform(x,m['representation']);d=distances(z,m['centers']);labels=d.argmin(axis=1);err=d[np.arange(len(x)),labels]
        sil=silhouette(z,labels)
        fitruns=m['runs'];selected=fitruns[m['chosen']]
        stability=[ari(selected['labels'],r['labels']) for r in fitruns]
        # Fixed 10x10 composition, even if some class/cluster were absent.
        counts=np.zeros((10,10),int);np.add.at(counts,(labels,y),1)
        report['arms'][name]={'ari':ari(y,labels),'nmi':nmi(y,labels),'silhouette':float(sil.mean()),'mean_squared_distance':float(err.mean()),'counts':np.bincount(labels,minlength=10).tolist(),'composition_cluster_by_digit':counts.tolist(),'labels':labels.tolist(),'squared_distances':err.tolist(),'silhouettes':sil.tolist(),'fit_ari_to_selected_by_restart':stability,'fit_best_inertia':selected['inertia'],'fit_worst_inertia':max(r['inertia'] for r in fitruns),'fit_selected_restart':m['chosen'],'examples':[{'row':i,'pixels':x[i].tolist(),'digit':int(y[i]),'cluster':int(labels[i]),'distances':d[i].tolist()} for i in plan['test_row_examples']]}
    write(out/'report.json',report);return report


def predict(pixels,models):
    x=np.asarray(pixels,dtype=float)
    if x.shape!=(64,) or not np.isfinite(x).all() or (x<0).any() or (x>1).any():raise ValueError('64 finite pixels in [0,1] required')
    out={}
    for arm,m in models.items():
        d=distances(transform(x[None,:],m['representation']),m['centers'])[0]
        out[arm]={'cluster':int(np.argmin(d)),'squared_distances':d.tolist()}
    return out

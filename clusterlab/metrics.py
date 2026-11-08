"""Independent contingency scores and all-pair Euclidean silhouettes."""
import math
import numpy as np
from .kmeans import matrix,distances


def contingency(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.ndim!=1 or b.shape!=a.shape or not len(a):raise ValueError('equal nonempty label vectors required')
    ua,ia=np.unique(a,return_inverse=True);ub,ib=np.unique(b,return_inverse=True)
    c=np.zeros((len(ua),len(ub)),dtype=np.int64);np.add.at(c,(ia,ib),1)
    return c


def ari(a,b):
    c=contingency(a,b);n=int(c.sum())
    choose=lambda z:int(z)*(int(z)-1)//2
    if n<2:return 1.
    both=sum(choose(z) for z in c.flat);ra=sum(choose(z) for z in c.sum(axis=1));cb=sum(choose(z) for z in c.sum(axis=0));pairs=choose(n)
    # Integer cross multiplication preserves exact degeneracy and avoids subtraction loss.
    den=(ra+cb)*pairs-2*ra*cb
    return (2*(both*pairs-ra*cb)/den) if den else 1.


def nmi(a,b):
    c=contingency(a,b);n=int(c.sum());r=c.sum(axis=1);s=c.sum(axis=0)
    h=lambda x:-sum(v/n*math.log(v/n) for v in x if v)
    den=h(r)+h(s)
    if den==0:return 1.
    mi=sum(int(c[i,j])/n*math.log(int(c[i,j])*n/(int(r[i])*int(s[j]))) for i,j in zip(*np.nonzero(c)))
    return 2*mi/den


def silhouette(x,labels,chunk=128):
    x=matrix(x);labels=np.asarray(labels)
    if labels.shape!=(len(x),) or type(chunk) is not int or chunk<1:raise ValueError('invalid silhouette inputs')
    ids=np.unique(labels)
    if not 2<=len(ids)<len(x):raise ValueError('silhouette needs 2..n-1 groups')
    members=[np.flatnonzero(labels==j) for j in ids];out=np.zeros(len(x))
    for start in range(0,len(x),chunk):
        end=min(len(x),start+chunk);d=np.sqrt(distances(x[start:end],x))
        for row in range(start,end):
            own=int(np.flatnonzero(ids==labels[row])[0]);group=members[own]
            if len(group)==1:continue
            # Explicitly exclude self, including any small numerical diagonal residue.
            a=d[row-start,group[group!=row]].mean()
            b=min(d[row-start,g].mean() for j,g in enumerate(members) if j!=own)
            out[row]=(b-a)/max(a,b) if max(a,b)>0 else 0
    return out

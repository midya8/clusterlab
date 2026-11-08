"""Exact integer assignment certificates for rounded binary64 cost matrices.

No clustering, fitting, data loading, or digit-label access happens here.
"""
import math
import numpy as np
from .metrics import ari
from .preprocess import inverse
from .io import digest


def integer_assignment(costs):
    """Shortest augmenting-path Hungarian method, exact unbounded integers.

    Return one deterministic optimum and feasible duals. Ties retain the first
    predecessor and visit the lowest column. This is not a uniqueness claim.
    """
    c=[list(row) for row in costs]; n=len(c)
    if not 1<=n<=64 or any(len(row)!=n or any(type(x) is not int for x in row) for row in c):
        raise ValueError('expected square integer costs, size 1..64')
    u=[0]*(n+1);v=[0]*(n+1);p=[0]*(n+1);way=[0]*(n+1)
    for i in range(1,n+1):
        p[0]=i;j0=0;slack=[None]*(n+1);used=[False]*(n+1)
        while True:
            used[j0]=True;i0=p[j0];delta=None;j1=0
            for j in range(1,n+1):
                if not used[j]:
                    current=c[i0-1][j-1]-u[i0]-v[j]
                    if slack[j] is None or current<slack[j]:slack[j]=current;way[j]=j0
                    if delta is None or slack[j]<delta:delta=slack[j];j1=j
            for j in range(n+1):
                if used[j]:u[p[j]]+=delta;v[j]-=delta
                elif j:slack[j]-=delta
            j0=j1
            if not p[j0]:break
        while True:
            j1=way[j0];p[j0]=p[j1];j0=j1
            if not j0:break
    assignment=[0]*n
    for j in range(1,n+1):assignment[p[j]-1]=j-1
    result={'assignment':assignment,'row_dual':u[1:],'column_dual':v[1:],
            'total_units':sum(c[i][assignment[i]] for i in range(n))}
    verify_assignment(c,result)
    return result


def verify_assignment(c,w):
    """Independent exact feasibility, tight-edge and equal-objective checks."""
    n=len(c);a=w['assignment'];u=w['row_dual'];v=w['column_dual']
    if len(a)!=n or len(u)!=n or len(v)!=n or any(type(x) is not int for x in a+u+v) or sorted(a)!=list(range(n)):
        raise ValueError('invalid assignment or potentials')
    if any(u[i]+v[j]>c[i][j] for i in range(n) for j in range(n)):
        raise ValueError('dual infeasible')
    if any(u[i]+v[a[i]]!=c[i][a[i]] for i in range(n)):
        raise ValueError('assigned edge is not tight')
    if sum(c[i][a[i]] for i in range(n))!=w['total_units'] or sum(u)+sum(v)!=w['total_units']:
        raise ValueError('primal/dual objective mismatch')
    return True


def centroid_certificate(left,right):
    """Round each direct squared distance to binary64, then solve EXACTLY.

    The certificate covers this saved rounded matrix, not ideal real-number
    squared distances. A common power-of-two denominator introduces no further
    quantization. Never infer cluster-level semantic correspondence from it.
    """
    a=np.asarray(left,dtype=float);b=np.asarray(right,dtype=float)
    if a.ndim!=2 or a.shape!=b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('same-sized finite center matrices required')
    with np.errstate(over='ignore',invalid='ignore'):
        distances=np.sum((a[:,None,:]-b[None,:,:])**2,axis=2)
    if not np.isfinite(distances).all():raise ValueError('squared-distance overflow')
    ratios=[[float(x).as_integer_ratio() for x in row] for row in distances]
    scale=max(d for row in ratios for _,d in row)
    units=[[num*(scale//den) for num,den in row] for row in ratios]
    w=integer_assignment(units)
    # JSON integer strings protect exactness when a browser parses the witness.
    return {'cost_units':[[str(x) for x in row] for row in units],'scale':str(scale),
            'assignment':w['assignment'],'row_dual':[str(x) for x in w['row_dual']],
            'column_dual':[str(x) for x in w['column_dual']],
            'total_units':str(w['total_units']),'mean_squared_displacement':w['total_units']/scale/len(a)}


def restart_audit(models):
    """Only saved fitting partitions/centers; no raw data or evaluation report."""
    arms={}
    for name,model in models.items():
        runs=model['runs'];pairs={};values=[]
        for i in range(len(runs)):
            for j in range(i,len(runs)):
                w=centroid_certificate(runs[i]['centers'],runs[j]['centers'])
                w.update({'left':i,'right':j,'ari':ari(runs[i]['labels'],runs[j]['labels'])})
                pairs[f'{i}-{j}']=w
                if i!=j:values.append(w['ari'])
        arms[name]={'chosen':model['chosen'],'pairs':pairs,
           'pairwise_ari_min':min(values),'pairwise_ari_max':max(values),
           'pairwise_ari_mean':math.fsum(values)/len(values),
           'representation':model['representation'],
           'runs':[{'seed':r['seed'],'inertia':r['inertia'],'centers':r['centers'],
                    'pixel_centers':inverse(r['centers'],model['representation']).tolist()} for r in runs]}
    return {'schema':1,'models_sha256':digest(models),'fitting_rows':3823,
            'scope':'same-data initialization sensitivity; no sampling stability or true-digit accuracy',
            'cost_semantics':'exact assignment for saved rounded binary64 squared-distance matrices',
            'arms':arms}

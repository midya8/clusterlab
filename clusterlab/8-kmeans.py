"""Lloyd descent with deterministic tie rules and complete restart witnesses."""
import numpy as np


def matrix(x):
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():raise ValueError('expected finite nonempty matrix')
    return x


def distances(x,c):
    x,c=matrix(x),matrix(c)
    if x.shape[1]!=c.shape[1]:raise ValueError('feature mismatch')
    # Direct differences avoid cancellation of the squared-norm identity.
    with np.errstate(over='ignore',invalid='ignore'):
        d=np.sum((x[:,None,:]-c[None,:,:])**2,axis=2)
    if not np.isfinite(d).all():raise ValueError('distance overflow')
    return d


def initialize(x,k,seed,method='plus'):
    x=matrix(x)
    if type(k) is not int or not 1<=k<=len(x) or method not in ('plus','uniform'):raise ValueError('invalid initialization')
    if len(np.unique(x,axis=0))<k:raise ValueError('k exceeds distinct rows')
    if type(seed) is not int or seed<0:raise ValueError('seed must be a nonnegative integer')
    rng=np.random.default_rng(seed)
    if method=='uniform':
        # Sample distinct VALUES, preserving first occurrence order.
        _,ids=np.unique(x,axis=0,return_index=True);ids=np.sort(ids)
        selected=rng.choice(ids,k,replace=False).tolist()
    else:
        selected=[int(rng.integers(len(x)))];d=distances(x,x[selected]).min(axis=1)
        while len(selected)<k:
            u=float(rng.random())*float(d.sum())
            j=int(np.searchsorted(np.cumsum(d),u,side='right'))
            j=min(j,len(x)-1)
            if d[j]<=0:j=int(np.flatnonzero(d>0)[0])
            selected.append(j);d=np.minimum(d,distances(x,x[[j]])[:,0])
    return x[selected].copy(),selected


def lloyd(x,centers,max_iter=200):
    x,c=matrix(x),matrix(centers).copy()
    if x.shape[1]!=c.shape[1] or len(c)>len(x) or type(max_iter) is not int or max_iter<1:raise ValueError('invalid Lloyd config')
    history=[];last=None;repairs=[]
    for iteration in range(max_iter+1):
        d=distances(x,c);labels=np.argmin(d,axis=1);loss=float(d[np.arange(len(x)),labels].sum())
        counts=np.bincount(labels,minlength=len(c))
        history.append({'iteration':iteration,'inertia':loss,'counts':counts.tolist()})
        if last is not None and np.array_equal(labels,last):
            reason='stable_assignments';break
        if iteration==max_iter:
            reason='iteration_limit';break
        last=labels.copy();new=c.copy()
        for j in range(len(c)):
            if counts[j]:new[j]=x[labels==j].mean(axis=0)
        # Empty centers cannot increase cost: all nonempty means remain available.
        # Reuse the farthest currently represented point, then update distances.
        nearest=distances(x,new[counts>0]).min(axis=1)
        for j in np.flatnonzero(counts==0):
            row=int(np.argmax(nearest));new[j]=x[row]
            repairs.append({'iteration':iteration,'cluster':int(j),'row':row})
            nearest=np.minimum(nearest,distances(x,x[[row]])[:,0])
        c=new
    if any(b['inertia']>a['inertia']+1e-9*max(1,a['inertia']) for a,b in zip(history,history[1:])):raise ArithmeticError('nonmonotone Lloyd objective')
    return {'centers':c.tolist(),'labels':labels.tolist(),'inertia':loss,'history':history,'repairs':repairs,'reason':reason}


def fit(x,k=10,seeds=tuple(range(100,110)),method='plus',max_iter=200):
    x=matrix(x)
    if not seeds or any(type(s) is not int or s<0 for s in seeds) or len(set(seeds))!=len(seeds):raise ValueError('need distinct restart seeds')
    runs=[]
    for seed in seeds:
        c,ids=initialize(x,k,int(seed),method)
        run=lloyd(x,c,max_iter);run.update(seed=int(seed),initial_rows=ids);runs.append(run)
    chosen=min(range(len(runs)),key=lambda i:(runs[i]['inertia'],i))
    return {'k':k,'method':method,'chosen':chosen,'runs':runs,'centers':runs[chosen]['centers']}

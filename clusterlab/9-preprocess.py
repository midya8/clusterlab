"""Fitting-only PCA. Whitening changes the metric; it is not neutral scaling."""
import numpy as np
from .kmeans import matrix


def fit_transform(x,mode='raw',rank=16):
    x=matrix(x)
    if mode not in ('raw','pca','white') or (mode!='raw' and (type(rank) is not int or not 1<=rank<=min(x.shape))):raise ValueError('invalid representation')
    if mode=='raw':return {'mode':mode,'mean':np.zeros(x.shape[1]).tolist(),'basis':np.eye(x.shape[1]).tolist(),'scale':np.ones(x.shape[1]).tolist()}
    mean=x.mean(axis=0);_,s,vt=np.linalg.svd(x-mean,full_matrices=False)
    basis=vt[:rank].T.copy()
    # Fix arbitrary SVD signs by the largest absolute loading, first index on ties.
    for j in range(rank):
        if basis[np.argmax(abs(basis[:,j])),j]<0:basis[:,j]*=-1
    scale=s[:rank]/np.sqrt(len(x)-1) if mode=='white' else np.ones(rank)
    if (scale<=1e-12).any():raise ValueError('degenerate whitened direction')
    return {'mode':mode,'mean':mean.tolist(),'basis':basis.tolist(),'scale':scale.tolist(),'singular_values':s.tolist(),'rank':rank}


def transform(x,model):
    x=matrix(x);mean=np.asarray(model['mean']);b=np.asarray(model['basis']);s=np.asarray(model['scale'])
    if b.ndim!=2 or mean.shape!=(x.shape[1],) or b.shape[0]!=x.shape[1] or s.shape!=(b.shape[1],) or not np.isfinite(b).all() or not np.isfinite(mean).all() or not np.isfinite(s).all() or (s<=0).any():raise ValueError('invalid representation model')
    return ((x-mean)@b)/s


def inverse(z,model):
    return (matrix(z)*np.asarray(model['scale']))@np.asarray(model['basis']).T+np.asarray(model['mean'])

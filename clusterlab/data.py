"""Original UCI numeric split, content hashes and deterministic training-only split."""
from pathlib import Path
import hashlib
import numpy as np
from .io import read
ROOT=Path(__file__).parent


def load(part='train'):
    if part not in ('train','test'):raise ValueError('partition must be train or test')
    path=ROOT/'data'/('optdigits.tra' if part=='train' else 'optdigits.tes')
    manifest=read(ROOT/'data/manifest.json')
    raw=path.read_bytes()
    expected=manifest[path.name]['sha256']
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('original numeric data hash mismatch')
    a=np.loadtxt(path,delimiter=',',dtype=int)
    n=3823 if part=='train' else 1797
    if a.shape!=(n,65) or (a[:,:64]<0).any() or (a[:,:64]>16).any() or (a[:,64]<0).any() or (a[:,64]>9).any():raise ValueError('invalid original data')
    return a[:,:64]/16.,a[:,64]


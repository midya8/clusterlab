"""Five offline CLIs: demo, train, evaluate, predict, stability."""
import argparse,json
from pathlib import Path
from .experiment import train,evaluate,predict,ROOT
from .io import read,save
from .stability import restart_audit

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='cmd',required=True)
    s.add_parser('demo')
    for cmd in ['train','evaluate']:s.add_parser(cmd).add_argument('--out',type=Path,required=True)
    q=s.add_parser('predict');q.add_argument('input',type=Path);q.add_argument('--models',type=Path,default=ROOT/'artifacts/models.json')
    q=s.add_parser('stability');q.add_argument('--models',type=Path,default=ROOT/'artifacts/models.json');q.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    try:
        if a.cmd=='stability':save(a.out,restart_audit(read(a.models)));print('wrote label-free restart audit, exact rounded-matrix certificates')
        elif a.cmd=='train':train(a.out);print('trained four fixed arms without test access')
        elif a.cmd=='evaluate':evaluate(a.out);print('evaluated frozen models')
        elif a.cmd=='predict':print(json.dumps(predict(read(a.input)['pixels'],read(a.models)),allow_nan=False))
        else:
            r=read(ROOT/'artifacts/report.json')
            print('Cluster IDs are not digit labels. External audit only; no classifier accuracy.')
            for arm,v in r['arms'].items():print(f"{arm}: ARI {v['ari']:.6f}; NMI {v['nmi']:.6f}; silhouette {v['silhouette']:.6f}")
    except (ValueError,KeyError,OSError) as e:p.exit(2,str(e)+'\n')

if __name__=='__main__':main()

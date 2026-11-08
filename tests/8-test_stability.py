"""Assignment correctness is checked without calling the Hungarian solver."""
import itertools,json,math,random,unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
from clusterlab.stability import integer_assignment,verify_assignment,centroid_certificate,restart_audit
from clusterlab.io import read,dumps,digest
from clusterlab.experiment import ROOT
from clusterlab.metrics import ari


def subset_minimum(c):
    # Assign rows by population count. No augmenting paths or dual potentials.
    dp={0:0};n=len(c)
    for row in range(n):
        nxt={}
        for mask,total in dp.items():
            for col in range(n):
                if not mask>>col&1:
                    key=mask|1<<col;v=total+c[row][col]
                    if key not in nxt or v<nxt[key]:nxt[key]=v
        dp=nxt
    return dp[(1<<n)-1]


def unpack(w):
    return [[int(x) for x in r] for r in w['cost_units']],{k:([int(x) for x in w[k]] if k in ['row_dual','column_dual'] else int(w[k]) if k=='total_units' else w[k]) for k in ['assignment','row_dual','column_dual','total_units']}


class AssignmentTests(unittest.TestCase):
    def test_random_integer_exhaustive(self):
        rng=random.Random(51)
        for n in range(1,7):
            for _ in range(12):
                c=[[rng.randrange(-9,10)*2**80+rng.randrange(3) for _ in range(n)] for _ in range(n)]
                w=integer_assignment(c)
                expected=min(sum(c[i][p[i]] for i in range(n)) for p in itertools.permutations(range(n)))
                self.assertEqual(w['total_units'],expected);self.assertTrue(verify_assignment(c,w))
    def test_zero_ties_and_negative_costs(self):
        for n in [1,2,10]:
            self.assertEqual(integer_assignment([[0]*n for _ in range(n)])['assignment'],list(range(n)))
        self.assertEqual(integer_assignment([[-4,-3],[-2,-10]])['total_units'],-14)
    def test_invalid_costs(self):
        for c in [[],[[1,2]],[[True]],[[1.0]],[[float('nan')]],[[0]*65 for _ in range(65)]]:
            with self.assertRaises(ValueError):integer_assignment(c)
    def test_certificate_tampering(self):
        c=[[0,4],[5,0]];w=integer_assignment(c)
        for key,value in [('assignment',[0,0]),('assignment',[1,0]),('row_dual',[1,0]),('column_dual',[1,0]),('total_units',1)]:
            bad=deepcopy(w);bad[key]=value
            with self.assertRaises(ValueError):verify_assignment(c,bad)
    def test_label_permutation_and_greedy_failure(self):
        a=[[0.0],[10.0],[20.0]];b=[a[2],a[0],a[1]]
        w=centroid_certificate(a,b);self.assertEqual(w['assignment'],[1,2,0]);self.assertEqual(w['total_units'],'0')
        # Both left centers prefer right center 0. A rowwise nearest rule is not a bijection.
        c=[[1,4],[0,9]];self.assertEqual([min(range(2),key=lambda j:r[j]) for r in c],[0,0])
        self.assertEqual(integer_assignment(c)['assignment'],[1,0])
        self.assertEqual(ari([0,0,1,1,2,2],[2,2,0,0,1,1]),1)
    def test_binary64_costs_have_no_extra_quantization(self):
        a=[[0.1,0.3],[0.6,0.9]];b=[[0.3,0.1],[0.5,0.8]];w=centroid_certificate(a,b);scale=int(w['scale']);c,z=unpack(w)
        for i in range(2):
            for j in range(2):
                num,den=float(np.sum((np.asarray(a[i])-b[j])**2)).as_integer_ratio()
                self.assertEqual(c[i][j]*den,num*scale)
        self.assertTrue(verify_assignment(c,z))
    def test_subnormals_overflow_and_dimensions(self):
        tiny=math.sqrt(float.fromhex('0x0.0000000000001p-1022'))
        w=centroid_certificate([[0.0]],[[tiny]]);self.assertGreater(int(w['scale']),2**1000)
        self.assertEqual(int(w['cost_units'][0][0]),1)
        for a,b in [([[1e308]],[[-1e308]]),([[0]],[[0,1]]),([[float('nan')]],[[0]]),([],[])]:
            with self.assertRaises(ValueError):centroid_certificate(a,b)
    def test_all_220_real_certificates_subset_oracle(self):
        saved=read(ROOT/'artifacts/stability.json');models=read(ROOT/'artifacts/models.json')
        self.assertEqual(saved['models_sha256'],digest(models));n=0
        for arm,a in saved['arms'].items():
            self.assertEqual(len(a['pairs']),55);values=[]
            for w in a['pairs'].values():
                c,z=unpack(w);self.assertTrue(verify_assignment(c,z));self.assertEqual(subset_minimum(c),z['total_units']);n+=1
                self.assertEqual(w['ari'],ari(models[arm]['runs'][w['left']]['labels'],models[arm]['runs'][w['right']]['labels']))
                if w['left']!=w['right']:values.append(w['ari'])
            self.assertEqual(a['pairwise_ari_min'],min(values));self.assertEqual(a['pairwise_ari_mean'],math.fsum(values)/45)
        self.assertEqual(n,220)
    def test_no_dataset_or_holdout_access_and_exact_reproduction(self):
        models=read(ROOT/'artifacts/models.json');saved=read(ROOT/'artifacts/stability.json')
        # The API receives frozen artifacts; every access to data loading is forbidden.
        with patch('clusterlab.data.load',side_effect=AssertionError('no data loader')):
            actual=restart_audit(models)
        self.assertEqual(dumps(actual),dumps(saved))
        bundle=Path('docs/stability-data.js').read_text()
        browser=json.loads(bundle.split('=',1)[1].rsplit(';',1)[0])
        for key,value in actual.items():self.assertEqual(browser[key],value)
        self.assertEqual(len(browser['examples']),20)
        self.assertNotIn('digit',browser['examples'][0])
        self.assertNotIn('composition_cluster_by_digit',bundle);self.assertNotIn('labels',actual['arms']['raw_plus']['runs'][0])
    def test_chosen_models_and_original_artifacts_unchanged(self):
        m=read(ROOT/'artifacts/models.json');a=read(ROOT/'artifacts/stability.json')
        for k,v in m.items():
            chosen=min(range(10),key=lambda i:(v['runs'][i]['inertia'],i))
            self.assertEqual(chosen,v['chosen']);self.assertEqual(chosen,a['arms'][k]['chosen'])
            self.assertEqual(v['centers'],a['arms'][k]['runs'][chosen]['centers'])

if __name__=='__main__':unittest.main()

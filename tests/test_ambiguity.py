"""Independent full-cost dynamic programs, not reduced-cost counting logic."""
import itertools,math,random,unittest,json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from fractions import Fraction
from clusterlab.stability import integer_assignment,verify_assignment
from clusterlab.ambiguity import ambiguity,optimal_count,audit
from clusterlab.io import read,dumps,digest
from clusterlab.experiment import ROOT


def top_two(c):
    # Retain the two cheapest distinct partial permutations, including equal cost.
    dp={0:[0]};n=len(c)
    for i in range(n):
        nxt={}
        for mask,values in dp.items():
            for j in range(n):
                if not mask>>j&1:
                    key=mask|1<<j;nxt.setdefault(key,[]).extend(v+c[i][j] for v in values)
        dp={k:sorted(v)[:2] for k,v in nxt.items()}
    return dp[(1<<n)-1]


def optimum_and_count(c):
    # Counts all minimal-cost assignments using original costs, no dual graph.
    dp={0:(0,1)};n=len(c)
    for i in range(n):
        nxt={}
        for mask,(total,count) in dp.items():
            for j in range(n):
                if mask>>j&1:continue
                key=mask|1<<j;v=total+c[i][j]
                if key not in nxt or v<nxt[key][0]:nxt[key]=(v,count)
                elif v==nxt[key][0]:nxt[key]=(v,nxt[key][1]+count)
        dp=nxt
    return dp[(1<<n)-1]


def excluded_minimum(c,row,col):
    dp={0:0};n=len(c)
    for i in range(n):
        nxt={}
        for mask,total in dp.items():
            for j in range(n):
                if mask>>j&1 or (i==row and j==col):continue
                key=mask|1<<j;v=total+c[i][j]
                if key not in nxt or v<nxt[key]:nxt[key]=v
        dp=nxt
    return dp[(1<<n)-1]


def check_exclusion(c,e):
    d=deepcopy(c);i,j=e['forbidden_row'],e['forbidden_column'];d[i][j]=int(e['penalty_units'])
    w={k:([int(x) for x in e[k]] if k in ['row_dual','column_dual'] else int(e[k]) if k=='total_units' else e[k]) for k in ['assignment','row_dual','column_dual','total_units']}
    return verify_assignment(d,w)


class AmbiguityTests(unittest.TestCase):
    def test_exhaustive_random_signed_huge_costs(self):
        rng=random.Random(74)
        for n in range(2,7):
            for _ in range(7):
                c=[[rng.randrange(-5,6)*2**90+rng.randrange(2) for _ in range(n)] for _ in range(n)]
                w=integer_assignment(c);a=ambiguity(c,w);values=sorted(sum(c[i][p[i]] for i in range(n)) for p in itertools.permutations(range(n)))
                self.assertEqual(int(a['gap_units']),values[1]-values[0]);self.assertEqual(int(a['optimal_count']),values.count(values[0]))
                for e in a['exclusions']:self.assertTrue(check_exclusion(c,e));self.assertEqual(int(e['total_units']),excluded_minimum(c,e['forbidden_row'],e['forbidden_column']))
    def test_all_zero_and_block_ties(self):
        for n in range(2,11):
            a=ambiguity([[0]*n for _ in range(n)]);self.assertEqual(a['optimal_count'],str(math.factorial(n)));self.assertFalse(a['unique']);self.assertEqual(a['gap_units'],'0')
        a=ambiguity([[0,0,5],[0,0,5],[5,5,0]]);self.assertEqual(a['optimal_count'],'2')
    def test_tight_unused_edge_does_not_prove_alternate_optimum(self):
        c=[[0,0],[1,0]];w=integer_assignment(c)
        self.assertEqual(c[0][1],w['row_dual'][0]+w['column_dual'][1])
        a=ambiguity(c,w);self.assertTrue(a['unique']);self.assertEqual(a['optimal_count'],'1');self.assertEqual(a['gap_units'],'1')
    def test_singleton_has_no_distinct_assignment(self):
        a=ambiguity([[-100]])
        self.assertTrue(a['unique']);self.assertEqual(a['optimal_count'],'1');self.assertEqual(a['exclusions'],[]);self.assertIsNone(a['gap_units']);self.assertIsNone(a['best_distinct']);self.assertIsNone(a['strict_sufficient_cost_radius'])
    def test_penalty_bound_signed_and_large_range(self):
        for c in [[[-100,2],[3,-1]],[[2**200,-2**200],[4,5]]]:
            a=ambiguity(c);n=len(c);low=min(map(min,c));high=max(map(max,c))
            for e in a['exclusions']:
                penalty=int(e['penalty_units']);self.assertGreater(penalty+(n-1)*low,n*high);self.assertNotEqual(e['assignment'][e['forbidden_row']],e['forbidden_column'])
    def test_sufficient_radius_strict_not_a_sharp_claim(self):
        c=[[0,1],[1,0]];a=ambiguity(c);rho=Fraction(**{k:int(v) for k,v in a['strict_sufficient_cost_radius'].items()});self.assertEqual(rho,Fraction(1,2))
        # Exhaust all 3^4 perturbation patterns below the conservative bound.
        for delta in itertools.product([Fraction(-49,100),Fraction(0),Fraction(49,100)],repeat=4):
            d=[[Fraction(c[i][j])+delta[2*i+j] for j in range(2)] for i in range(2)]
            self.assertLess(d[0][0]+d[1][1],d[0][1]+d[1][0])
        # At the boundary, admissible entrywise changes can create a tie.
        d=[[Fraction(1,2),Fraction(1,2)],[Fraction(1,2),Fraction(1,2)]];self.assertEqual(d[0][0]+d[1][1],d[0][1]+d[1][0])
        larger=ambiguity([[0,1,100],[1,0,100],[100,100,0]])
        self.assertEqual(larger['strict_sufficient_cost_radius'],{'numerator':'1','denominator':'3'})
        # Only rows 0 and 1 move in the best replacement, so 1/3 is conservative.
    def test_all_220_real_best_distinct_and_counts_full_cost_dp(self):
        s=read(ROOT/'artifacts/stability.json');a=read(ROOT/'artifacts/ambiguity.json');self.assertEqual(a['stability_sha256'],digest(s));n=0
        for name,ps in a['arms'].items():
            for key,value in ps.items():
                w=s['arms'][name]['pairs'][key];c=[[int(x) for x in row] for row in w['cost_units']];top=top_two(c);best,count=optimum_and_count(c)
                self.assertEqual(best,top[0]);self.assertEqual(int(value['optimal_count']),count);self.assertEqual(int(value['gap_units']),top[1]-top[0]);self.assertEqual(value['unique'],count==1)
                for e in value['exclusions']:self.assertTrue(check_exclusion(c,e))
                n+=1
        self.assertEqual(n,220)
    def test_full_no_loader_reproduction_and_bundle(self):
        s=read(ROOT/'artifacts/stability.json');a=read(ROOT/'artifacts/ambiguity.json')
        with patch('clusterlab.data.load',side_effect=AssertionError('no dataset access')):actual=audit(s)
        self.assertEqual(dumps(actual),dumps(a))
        bundle=json.loads(Path('docs/ambiguity-data.js').read_text().split('=',1)[1].rsplit(';',1)[0])
        for key,value in actual.items():self.assertEqual(value,bundle[key])
    def test_mutated_exclusion_witness_rejected(self):
        c=[[0,1],[1,0]];e=ambiguity(c)['exclusions'][0]
        for key,value in [('row_dual',['0','0']),('total_units','99'),('assignment',[0,1]),('penalty_units','0')]:
            bad=deepcopy(e);bad[key]=value
            with self.assertRaises(ValueError):check_exclusion(c,bad)
    def test_size_and_invalid_inputs(self):
        for c in [[],[[0]*17 for _ in range(17)],[[True]],[[1.0]],[[1,2]]]:
            with self.assertRaises(ValueError):ambiguity(c)

if __name__=='__main__':unittest.main()

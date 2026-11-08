"""Derive uniqueness and best-distinct matching witnesses without raw data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from clusterlab.experiment import ROOT
from clusterlab.ambiguity import audit,ambiguity
from clusterlab.stability import integer_assignment
from clusterlab.io import read,save,dumps
r=audit(read(ROOT/'artifacts/stability.json'));save(ROOT/'artifacts/ambiguity.json',r)
fixtures={}
for name,costs in {'tied':[[0,0,5],[0,0,5],[5,5,0]],'fragile':[[0,1],[1,0]],'separated':[[0,20],[30,0]],'zero_reduced_but_unique':[[0,0],[1,0]]}.items():
 w=integer_assignment(costs);a=ambiguity(costs,w)
 fixtures[name]={'cost_units':[[str(v) for v in row] for row in costs],
 'scale':'1','assignment':w['assignment'],'row_dual':[str(v) for v in w['row_dual']],
 'column_dual':[str(v) for v in w['column_dual']],'total_units':str(w['total_units']),'audit':a}
bundle=dict(r);bundle['fixtures']=fixtures
Path('docs/ambiguity-data.js').write_text('globalThis.CLUSTER_AMBIGUITY='+dumps(bundle)+';\n')

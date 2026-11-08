"""Exact non-uniqueness and replacement audits of a fixed cost matrix.

Not sampling uncertainty or a certificate about ideal real-valued distances.
"""
from fractions import Fraction
from .stability import integer_assignment,verify_assignment
from .io import digest


def optimal_count(c,w):
    """Count perfect matchings in the exact zero-reduced-cost graph, n<=16.

    Every optimum uses only tight edges under any optimal dual. Conversely every
    all-tight perfect matching attains the dual objective and is optimal.
    """
    verify_assignment(c,w);n=len(c)
    if n>16:raise ValueError('exact counting is limited to size 16')
    dp={0:1}
    for i in range(n):
        nxt={}
        for mask,count in dp.items():
            for j in range(n):
                if not mask>>j&1 and c[i][j]==w['row_dual'][i]+w['column_dual'][j]:
                    key=mask|1<<j;nxt[key]=nxt.get(key,0)+count
        dp=nxt
    return dp.get((1<<n)-1,0)


def ambiguity(costs,selected=None):
    """Best *distinct* assignment, which may have exactly the same objective.

    All n exclusion subproblems overlap. They cover every distinct permutation,
    so their minimum is the cheapest replacement. They are not a K-best ranking.
    """
    c=[list(row) for row in costs]
    if not 1<=len(c)<=16 or any(len(row)!=len(c) or any(type(x) is not int for x in row) for row in c):raise ValueError('square integer cost matrix, size 1..16 required')
    w=integer_assignment(c) if selected is None else selected
    verify_assignment(c,w);n=len(c)
    if n>16:raise ValueError('ambiguity audit limited to size 16')
    count=optimal_count(c,w)
    if n==1:return {'optimal_count':str(count),'unique':True,'exclusions':[],
        'best_distinct':None,'gap_units':None,'strict_sufficient_cost_radius':None}
    low=min(map(min,c));high=max(map(max,c))
    # Any feasible allowed assignment costs <= n*high. An assignment containing
    # the penalty costs >= penalty+(n-1)*low > n*high, including signed matrices.
    penalty=n*high-(n-1)*low+1
    exclusions=[]
    for i,j in enumerate(w['assignment']):
        changed=[row[:] for row in c];changed[i][j]=penalty
        z=integer_assignment(changed)
        if z['assignment'][i]==j:raise ValueError('exclusion penalty proof failed')
        verify_assignment(changed,z)
        total=sum(c[r][col] for r,col in enumerate(z['assignment']))
        if total!=z['total_units']:raise ValueError('forbidden edge used')
        exclusions.append({'forbidden_row':i,'forbidden_column':j,
            'penalty_units':str(penalty),'assignment':z['assignment'],
            'row_dual':[str(x) for x in z['row_dual']],
            'column_dual':[str(x) for x in z['column_dual']],
            'total_units':str(total)})
    best=min(range(n),key=lambda i:(int(exclusions[i]['total_units']),i))
    gap=int(exclusions[best]['total_units'])-w['total_units']
    if gap<0 or (count==1)!=(gap>0):raise ValueError('uniqueness disagreement')
    radius=Fraction(gap,2*n)
    return {'optimal_count':str(count),'unique':count==1,'exclusions':exclusions,
            'best_distinct':best,'gap_units':str(gap),
            'strict_sufficient_cost_radius':{'numerator':str(radius.numerator),'denominator':str(radius.denominator)}}


def audit(stability):
    result={}
    for name,arm in stability['arms'].items():
        pairs={}
        for key,w in arm['pairs'].items():
            c=[[int(x) for x in r] for r in w['cost_units']]
            chosen={k:([int(x) for x in w[k]] if k in ['row_dual','column_dual'] else int(w[k]) if k=='total_units' else w[k]) for k in ['assignment','row_dual','column_dual','total_units']}
            pairs[key]=ambiguity(c,chosen)
        result[name]=pairs
    return {'schema':1,'stability_sha256':digest(stability),'arms':result,
            'scope':'exact uniqueness/replacement costs for saved rounded matrices; not statistical uncertainty'}

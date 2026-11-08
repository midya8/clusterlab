# Is an optimal centroid matching unique?

ClusterLab v0.3. Every v0.2 cost matrix and matching witness stays unchanged. This is an exact audit of those **saved rounded binary64 cost matrices**, not a new fit, test-label comparison, ideal-real arithmetic certificate, or statistical uncertainty estimate.

## Optimal is not unique

For a selected optimum p in an n-by-n integer cost matrix C, define E_i as all full permutations that **do not use** the selected edge (i,p(i)). Every permutation different from p belongs to at least one E_i. Thus

`best_distinct_cost = min_i min_{q in E_i} sum_r C[r,q(r)]`.

These n subproblems overlap. We use the cover to find the best *different permutation*, not the next strictly larger objective, a disjoint ranking, or an alternative-count sum. If another optimum exists the replacement gap is zero. For n=1 no distinct permutation exists: the gap, replacement and radius are null, not zero or infinity.

## A finite penalty with a proof

Let L=min C and H=max C. For one excluded edge, replace its cost with

`M = n*H - (n-1)*L + 1`.

All arithmetic is integer, including signed and huge costs. Any permutation avoiding that edge costs at most n*H. Any permutation using M costs at least M+(n-1)*L = n*H+1. For a complete square matrix with n>=2, an edge-avoiding permutation exists. Therefore the unconstrained optimum for this modified matrix cannot use the excluded edge. All remaining costs are original, so its objective is the exact constrained optimum. We export a primal permutation, feasible dual potentials, tight-edge equalities and equal primal/dual objectives for **each modified matrix**. Those duals need not be feasible for the original matrix; the witness names the penalty and forbidden edge explicitly.

There are 220 original matching problems, each with 10 exclusion problems, giving **2,200 exact constrained certificates**. Browser views reverse the matching by transposing costs, inverting permutations and swapping row/column duals, including each exclusion. Each original selected edge is covered exactly once. The browser validates the penalty formula and the forbidden-edge condition as well as every primal/dual certificate.

## Count the optima, not the zero edges

Under any optimal dual (u,v), every optimal permutation uses only edges satisfying `u_i+v_j=C_ij`: otherwise summing the positive reduced costs would exceed the dual bound. Conversely, every full all-tight permutation attains that bound and is optimal.

A column-mask dynamic program counts perfect matchings in this tight graph with exact integers. State (i,mask) records the number of ways to assign the first i rows to the columns in mask. It never counts partial paths as full permutations. The count is exponential in n, so this API is explicitly limited to n<=16; real matrices have n=10. The Hungarian assignment routine's broader size cap is not a claim that counting scales the same way.

An unused tight edge alone is insufficient. In `[[0,0],[1,0]]`, edge (0,1) is tight under the zero dual, but it cannot be extended to another all-tight perfect matching. There is exactly one optimum; the best distinct permutation costs 1 more. In `[[0,0,5],[0,0,5],[5,5,0]]`, there are exactly two optima and zero replacement gap. An all-zero n-by-n matrix has n! optima. These controls are visible in the browser and checked by exhaustive permutations.

## A conservative entrywise-cost bound

Let Delta be the best-distinct minus optimal total cost, in original distance-cost units. If every matrix entry changes by **strictly less** than rho=Delta/(2n), each permutation's total changes by less than n*rho, so the difference between the selected permutation and any competitor changes by less than 2n*rho=Delta. A unique selected optimum remains unique under that condition.

The exported bound is exact rational in the scaled integer cost units; dividing by the original power-of-two scale gives distance-cost units. Decimal rendering is approximate. This is a **sufficient bound**, not a sharp radius or the largest admissible perturbation. It ignores that two permutations might share many edges. For example, an isolated two-row alternative in a 3-by-3 matrix can make the 2n bound conservative. At a zero gap there is no positive strict-uniqueness bound. At the boundary, the strict guarantee does not apply. This is not a bound on input pixels, centroid positions, floating-point error, sampling variation, ARI, or real-data semantic stability.

## What the unchanged real fits show

All 220 saved rounded-cost matching problems have exactly one optimum. Among **distinct restart pairs** within each arm, the smallest total assignment replacement gaps are:

| Fixed arm | Seeds | Total squared-distance gap |
|---|---|---:|
| Raw k-means++ | 102 / 107 | 0.008537073166146047 |
| Raw uniform | 101 / 102 | 0.0077131088382196705 |
| PCA16 k-means++ | 101 / 103 | 0.16209251951934533 |
| Whitened PCA16 k-means++ | 107 / 109 | 0.004195423594570702 |

These are geometry-specific costs. Do not rank their magnitudes across feature representations. The smallest-gap control compares exact rational gaps only within the current arm. Unique geometric matching does not mean unique clustering, a large margin, useful groups, a statistically stable result, or agreement on each input. Original models, inertia winners, fitting ARI matrices and holdout scores are unchanged.

## Independent checks

- All 220 exact best-distinct objective gaps agree with a full-cost subset DP retaining the two cheapest **distinct permutations**, including equal objective values.
- All 220 optimum counts agree with a separate original-cost minimum-and-count DP, independent of reduced-cost tight-edge counting.
- Every one of 2,200 exclusion witnesses is checked exactly. Tiny signed/90-bit randomized matrices are independently exhaustively enumerated; their per-edge exclusion costs also agree with a forbidden-edge subset DP.
- Controls cover ties, n!, unused tight edges, n=1, malformed input, very large signed ranges, witness mutation, strict perturbation patterns and a conservative-not-sharp example.
- Browser BigInt checks all 400 directed pair views and 4,000 oriented exclusion witnesses. It rejects modified coverage, penalty, duals, cost, assignment, gap, count or radius. Real Chrome exercises both tied and unused-tight-edge fixture choices, smallest-gap selection, all model controls, arbitrary pixels and copy/download at 320/390/1280 widths.
- Combined project: **48 Python tests and 13 Node/real-Chrome tests**. Installed v0.3 wheel imports ten modules and runs six CLIs, including complete original training/evaluation and exact stability/ambiguity reproduction outside the checkout.

```bash
python -m clusterlab ambiguity --out /tmp/ambiguity.json
python examples/export_ambiguity.py
python -m unittest discover -s tests -v
node --test tests/*.cjs
```

## Sources and attribution

This uses established assignment/duality ideas and an explicit exclusion-cover proof, not a new algorithm. It does **not** implement or claim the performance of the specialized reoptimization method in the ranking paper.

- Pedersen, Nielsen & Andersen, [An algorithm for ranking assignments using reoptimization](https://www.research.relund.dk/publications/pdf/pedersen08a.pdf), manuscript May 9, 2007: ranked assignment scope, solution-set partitioning and distinct-solution alternatives. Our overlapping single-edge cover is simpler and does not enumerate K best solutions.
- Kleinberg/Tardos course material, Princeton, [Network Flow III](https://cs.princeton.edu/~wayne/kleinberg-tardos/pdf/07NetworkFlowIII.pdf): alternating paths and reduced-cost assignment optimization.
- Bertrand, Massias & Vayer, [Hungarian algorithm notes](https://mathurinm.github.io/assets/2024_ens_ot/hungarian.pdf), updated December 10, 2025: primal/dual feasibility and complementary slackness.
- Bricker, University of Iowa, [Hungarian optimality](https://user.engineering.uiowa.edu/~dbricker/Stacks_pdf1/Hungarian_optimality.pdf): equal primal/dual objective certificate.
- von Luxburg, [Clustering Stability: An Overview](https://arxiv.org/pdf/1007.1075), 2010, and [MPI publication record](https://is.mpg.de/ics/publications/6333): why optimization sensitivity must not be confused with sampling stability.
- [scikit-learn clustering evaluation documentation](https://scikit-learn.org/stable/modules/clustering.html): unchanged fitting-partition ARI interpretation; no runtime dependency.
- Original unchanged data: [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), Alpaydin & Kaynak (1998), CC BY 4.0. Derived matching summaries are new transformations. Existing dataset credit and numeric hashes remain intact. No endorsement; original code MIT.

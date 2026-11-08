# Restart stability, without a semantic shortcut

ClusterLab v0.2. Original 40 frozen fits remain unchanged. The audit reads only saved fitting assignments, centers and representations. It neither loads raw training/test rows nor reads evaluation labels. Browser example pixels come from the same first 20 official test rows as v0.1; their digit labels are omitted. Examples do not select the reported pairs or fitting winners.

## What is compared

Each arm keeps its original k=10, representation, seeds 100-109 and minimum-fitting-inertia selected run. Compare all 45 distinct unordered pairs per arm (180 total), plus 10 self-pairs per arm (40 total). A reversed view transposes the costs, swaps the dual vectors and inverts the permutation. It is not a separately fitted model. The ARI matrix compares the **same 3,823 fitting rows** under two fitted partitions, not clusters versus true digit classes. The earlier independent pair-count/null-permutation ARI tests still apply. There is no new external accuracy or holdout search.

| Arm | Minimum fitting ARI | Mean over 45 pairs | Maximum fitting ARI |
|---|---:|---:|---:|
| Raw k-means++ | 0.556972 | 0.756624 | 0.985823 |
| Raw uniform | 0.538878 | 0.745054 | 0.961776 |
| PCA16 k-means++ | 0.641175 | 0.823567 | 1.000000 |
| Whitened PCA16 k-means++ | 0.551122 | 0.741345 | 0.993739 |

Self-pairs are excluded from these summaries. These fixed-run descriptive values are not confidence intervals or a general initializer/representation ranking. In the raw-plus arm, seeds 106 and 107 are the least-agreeing pair; seed 107 remains the fitting-inertia winner. No resampled datasets, perturbation experiments or population stability are estimated. ARI is label-permutation invariant and chance-adjusted under its pair-count null, but pair-count weighting can obscure disagreement in small clusters. Stability does not establish a useful digit taxonomy or the correct number of clusters.

## Matching is a different question

For centers A and B in the **same representation**, form C_ij = sum_d (A_id - B_jd)^2 by direct coordinate differences and binary64 summation. Find a permutation p minimizing sum_i C_i,p(i). All ten centers have equal weight: this is neither population-weighted transport nor a maximum-overlap partition matching. It need not maximize assignment agreement for a particular point. Never compare movement between raw/PCA/whitened metrics as if units were unchanged. Inverse-mapped centroid images clip only their display, not costs or inference.

Raw center ID comparison can fail even when centers are merely permuted: [0,10,20] versus [20,0,10] has zero optimal matching cost and permutation [1,2,0]. Rowwise nearest choices can collide: costs [[1,4],[0,9]] prefer column 0 twice; the bijection optimum instead assigns [1,0] at cost 4.

## Exactness has a boundary

First compute each C_ij in binary64. Every resulting finite float has an exact integer numerator / power-of-two denominator. Set Q to the largest denominator and Cint_ij = numerator * (Q / denominator). This introduces **no further quantization**. The exact certificate is for this stored rounded matrix, not an ideal real-number distance matrix. Overflow/nonfinite inputs are rejected. Underflow can already have occurred in binary64; the integer step does not repair it.

The manual Hungarian shortest-augmenting-path solver uses Python unbounded integers. It returns a row-to-column permutation p and integer dual vectors u,v. The independent verifier checks:

1. p is a full bijection.
2. u_i + v_j <= Cint_ij for every edge (dual feasibility).
3. u_i + v_p(i) = Cint_i,p(i) on every assigned edge (complementary slackness).
4. Sum_i Cint_i,p(i) = sum_i u_i + sum_j v_j = saved total.

Any feasible assignment costs at least the feasible dual total. The matching attaining that bound is therefore optimal for the saved matrix. Divide by Q and k only for the displayed mean squared displacement. Dual gauge freedom and alternate optimal permutations remain; deterministic lowest-column traversal is not a uniqueness claim or a claim of lexicographically smallest optimum. This exact assignment result is **not** a global-optimality certificate for any k-means fit.

JSON stores integer matrices, scale, duals and objective as decimal **strings**. The browser checks them with BigInt without trusting floating-point subtraction or an epsilon. Rendering the decimal movement value uses Number and is approximate. Each witness includes the canonical models JSON SHA-256, seeds, original chosen seed, both actual inference vectors and exact matching data. The model files are already canonical JSON, so their byte hash equals the canonical model digest. A hash binds bytes, not the honesty of a researcher or a test-tuning history.

## Independent verification and packaging

The 220 exact objectives agree with a subset dynamic program over column masks, independent of augmenting paths and dual updates. Tiny signed/80-bit random integer matrices are additionally checked by all permutations. Tests cover ties, singletons, subnormal cost ratios, overflow, malformed input, invalid bijections, altered duals/totals and label permutations. All 220 Python certificates reproduce byte-for-byte without access to a dataset loader. The original saved fits and evaluation plans are preserved.

Scalar JavaScript inference is compared with native NumPy distances for all 40x20 = 800 public model/example pairs. Browser checks exercise all 4x10x10 = 400 directed pair views, verify the exact witness, inspect all 1,280 displayed pixels in a two-atlas view, edit custom inputs, copy denied/successful and download JSON, and expand the matrix/witness at 320/390/1280 widths. The matrix has a scrollable accessible values table. The installed v0.2 wheel imports nine modules and runs all five CLIs outside the checkout, including full original refit/holdout evaluation and stability JSON reproduction. Combined suite: 38 Python and 10 Node/real-Chrome tests.

Reproduce:

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
node --test tests/*.cjs
python -m clusterlab stability --out /tmp/stability.json
python examples/export_stability.py
```

## Sources

- Bertrand, Massias & Vayer, [Optimal Transport notes, Hungarian algorithm](https://mathurinm.github.io/assets/2024_ens_ot/hungarian.pdf), updated December 10, 2025: assignment LP, primal/dual feasibility, tight edges and augmenting paths. Established algorithm, implemented independently here.
- Bricker, University of Iowa, [Hungarian optimality](https://user.engineering.uiowa.edu/~dbricker/Stacks_pdf1/Hungarian_optimality.pdf): complementary slackness and equal primal/dual objectives. The exact integer certificate follows this mathematical condition, not a tolerance heuristic.
- von Luxburg (2010), [Clustering Stability: An Overview](https://arxiv.org/pdf/1007.1075), plus [MPI publication record](https://is.mpg.de/ics/publications/6333): stability protocols, initialization versus sampling effects, and limits of stability-based model selection. We do not claim its asymptotic results apply to these finite digits.
- [scikit-learn clustering metrics documentation](https://scikit-learn.org/stable/modules/clustering.html) and [chance-adjustment example](https://scikit-learn.org/stable/auto_examples/cluster/plot_adjusted_for_chance_measures.html): partition permutation invariance, ARI versus NMI chance adjustment. No scikit-learn runtime dependency.
- [NumPy SVD documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html): original fitting-only PCA representation; unchanged.
- Alpaydin & Kaynak (1998), [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), DOI 10.24432/C50P49, CC BY 4.0: original unchanged data and split. Derived restart summaries and center images are new transformations, not new source data.

No cited author or data provider endorses this project. No new clustering algorithm, statistical uncertainty guarantee or production recognition capability is claimed.

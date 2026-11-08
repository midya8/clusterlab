# ClusterLab method and audit protocol

## Scope and split

The UCI original numeric digit files contain 3,823 official training and 1,797 official test rows. Each row has 64 integer block counts in [0,16] and a class code. Features are divided by 16. The files are copied byte-for-byte, checked against `clusterlab/data/manifest.json` and distributed under CC BY 4.0 with attribution and change notes. No test row enters PCA, whitening, initialization, fitting, repair or restart selection. All training features are used; no validation search exists. Training labels are loaded for file schema validation but ignored by every fit operation. The full-refit test replaces them with an object that cannot be inspected.

The original UCI description says 30 people contributed to training and a different 13 to test. We do not have per-row writer IDs, do not independently reconstruct that division, and do not claim near-duplicate/style independence. Tests exclude exact feature duplicates across official partitions. Labels are used only after fitting for external ARI, NMI and cluster-composition counts, not for a test-derived cluster-to-digit classifier.

## Four fixed arms

Before test access, `train` writes a protocol naming k=10, rank 16, 200 updates and seeds 100-109 for four arms: raw pixels with D² initialization, raw pixels with uniform initialization, PCA16 with D², whitened PCA16 with D². Parameters are fixed by the experiment design, not chosen from test results. Every arm chooses minimum training inertia among its ten restarts; exact ties choose the earliest restart. No arm is selected as the final winner. Matching seeds does not mean matching initial centers or random-draw trajectories between different algorithms.

### Fitting-only representation

Center X with its fitting mean μ and compute NumPy reduced SVD `X-μ = U diag(s) Vᵀ`. PCA coordinates are `(x-μ)V16`. For reproducible orientation, each singular vector is flipped if its largest-absolute-loading entry is negative, lowest index on exact ties. Whitened coordinates divide component j by `s_j/sqrt(n-1)`, giving sample covariance I on fitting rows. A scale <= 1e-12 is rejected rather than silently divided. Raw coordinates are exactly the normalized 64 features, without centering or further scaling.

PCA discards directions and whitening reweights them. These change the metric; a smaller numerical inertia or larger silhouette after this change does not establish a better fit in the original metric. No whitening advantage is asserted. Inverse-mapped centers are `(center * scale)V16ᵀ+μ`; they can leave [0,1]. Only canvas/SVG shades are clipped to [0,1], never inference coordinates or saved centers. Raw centers are literal mean digit block counts; the drawing must not be read as a generative sample or class prototype guarantee.

## Initialization and Lloyd steps

The objective is `J(C)=sum_i min_j ||z_i-c_j||²` in the arm's own representation. Direct coordinate differences compute distances to avoid cancellation in a squared-norm expansion. Nonfinite inputs and numeric distance overflow are rejected. Assignment ties use the first center index.

For D² initialization choose the first row uniformly, then choose each next row proportionally to its squared distance from its nearest existing center. The saved initial row indices and seed make the draw inspectable. All-zero remaining mass means insufficient distinct feature vectors and is rejected at input validation. An edge fallback selects the first positive-distance row if floating cumulative arithmetic lands on a zero-weight boundary. This is a numeric guard, not a separate optimization. Uniform initialization samples distinct feature values without replacement, preserving first-occurrence row order before the draw; the original fitting data have no duplicate feature rows. On data with duplicates this is uniform over distinct values, not frequency-weighted rows.

Lloyd repeatedly assigns all points, records inertia and counts, and replaces nonempty centers by their assigned mean. For each empty cluster, it places a center on the fitting row farthest from the current nonempty/newly repaired centers; a distance tie chooses the first row. The next assignment cannot raise cost because all updated nonempty means remain available. A complete repair log records iteration, cluster and row. The real-data runs had no empty-cluster repair; an explicit duplicate-center fixture checks repair behavior.

Stop when assignments equal the previous iteration, or the update budget is exhausted. A history entry is an assignment/objective evaluation; entry 0 is before any mean update. On a budget stop the final labels match the returned centers, but means need not yet match those labels; the `iteration_limit` reason is retained. Every shipped real-data run stopped with stable assignments. Monotonicity is checked with `1e-9*max(1,previous inertia)` numeric tolerance, not used as the stopping criterion. The chosen centers and assignments are retained alongside all other runs.

## External and internal audits

ARI forms the contingency counts n_ab, row sizesa and column sizesb. Let S=sum choose(n_ab,2), A=sum choose(a,2), B=sum choose(b,2), T=choose(n,2). Then `ARI = 2(ST-AB)/((A+B)T-2AB)`, with integer cross-products before final division. Perfect degenerate partitions return 1. It is unchanged by label permutation and can be negative. The independent test enumerates all distinct label permutations for a small fixed-marginal null, computes raw pair agreements, and chance-adjusts that score independently.

Arithmetic NMI is `2 I(U;V)/(H(U)+H(V))`, with natural logs, zero terms excluded and both-zero-entropy partitions scored 1. It is not adjusted for chance and can prefer many small groups. Fixed k avoids a search over this effect but is not a full statistical correction. The test computes entropy and joint entropy independently. No arbitrary numeric cluster/digit ID equality is scored as accuracy.

For silhouette, a_i is the mean Euclidean distance to every other held-out row in the same cluster, and b_i is the smallest mean distance to another held-out cluster. `s_i=(b_i-a_i)/max(a_i,b_i)`. Singletons and zero-distance degeneracies score 0. All 1,797 rows are evaluated with all-pair distances, chunked to bound memory, not a subsample and not distances to centroids. Metric geometry changes across arms and favors compact/convex groups; this is a diagnostic, not ground truth.

The saved report includes all assignments, squared distances, silhouettes,10x10 composition matrices and fitting ARI-to-selected values for every restart. This last score measures assignment agreement between restarts on the same fitting rows, not sampling uncertainty. Every public row is selected by fixed index 0-19, not by score. All80model/example combinations have independent scalar JavaScript parity checks; editable browser inputs are recomputed, not nearest saved-example replay.

## Mutation guard and validation

Before test load, evaluation checks SHA256 of `protocol.json` and `models.json` against `evaluation-plan.json`. A mismatch fails before the loader. This detects accidental changes, not intentional leakage if a person rewrites the plan after seeing test results. The full-refit test forbids test access entirely and compares all 40 initializations, chosen runs, assignments and learned centers; the separate full-evaluation test reproduces every held-out distance/silhouette and composition matrix.

All28Python tests cover an independent scalar objective, exact six-point31partition oracle, D² cumulative sampling oracle, low-index ties, duplicate-value limits, empty repairs, monotonicity, budget reporting, PCA covariance/eigenspectrum/reconstruction, whitening geometry, strict JSON and frozen guards. Six Node/Chrome tests cover all80native cases and all 640 atlas pixel values, custom pixel edits, keyboard, labels not affecting inference, copy failure/success, download and expanded 320/390/1280 layout/touch controls. Package CI imports all 8 modules and exercises all 4 CLIs outside the checkout, including all 40 refits and full holdout.

The SVG is deterministic, checking all 640 centroid shades and16 shown score cells plus the entire report in compressed metadata. Gzip timestamp is 0 and descriptive OS header 255 is normalized for cross-Python portability; payload/CRC are unchanged. Metadata mutation and learned-center mutation tests prevent a plausible but stale figure. Browser inference uses separate scalar projection and distance code; no training labels, class assignment heuristic or held-out composition enters it.

The Chrome harness waits for both the port file and a live debugger page target. Discovery alone has a bounded retry for the startup race where the port file precedes the HTTP listener. Independent mocks test delayed discovery, the first ready target, timeout and browser exit. Actual inference assertions are never retried.

## Findings and limits

Raw++ fitting inertia 9682.7571 is slightly below raw uniform 9683.1241. Yet raw uniform test ARI .665149 is slightly above raw++ .662690, while raw++ NMI .751821 exceeds uniform .749688. PCA16 test ARI .669707 is the highest of these four; it is not a test-selected deployment or a general dimensionality result. Whiten16 ARI .665596 / NMI .739303 does not dominate. All numbers describe this fixed dataset/design; no causal component isolation, seed-population estimate or confidence interval is offered.

Independent exact rational enumeration of all 31 binary nonempty partitions of `[0,1,4,5,9,10]` gives global two-center inertia 17.5. Lloyd initialized at [0,1] stops at 26.5; [4,5] stops at 22.6666667. Thus stable local descent does not prove a global solution. K-means++ has a published expected approximation guarantee for the seeding distribution; this implementation does not turn that theorem into a per-run optimality certificate. There is no certified global optimum for the real-digit models.

## Sources and attribution

All six inspected November 8, 2026. Established methods, original implementation; no endorsement.

1. Arthur & Vassilvitskii (2007), **k-means++: The Advantages of Careful Seeding**, SODA. D² algorithm, expected rather than per-run guarantee: https://theory.stanford.edu/~sergei/papers/kMeansPP-soda
2. Lloyd (1982), **Least Squares Quantization in PCM**, IEEE Transactions on Information Theory. Nearest assignment/centroid conditions and local optimization: https://web.stanford.edu/class/ee398a/handouts/papers/Lloyd%20-%20Least%20Squares%20Q%20in%20PCM.pdf
3. scikit-learn **Clustering** documentation. ARI, MI normalization, all-pair silhouette definitions and geometry limits: https://scikit-learn.org/stable/modules/clustering.html
4. scikit-learn **Adjustment for chance in clustering performance evaluation** example. Unadjusted NMI's fragmentation bias: https://scikit-learn.org/stable/auto_examples/cluster/plot_adjusted_for_chance_measures.html
5. NumPy **numpy.linalg.svd** documentation. Reduced SVD conventions: https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
6. Alpaydin & Kaynak (1998), **Optical Recognition of Handwritten Digits**, UCI, CC BY 4.0, DOI 10.24432/C50P49. Official original split, block-count preprocessing, writer-group description and attribution: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits


## v0.2 restart stability

The existing 40 fits and all held-out numbers above are unchanged. [STABILITY.md](STABILITY.md) documents the new label-free same-data partition agreement and exact rounded-cost center matching audit. The separate browser page exposes every saved restart. Combined tests: 38 Python, 10 Node/Chrome; installed wheel: nine modules, five CLIs.

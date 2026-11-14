# ClusterLab

ClusterLab fits k-means to the UCI handwritten-digit data with manual NumPy code. It keeps all 40 restarts, their objective histories and the checks behind each selected run. Two browser inspectors let you edit pixels, compare restarts and inspect centroid-matching certificates. Labels are used only after fitting for the external audit, never to fit a model or choose a restart. There is no scikit-learn dependency or remote model API.

![Ten learned means, external clustering scores and a local-minimum counterexample](assets/cluster-audit.svg)

## Run it

```bash
python -m pip install -e .
python -m clusterlab demo
python -m unittest discover -s tests -v
node --test tests/*.cjs   # Node 22+ and Chrome; CHROME_BIN overrides the executable
bash examples/demo.sh
```

Try the [live clustering inspector](https://midya8.github.io/clusterlab/), or open `docs/index.html` locally. It recomputes all ten distances for editable 8x8 digits, with keyboard controls, four fixed model arms, a centroid atlas, test-label composition, every restart's descent curve and copy/download JSON witnesses. No server, uploads, analytics or remote model calls. Twenty original test rows are included by index, not selected for success.

## Which inspector to use

- **Original inspector:** edit digit pixels, inspect distances and compare the four fixed representation/initializer arms.
- **Restart inspector:** compare all 40 frozen models and their fitting partitions, with exact centroid-assignment witnesses for the saved rounded costs.
- **Matching ambiguity:** inspect optimal-permutation counts, best-distinct alternatives and a sufficient cost-perturbation bound.

Centroid-assignment optimality is not global k-means optimality. Same-data seed
sensitivity is not sampling stability, and a matching gap is not uncertainty
about a digit label or robustness to pixel edits.

## Centroid-matching ambiguity (0.3)

Every original centroid-matching witness now has ten exact selected-edge exclusion certificates. Their minimum is the cheapest **different permutation**, which may tie the optimum. An exact tight-graph DP counts optimal permutations; an independent full-cost DP checks all 220 counts and best-distinct gaps. All 2,200 exclusion witnesses have attaining permutations and exact dual bounds for rigorously penalized matrices. The browser validates 4,000 oriented exclusions across its 400 pair views.

All real saved matrices happen to have unique optima. The browser includes a two-optimum control and a misleading unused-tight-edge control, exact replacement witnesses, a smallest-gap picker, and a strict **sufficient** entrywise-cost perturbation bound. This is not statistical uncertainty, a sharp radius, input-pixel robustness, or global clustering optimality. Original models, matching matrices, selected seeds and held-out scores are unchanged. [AMBIGUITY.md](AMBIGUITY.md) gives the penalty/exclusion-cover proof and numerical boundary.

```bash
python -m clusterlab ambiguity --out /tmp/matching-ambiguity.json
python examples/export_ambiguity.py
```

## Forty frozen restarts

The [live restart inspector](https://midya8.github.io/clusterlab/stability.html) runs actual inference for **all 40 saved models**, not only the selected four. Compare any two seeds in one representation: all ten aligned centroid pairs, the complete 10x10 fitting-partition ARI matrix, arbitrary pixel edits and exact downloadable witnesses.

- **180 distinct pairs + 40 self-pairs**. Manual shortest-augmenting-path Hungarian matching returns a permutation and dual potentials. Each binary64 squared-distance cost becomes its exact power-of-two ratio. Python unbounded integers and browser BigInt verify dual feasibility, tight matched edges and equal primal/dual objectives. These are exact certificates for the **saved rounded matrices**, not real-number arithmetic or global k-means optimality.
- All 220 assignments agree with an independent subset-DP optimum; randomized tiny signed/huge-integer costs also agree with exhaustive permutations. Counterexamples show why raw cluster IDs and rowwise nearest-center matching are misleading.
- No refitting, changed models or new holdout-based selection. The original minimum-fitting-inertia winners and all previous holdout scores stay unchanged. Same-data initialization sensitivity is not sampling stability, uncertainty, useful-class discovery, or a way to select k. Center movement is geometry-specific and matching weights centers equally, not by population.

```bash
python -m clusterlab stability --out /tmp/restart-witnesses.json
python examples/export_stability.py
```

48 Python tests and 13 Node/real-Chrome tests cover the combined project. The restart inspector checks all 800 model/example pairs against native references, all 400 directed pair controls, 1,280 displayed centroid pixels, arbitrary edited inputs and copy/download at mobile and desktop widths. [STABILITY.md](STABILITY.md) gives the algorithm, precision boundary, measured ranges and sources.

## Fitting and checks

- D² k-means++ and uniform distinct-value initializers, explicit lowest-index assignment ties, Lloyd mean updates, empty-center repairs and full monotone objective histories.
- Ten seeds per arm; the selected run minimizes **fitting inertia**, not a label score. All 40 starts, center arrays, fitting assignments and initial row indices remain in the model artifact.
- Four prespecified arms: normalized 64-pixel k-means++ and uniform, fitting-only 16-component PCA, and 16-component whitened PCA. No test-based choice of k, dimension, normalization, initializer or seed.
- Independent ARI pair-count/null-permutation reference, entropy-based NMI reference, scalar all-pair silhouette reference, eigenspectrum/whitening checks and an exhaustive exact-rational local-optimum counterexample.
- A full refit test forbids test-loader access and supplies unusable training labels. Frozen file hashes are checked before held-out data load. All-row test evaluation is reproduced independently of saved scores.
- Python/JavaScript inference parity on all 80 public model/example pairs. Real Chrome exercises edits, keyboard, all 640 displayed centroid pixels, copy denial/success, download and expanded 320/390/1280 layouts.

## Results on the official test split

Original UCI 3,823 training / 1,797 test rows stay separate. Training uses **only training features**; digit labels enter after fitting for the external audit. Each arm chooses its lowest fitting inertia from seeds 100-109. All 40 runs reached stable assignments within the 200-update budget.

| Arm | Adjusted Rand | Arithmetic NMI | Silhouette in its own metric |
|---|---:|---:|---:|
| 64 pixels / k-means++ | 0.662690 | 0.751821 | 0.186434 |
| 64 pixels / uniform | 0.665149 | 0.749688 | 0.186142 |
| 16 PCA / k-means++ | 0.669707 | 0.740137 | 0.224268 |
| 16 whitened PCA / k-means++ | 0.665596 | 0.739303 | 0.149408 |

Raw k-means++ has slightly smaller fitting inertia than raw uniform (9682.7571 versus 9683.1241), but **uniform has slightly higher held-out ARI**. That is a descriptive result for these fixed starts, not a general seeding conclusion. ARI and NMI do not select the same arm. NMI is not chance-adjusted.

**PCA and whitening change the metric.** Cross-arm inertia/silhouette values are not directly comparable as if distances were unchanged. Silhouette uses all held-out Euclidean point-to-point distances in each arm's representation, with singleton silhouettes zero, not squared or centroid distances.

Cluster IDs are not digit predictions. We do not remap clusters with test labels and call that classification accuracy. A distance or runner-up gap is not a calibrated probability. This is unsupervised vector quantization, not OCR, semantic discovery or a production digit recognizer.

## Reproduce all 40 fits and the holdout audit

```bash
OPENBLAS_NUM_THREADS=1 python -m clusterlab train --out /tmp/clusterlab-refit
OPENBLAS_NUM_THREADS=1 python -m clusterlab evaluate --out /tmp/clusterlab-refit
python -m clusterlab predict examples/predict.json
```

`train` writes the fixed protocol before loading training features and never loads test rows. `evaluate` first checks the frozen protocol/model hashes in the saved evaluation plan, then opens the original test file. Changing these artifacts invalidates the plan. The hash guard detects mutation, not deliberate test-driven tuning by a person. There is no validation search: all dimensions, k, starts and geometries are fixed in advance.

The installed wheel includes data, artifacts and all ten modules. Its package CI runs all six CLIs outside the checkout, including a complete refit/evaluation. Rooted source examples export `docs/data.js` and the deterministic SVG; those publishing helpers are not installed console commands.

## Descent is not global optimality

On the tiny one-dimensional fixture `[0, 1, 4, 5, 9, 10]`, k=2 and initial centers `[0, 1]`, Lloyd converges monotonically to inertia **26.5**. Exhausting all 31 nonempty binary partitions with exact fractions gives **17.5**. The tests check both. We do not have a global-optimality certificate for the real-data fits. More restarts do not turn these finite observations into one.

## Data, credit and limits

Alpaydin, E. & Kaynak, C. (1998), [Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), UCI Machine Learning Repository, DOI 10.24432/C50P49. Original numeric files are unmodified, hashes checked at load, licensed CC BY 4.0. [Dataset license and changes](clusterlab/data/LICENSE.md). Normalization divides 8x8 block counts by 16; PCA and whitening are derived representations. UCI describes different writer groups in its original train/test split; individual writer IDs are not supplied. No exact feature duplicate crosses the split; near-duplicate/style dependence is not excluded.

This implements established algorithms, not a new clustering method. One dataset, fixed seeds, no confidence interval or general superiority claim. [METHOD.md](METHOD.md) gives equations, stopping and repair mechanics, geometry caveats, all six primary/documentation/data sources and the audit trail. No endorsement by cited authors or data providers. Original code MIT.

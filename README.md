# Segment Stability Bootstrap (CS05)

Are KMeans segments stable across resamples and across time, on real UK retailer transactions?

**Core question:** before a segment is used for decisions, does it survive resampling and a shift in the snapshot date?

## Setup

- Main snapshot: **2011-06-01**, the same as [CS01](../rfm-kmeans-segmentation/README.md). Customers: **4,933**.
- Early snapshot: **2010-09-01**, nine months earlier. Customers with history before it: **3,321**. All of them also appear at the main snapshot.
- Features: recency, frequency, monetary (capped at the 99th percentile per window). `log1p`, then `RobustScaler`.
- Model: `KMeans(k=3, n_init=10, random_state=42)`. Smallest segment: 22.7%.
- Dataset: Online Retail II (Kaggle `mashlyn/online-retail-ii-uci`, CC0). See [data/README.md](data/README.md).

## Pre-registered thresholds

Set in `core.py` before the bootstrap was run:

- **Stable** if the median bootstrap ARI ≥ **0.80** and the cross-window ARI ≥ **0.60**.
- Bootstrap: **50** resamples with replacement. Each resample is fitted with KMeans, then used to label all customers. ARI is measured against the main-snapshot labels.
- Cross-window: a model fitted on the early window is applied to the main window's features for shared customers. Its labels are compared with a model fitted directly on the main window. The scaler is fitted on the early window in both cases.

## Results

| Check | Result | Threshold | Outcome |
|---|---|---|---|
| Bootstrap median ARI (n = 50) | **0.944** | ≥ 0.80 | pass |
| Bootstrap mean ARI | 0.927 | | |
| Bootstrap 5th percentile ARI | 0.827 | | |
| Bootstrap minimum ARI | 0.799 | | |
| Cross-window ARI (3,321 shared customers) | **0.804** | ≥ 0.60 | pass |

**Verdict: stable under the pre-registered thresholds.**

Centroid drift between windows, matched one-to-one with the Hungarian method (distance in scaled feature space):

| Early centroid | Matched main centroid | Drift |
|---|---|---|
| 0 | 1 | 0.351 |
| 1 | 0 | 0.433 |
| 2 | 2 | 0.567 |

All three centroids moved, and the segment with the largest drift is the one whose centroid moved most. The sizes at the main snapshot are 2,091 / 1,122 / 1,720 (42%, 23%, 35%).

Figures:
- [`reports/figures/cs05_bootstrap_ari.png`](reports/figures/cs05_bootstrap_ari.png): bootstrap ARI distribution with the pre-set threshold
- [`reports/figures/cs05_centroid_drift.png`](reports/figures/cs05_centroid_drift.png): drift per matched centroid

## Limitations

- **The bootstrap measures sampling stability on one snapshot.** It does not measure temporal stability. The cross-window check covers time, but only across nine months and on a shared population. Customers who appear in both windows are the ones who were active earlier, so the comparison is not a fresh draw.
- **The early window has only nine months of history.** Features built on less history may be less precise, so a lower ARI could reflect less data as well as real change.
- **The thresholds are conventional choices, not calibrated values.** They were fixed before the run, but a different choice (for example, 0.70 or 0.90) would change the verdict.

## Reproduce

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest -q
cd notebooks && jupyter nbconvert --to notebook --execute --inplace segment_stability_bootstrap.ipynb
```

The notebook reads from `RETAIL_DATA_PATH` (default `~/.local/share/clv-data/online-retail-ii/online_retail_II.csv`).

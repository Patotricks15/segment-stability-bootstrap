import numpy as np
import pandas as pd
import pytest

from stabseg.core import (
    centroid_drift,
    clean_transactions,
    cross_window_ari,
    bootstrap_ari,
    rfm_features,
    stability_verdict,
)


def _blobs(seed=0, n=60):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [10.0, 10.0], [-10.0, 10.0]])
    X = np.vstack([c + rng.normal(scale=0.3, size=(n, 2)) for c in centers])
    labels = np.repeat([0, 1, 2], n)
    return X, labels


def test_clean_drops_cancellations_and_missing_customers():
    df = pd.DataFrame(
        [
            ["1", "A", 1, "2011-01-01", 10.0, 1.0],
            ["C2", "A", 1, "2011-01-02", 10.0, 1.0],
            ["3", "A", 1, "2011-01-03", 10.0, None],
        ],
        columns=["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID"],
    )
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    out = clean_transactions(df.assign(Description="x", Country="x"))
    assert list(out["Invoice"]) == ["1"]


def test_rfm_excludes_transactions_on_or_after_snapshot():
    tx = pd.DataFrame(
        {
            "Invoice": ["1", "2", "3"],
            "customer_id": [7, 7, 7],
            "InvoiceDate": pd.to_datetime(["2011-01-01", "2011-05-01", "2011-07-01"]),
            "revenue": [10.0, 20.0, 999.0],
        }
    )
    feats = rfm_features(tx, pd.Timestamp("2011-06-01"))
    assert feats.loc[7, "frequency"] == 2
    assert feats.loc[7, "monetary"] == 30.0


def test_bootstrap_ari_is_high_for_well_separated_clusters():
    X, labels = _blobs()
    scores = bootstrap_ari(X, labels, k=3, n_boot=5, seed=1)
    assert scores.shape == (5,)
    assert scores.min() > 0.95


def test_cross_window_ari_is_one_for_identical_windows():
    rng = np.random.default_rng(0)
    feats = pd.DataFrame(
        rng.lognormal(size=(120, 3)),
        columns=["recency_days", "frequency", "monetary"],
        index=range(120),
    )
    ari, labels = cross_window_ari(feats, feats.copy(), k=3)
    assert ari == pytest.approx(1.0)
    assert len(labels) == 120


def test_centroid_drift_zero_under_label_permutation():
    a = np.array([[0.0, 0.0], [5.0, 5.0], [-5.0, 5.0]])
    b = a[[2, 0, 1]]
    dist, _ = centroid_drift(a, b)
    assert np.allclose(dist, 0.0)


def test_centroid_drift_reports_shift():
    a = np.array([[0.0, 0.0], [5.0, 5.0]])
    b = np.array([[1.0, 0.0], [5.0, 6.0]])
    dist, _ = centroid_drift(a, b)
    assert dist.mean() == pytest.approx(1.0)


def test_stability_verdict_uses_both_thresholds():
    assert stability_verdict(0.9, 0.7) == "stable"
    assert stability_verdict(0.9, 0.5) == "not stable"
    assert stability_verdict(0.7, 0.9) == "not stable"

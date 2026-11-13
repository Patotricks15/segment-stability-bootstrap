"""Bootstrap, cross-window, and centroid-drift stability checks for KMeans segments on Online Retail II (CS05)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42
K = 3
SNAPSHOT_MAIN = pd.Timestamp("2011-06-01")
SNAPSHOT_EARLY = pd.Timestamp("2010-09-01")
N_BOOT = 50

# Pre-registered thresholds, fixed before the bootstrap was run.
STABLE_BOOTSTRAP_MEDIAN_ARI = 0.80
STABLE_CROSS_WINDOW_ARI = 0.60

DEFAULT_DATA_PATH = Path.home() / ".local/share/clv-data/online-retail-ii/online_retail_II.csv"


def data_path() -> Path:
    return Path(os.environ.get("RETAIL_DATA_PATH", DEFAULT_DATA_PATH))


def load_retail(path: Path | None = None) -> pd.DataFrame:
    raw = pd.read_csv(
        path or data_path(),
        encoding="latin1",
        parse_dates=["InvoiceDate"],
        dtype={"Invoice": str, "StockCode": str},
    )
    return clean_transactions(raw)


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns={"Customer ID": "customer_id"})
    df = df.dropna(subset=["customer_id"]).copy()
    df["customer_id"] = df["customer_id"].astype("int64")
    df = df[~df["Invoice"].str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)].copy()
    df["revenue"] = df["Quantity"] * df["Price"]
    return df[["Invoice", "customer_id", "InvoiceDate", "revenue"]]


def rfm_features(tx: pd.DataFrame, snapshot: pd.Timestamp, monetary_cap_quantile: float = 0.99) -> pd.DataFrame:
    """RFM features from transactions strictly before `snapshot`."""
    past = tx[tx["InvoiceDate"] < snapshot]
    grouped = past.groupby("customer_id")
    feats = pd.DataFrame(
        {
            "recency_days": (snapshot - grouped["InvoiceDate"].max()).dt.days,
            "frequency": grouped["Invoice"].nunique(),
            "monetary": grouped["revenue"].sum(),
        }
    )
    cap = feats["monetary"].quantile(monetary_cap_quantile)
    feats["monetary"] = feats["monetary"].clip(upper=cap)
    return feats


def fit_scaler(feats: pd.DataFrame) -> RobustScaler:
    return RobustScaler().fit(np.log1p(feats.clip(lower=0)))


def transform(feats: pd.DataFrame, scaler: RobustScaler) -> np.ndarray:
    return scaler.transform(np.log1p(feats.clip(lower=0)))


def fit_kmeans(X: np.ndarray, k: int = K) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)


def bootstrap_ari(X: np.ndarray, reference_labels: np.ndarray, k: int = K, n_boot: int = N_BOOT, seed: int = RANDOM_STATE) -> np.ndarray:
    """ARI between reference labels and labels from a KMeans fit on each bootstrap resample.

    Each resample draws rows with replacement, fits KMeans on the resample, and labels the full
    original set with the fitted model. The ARI is invariant to label permutation.
    """
    rng = np.random.default_rng(seed)
    scores = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(X), size=len(X))
        model = KMeans(n_clusters=k, n_init=5, random_state=RANDOM_STATE + b).fit(X[idx])
        scores[b] = adjusted_rand_score(reference_labels, model.predict(X))
    return scores


def cross_window_ari(feats_a: pd.DataFrame, feats_b: pd.DataFrame, k: int = K) -> tuple[float, pd.DataFrame]:
    """Compare a model fitted on window A with a model fitted on window B, on shared customers.

    The scaler is fitted on window A so both models share one coordinate system.
    Returns the ARI and the shared customers with both label sets.
    """
    shared = feats_a.index.intersection(feats_b.index)
    scaler = fit_scaler(feats_a)
    X_a = transform(feats_a.loc[shared], scaler)
    X_b = transform(feats_b.loc[shared], scaler)
    labels_from_a = fit_kmeans(transform(feats_a, scaler), k).predict(X_b)
    model_b = fit_kmeans(X_b, k)
    labels_from_b = model_b.labels_
    labels = pd.DataFrame({"label_from_a": labels_from_a, "label_from_b": labels_from_b}, index=shared)
    return adjusted_rand_score(labels_from_a, labels_from_b), labels


def centroid_drift(centroids_a: np.ndarray, centroids_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match centroids one-to-one with the Hungarian method and return the matched distances."""
    cost = np.linalg.norm(centroids_a[:, None, :] - centroids_b[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    return cost[rows, cols], cols


def stability_verdict(bootstrap_median: float, cross_window: float) -> str:
    if bootstrap_median >= STABLE_BOOTSTRAP_MEDIAN_ARI and cross_window >= STABLE_CROSS_WINDOW_ARI:
        return "stable"
    return "not stable"

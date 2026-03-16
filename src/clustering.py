"""
clustering.py
=============
KMeans clustering pipeline for the London accessibility analysis.

Includes:
* feature scaling
* k-range evaluation (elbow + silhouette)
* final model fitting
* cluster ranking (most → least accessible)
* per-tract underserved scoring and gap identification
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Features where a *higher* value means *worse* accessibility.
DISTANCE_FEATURES = [
    "dist_nearest_hospital",
    "dist_nearest_school",
    "dist_nearest_grocery",
]

# Features where a *lower* value means *worse* accessibility.
SUPPLY_FEATURES = [
    "num_bus_stops",
    "num_parks",
    "transit_coverage",
]

CLUSTER_FEATURES = DISTANCE_FEATURES + SUPPLY_FEATURES

# Human-readable labels for gap reporting
_GAP_LABELS = {
    "dist_nearest_hospital": "hospital access",
    "dist_nearest_school": "school access",
    "dist_nearest_grocery": "grocery access",
    "num_bus_stops": "bus stop access",
    "num_parks": "park access",
    "transit_coverage": "transit coverage",
}


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def prepare_feature_matrix(
    gdf: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    """Extract, impute, and scale clustering features.

    Returns
    -------
    X_scaled : ndarray
    scaler   : fitted StandardScaler
    X_df     : DataFrame of raw (imputed) features
    """
    if feature_cols is None:
        feature_cols = CLUSTER_FEATURES

    X_df = gdf[feature_cols].copy()

    # Replace missing values with 0 (a tract with no data = no service)
    X_df = X_df.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    return X_scaled, scaler, X_df


# ---------------------------------------------------------------------------
# k-range evaluation
# ---------------------------------------------------------------------------

def evaluate_kmeans_range(
    X_scaled: np.ndarray,
    k_range: range | list[int] | None = None,
    random_state: int = 42,
) -> tuple[list[int], list[float], list[float]]:
    """Run KMeans for every k in *k_range*.

    Returns (ks, inertias, silhouettes).
    """
    if k_range is None:
        k_range = range(2, 9)

    ks: list[int] = []
    inertias: list[float] = []
    silhouettes: list[float] = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        ks.append(k)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels)
        silhouettes.append(sil)
        print(f"  k={k}  inertia={km.inertia_:,.0f}  silhouette={sil:.4f}")

    return ks, inertias, silhouettes


# ---------------------------------------------------------------------------
# Optimal k selection
# ---------------------------------------------------------------------------

def choose_optimal_k(
    silhouettes: list[float],
    ks: list[int],
    manual_k: Optional[int] = None,
) -> int:
    """Return the k with the highest silhouette score, or *manual_k* if provided."""
    if manual_k is not None:
        print(f"  Using manual override k = {manual_k}")
        return manual_k
    best_idx = int(np.argmax(silhouettes))
    best_k = ks[best_idx]
    print(f"  Best k by silhouette = {best_k} (score {silhouettes[best_idx]:.4f})")
    return best_k


# ---------------------------------------------------------------------------
# Final model
# ---------------------------------------------------------------------------

def fit_final_kmeans(
    X_scaled: np.ndarray,
    k: int,
    random_state: int = 42,
) -> tuple[KMeans, np.ndarray]:
    """Fit the final KMeans model and return (model, labels)."""
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


# ---------------------------------------------------------------------------
# Cluster ranking
# ---------------------------------------------------------------------------

def rank_clusters_by_accessibility(
    gdf: pd.DataFrame,
    feature_cols: list[str] | None = None,
    cluster_col: str = "cluster_raw",
) -> dict[int, int]:
    """Rank clusters from most accessible (0) to least accessible (k-1).

    Builds a composite *disadvantage score* per cluster:
    * Distance features: higher standardised mean → more disadvantaged (+)
    * Supply features:   lower standardised mean → more disadvantaged (−)

    Returns a mapping {old_label: new_ranked_label}.
    """
    if feature_cols is None:
        feature_cols = CLUSTER_FEATURES

    cluster_means = gdf.groupby(cluster_col)[feature_cols].mean()

    # Standardise the cluster means for comparability
    scaler = StandardScaler()
    std_means = pd.DataFrame(
        scaler.fit_transform(cluster_means),
        index=cluster_means.index,
        columns=cluster_means.columns,
    )

    # Disadvantage: +distance, –supply
    disadv = pd.Series(0.0, index=std_means.index)
    for col in feature_cols:
        if col in DISTANCE_FEATURES:
            disadv += std_means[col]
        else:
            disadv -= std_means[col]

    ranked_order = disadv.sort_values().index.tolist()  # least disadvantaged first
    label_map = {old: new for new, old in enumerate(ranked_order)}
    return label_map


def apply_cluster_ranking(
    gdf: pd.DataFrame,
    label_map: dict[int, int],
    raw_col: str = "cluster_raw",
    ranked_col: str = "cluster",
) -> pd.DataFrame:
    """Add a *ranked_col* column by mapping raw labels through *label_map*."""
    gdf[ranked_col] = gdf[raw_col].map(label_map)
    return gdf


# ---------------------------------------------------------------------------
# Underserved scoring
# ---------------------------------------------------------------------------

def compute_underserved_score(
    gdf: pd.DataFrame,
    feature_cols: list[str] | None = None,
    scaler: StandardScaler | None = None,
    X_scaled: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute a continuous *underserved_score* and *biggest_accessibility_gap*.

    The score is the sum of standardised deficits:
    * distance features contribute positively (farther = worse)
    * supply features contribute negatively (fewer = worse, so we flip sign)

    Parameters
    ----------
    gdf : DataFrame with the raw feature columns.
    feature_cols : list of feature column names.
    scaler : a fitted StandardScaler (from prepare_feature_matrix).
    X_scaled : pre-computed scaled matrix; if None, re-scales from gdf.

    Returns gdf with two new columns: ``underserved_score`` and
    ``biggest_accessibility_gap``.
    """
    if feature_cols is None:
        feature_cols = CLUSTER_FEATURES

    if X_scaled is None:
        if scaler is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(gdf[feature_cols].fillna(0))
        else:
            X_scaled = scaler.transform(gdf[feature_cols].fillna(0))

    std_df = pd.DataFrame(X_scaled, index=gdf.index, columns=feature_cols)

    # Build deficit columns: positive = more underserved
    deficit = pd.DataFrame(index=gdf.index)
    for col in feature_cols:
        if col in DISTANCE_FEATURES:
            deficit[col] = std_df[col]          # higher distance = worse
        else:
            deficit[col] = -std_df[col]         # lower supply = worse (flip)

    gdf["underserved_score"] = deficit.sum(axis=1)

    # Biggest gap = the feature with the largest single deficit
    gap_labels = {col: _GAP_LABELS.get(col, col) for col in feature_cols}
    gdf["biggest_accessibility_gap"] = deficit.idxmax(axis=1).map(gap_labels)

    return gdf


# ---------------------------------------------------------------------------
# Cluster summary
# ---------------------------------------------------------------------------

def summarize_clusters(
    gdf: pd.DataFrame,
    feature_cols: list[str] | None = None,
    cluster_col: str = "cluster",
) -> pd.DataFrame:
    """Produce a summary table of mean feature values per ranked cluster."""
    if feature_cols is None:
        feature_cols = CLUSTER_FEATURES

    summary = gdf.groupby(cluster_col)[feature_cols].agg(["mean", "median", "count"])
    # Flatten MultiIndex columns
    summary.columns = ["_".join(c) for c in summary.columns]
    # Also add a simple count column
    summary["n_tracts"] = gdf.groupby(cluster_col).size()
    return summary.sort_index()

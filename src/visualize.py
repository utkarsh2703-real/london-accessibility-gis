"""
visualize.py
============
Plotting and interactive-map utilities for the London accessibility
analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Elbow / silhouette line charts
# ---------------------------------------------------------------------------

def plot_elbow_curve(
    ks: list[int],
    inertias: list[float],
    output_dir: Optional[Path] = None,
) -> plt.Figure:
    """Plot inertia vs. k (elbow method)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, inertias, "o-", linewidth=2)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method — KMeans Inertia")
    ax.set_xticks(ks)
    fig.tight_layout()
    if output_dir:
        fig.savefig(output_dir / "elbow_curve.png", dpi=150)
    return fig


def plot_silhouette_scores(
    ks: list[int],
    silhouettes: list[float],
    output_dir: Optional[Path] = None,
) -> plt.Figure:
    """Plot silhouette score vs. k."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, silhouettes, "s-", linewidth=2, color="darkorange")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score by k")
    ax.set_xticks(ks)
    fig.tight_layout()
    if output_dir:
        fig.savefig(output_dir / "silhouette_scores.png", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Cluster heatmap
# ---------------------------------------------------------------------------

def plot_cluster_heatmap(
    gdf: pd.DataFrame,
    feature_cols: list[str],
    cluster_col: str = "cluster",
    output_dir: Optional[Path] = None,
) -> plt.Figure:
    """Seaborn heatmap of mean feature values per ranked cluster.

    Shows standardised values so that features on different scales
    are visually comparable.
    """
    means = gdf.groupby(cluster_col)[feature_cols].mean()

    # Standardise column-wise for display
    display = (means - means.mean()) / means.std().replace(0, 1)

    fig, ax = plt.subplots(figsize=(10, max(4, len(means) * 0.8 + 1)))
    sns.heatmap(
        display,
        annot=means.round(1).values,
        fmt="",
        cmap="RdYlGn_r",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Cluster Feature Means (annotated: original units; colour: standardised)")
    ax.set_ylabel("Cluster")
    ax.set_xlabel("")
    fig.tight_layout()
    if output_dir:
        fig.savefig(output_dir / "cluster_heatmap.png", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Top underserved bar chart
# ---------------------------------------------------------------------------

def plot_top_underserved(
    gdf: pd.DataFrame,
    n: int = 10,
    tract_id_col: str = "CTUID",
    output_dir: Optional[Path] = None,
) -> plt.Figure:
    """Horizontal bar chart of the top-*n* most underserved tracts."""
    top = gdf.nlargest(n, "underserved_score")

    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.45)))
    bars = ax.barh(
        top[tract_id_col].astype(str),
        top["underserved_score"],
        color="tomato",
        edgecolor="darkred",
    )
    ax.set_xlabel("Underserved Score")
    ax.set_title(f"Top {n} Most Underserved Census Tracts")
    ax.invert_yaxis()
    fig.tight_layout()
    if output_dir:
        fig.savefig(output_dir / "top_underserved.png", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Folium interactive map
# ---------------------------------------------------------------------------

_CLUSTER_PALETTE = [
    "#2ecc71",  # best  – green
    "#27ae60",
    "#f1c40f",
    "#e67e22",
    "#e74c3c",
    "#c0392b",
    "#8e44ad",
    "#2c3e50",  # worst – dark
]


def _style_function(feature: dict, palette: list[str], k: int) -> dict:
    """Return a Folium style dict based on the cluster label."""
    cluster = feature["properties"].get("cluster", 0)
    color = palette[int(cluster) % len(palette)]
    return {
        "fillColor": color,
        "color": "#333",
        "weight": 1,
        "fillOpacity": 0.6,
    }


def create_folium_cluster_map(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    tract_id_col: str = "CTUID",
    cluster_col: str = "cluster",
) -> folium.Map:
    """Build and save an interactive Folium choropleth of clustered tracts.

    Popups include all key accessibility fields.
    """
    gdf_wgs = gdf.to_crs("EPSG:4326")

    center_lat = gdf_wgs.geometry.centroid.y.mean()
    center_lon = gdf_wgs.geometry.centroid.x.mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron")

    k = int(gdf_wgs[cluster_col].nunique())

    popup_fields = [
        tract_id_col,
        cluster_col,
        "underserved_score",
        "dist_nearest_hospital",
        "dist_nearest_school",
        "dist_nearest_grocery",
        "num_bus_stops",
        "num_parks",
        "transit_coverage",
        "biggest_accessibility_gap",
    ]

    # Round numeric popup values for readability
    popup_gdf = gdf_wgs.copy()
    for col in popup_fields:
        if col in popup_gdf.columns and pd.api.types.is_numeric_dtype(popup_gdf[col]):
            popup_gdf[col] = popup_gdf[col].round(2)

    geojson_str = popup_gdf[popup_fields + ["geometry"]].to_json()

    folium.GeoJson(
        geojson_str,
        name="Clusters",
        style_function=lambda feat: _style_function(feat, _CLUSTER_PALETTE, k),
        tooltip=folium.GeoJsonTooltip(
            fields=popup_fields,
            aliases=[
                "Tract ID",
                "Cluster",
                "Underserved Score",
                "Dist Hospital (m)",
                "Dist School (m)",
                "Dist Grocery (m)",
                "Bus Stops",
                "Parks",
                "Transit Coverage (m)",
                "Biggest Gap",
            ],
        ),
        popup=folium.GeoJsonPopup(
            fields=popup_fields,
            aliases=[
                "Tract ID",
                "Cluster",
                "Underserved Score",
                "Dist Hospital (m)",
                "Dist School (m)",
                "Dist Grocery (m)",
                "Bus Stops",
                "Parks",
                "Transit Coverage (m)",
                "Biggest Gap",
            ],
        ),
    ).add_to(m)

    # Add a simple colour legend
    legend_html = _build_legend_html(k)
    m.get_root().html.add_child(folium.Element(legend_html))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"  Map saved → {output_path}")
    return m


def _build_legend_html(k: int) -> str:
    """Return an HTML legend string for *k* clusters."""
    items = ""
    for i in range(k):
        color = _CLUSTER_PALETTE[i % len(_CLUSTER_PALETTE)]
        label = "Most accessible" if i == 0 else ("Least accessible" if i == k - 1 else f"Cluster {i}")
        items += (
            f'<li><span style="background:{color};width:14px;height:14px;'
            f'display:inline-block;margin-right:6px;border:1px solid #555;"></span>'
            f'{label}</li>'
        )
    return (
        '<div style="position:fixed;bottom:30px;left:30px;z-index:1000;'
        'background:white;padding:10px 14px;border-radius:6px;'
        'box-shadow:0 2px 6px rgba(0,0,0,0.3);font-size:13px;">'
        '<b>Accessibility Clusters</b>'
        f'<ul style="list-style:none;padding:4px 0;margin:0;">{items}</ul>'
        '</div>'
    )

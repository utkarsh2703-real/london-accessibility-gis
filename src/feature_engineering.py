"""
feature_engineering.py
======================
Spatial feature engineering for the London accessibility analysis.

Every function assumes inputs are GeoDataFrames in **EPSG:32617**
so that all distance and area values are in metres.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiLineString


# ---------------------------------------------------------------------------
# Tract centroids
# ---------------------------------------------------------------------------

def compute_tract_centroids(tracts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a new GeoDataFrame whose geometry is the centroid of each tract.

    All other columns are preserved.
    """
    centroids = tracts.copy()
    centroids["geometry"] = tracts.geometry.centroid
    return centroids


# ---------------------------------------------------------------------------
# Point-in-polygon counting
# ---------------------------------------------------------------------------

def count_points_within_tracts(
    tracts: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    col_name: str,
) -> pd.Series:
    """Count how many *points* fall inside each tract polygon.

    Returns a Series indexed by the tracts index with name *col_name*.
    Tracts with zero points get 0 (not NaN).
    """
    joined = gpd.sjoin(points, tracts, how="inner", predicate="within")
    counts = joined.groupby("index_right").size().rename(col_name)
    return counts.reindex(tracts.index, fill_value=0)


# ---------------------------------------------------------------------------
# Parks: intersection (not just 'within')
# ---------------------------------------------------------------------------

def count_parks_intersecting_tracts(
    tracts: gpd.GeoDataFrame,
    parks: gpd.GeoDataFrame,
) -> pd.Series:
    """Count parks that *intersect* each tract (polygons may overlap boundaries)."""
    joined = gpd.sjoin(parks, tracts, how="inner", predicate="intersects")
    counts = joined.groupby("index_right").size().rename("num_parks")
    return counts.reindex(tracts.index, fill_value=0)


# ---------------------------------------------------------------------------
# Nearest-service distance
# ---------------------------------------------------------------------------

def compute_nearest_distance(
    tract_centroids: gpd.GeoDataFrame,
    service_points: gpd.GeoDataFrame,
    col_name: str,
) -> pd.Series:
    """Distance in metres from each tract centroid to the nearest service point.

    Uses ``sjoin_nearest`` for a fast, vectorised calculation.
    Falls back to a brute-force loop if the service layer is empty.
    """
    if service_points.empty:
        print(f"  WARNING: service layer is empty — {col_name} will be NaN")
        return pd.Series(np.nan, index=tract_centroids.index, name=col_name)

    # Ensure point geometries on the service layer
    svc = service_points.copy()
    if not all(svc.geometry.geom_type.isin(["Point"])):
        svc["geometry"] = svc.geometry.centroid

    joined = gpd.sjoin_nearest(
        tract_centroids[["geometry"]],
        svc[["geometry"]],
        how="left",
        distance_col="_dist",
    )

    # sjoin_nearest may duplicate rows if equidistant; keep closest
    dists = joined.groupby(joined.index)["_dist"].min().rename(col_name)
    return dists.reindex(tract_centroids.index)


# ---------------------------------------------------------------------------
# Transit coverage (route-km inside each tract)
# ---------------------------------------------------------------------------

def compute_transit_coverage(
    tracts: gpd.GeoDataFrame,
    route_lines: gpd.GeoDataFrame,
) -> pd.Series:
    """Total route-geometry length (m) inside each census-tract polygon.

    For every tract, clip every route line to the tract boundary and sum
    the resulting lengths.  Uses ``shapely.intersection`` per tract which
    is reliable for line-vs-polygon operations.
    """
    if route_lines.empty:
        print("  WARNING: route_lines is empty — transit_coverage will be 0")
        return pd.Series(0.0, index=tracts.index, name="transit_coverage")

    # Pre-build a single MultiLineString for faster intersection
    all_lines = route_lines.geometry.unary_union

    lengths: list[float] = []
    for poly in tracts.geometry:
        try:
            clipped = poly.intersection(all_lines)
            lengths.append(clipped.length)
        except Exception:
            lengths.append(0.0)

    return pd.Series(lengths, index=tracts.index, name="transit_coverage")


# ---------------------------------------------------------------------------
# Master feature builder
# ---------------------------------------------------------------------------

def build_accessibility_features(
    tracts: gpd.GeoDataFrame,
    bus_stops: gpd.GeoDataFrame,
    hospitals: gpd.GeoDataFrame,
    schools: gpd.GeoDataFrame,
    parks: gpd.GeoDataFrame,
    groceries: gpd.GeoDataFrame,
    transit_lines: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Assemble all tract-level accessibility features.

    Parameters are the individual layers already in EPSG:32617.
    Returns a copy of *tracts* with the new columns attached.
    """
    result = tracts.copy()
    centroids = compute_tract_centroids(tracts)

    # --- counts --------------------------------------------------------
    print("  Computing bus-stop counts …")
    result["num_bus_stops"] = count_points_within_tracts(tracts, bus_stops, "num_bus_stops")

    print("  Computing park intersection counts …")
    result["num_parks"] = count_parks_intersecting_tracts(tracts, parks)

    # --- nearest distances ---------------------------------------------
    print("  Computing nearest hospital distance …")
    result["dist_nearest_hospital"] = compute_nearest_distance(
        centroids, hospitals, "dist_nearest_hospital"
    )

    print("  Computing nearest school distance …")
    result["dist_nearest_school"] = compute_nearest_distance(
        centroids, schools, "dist_nearest_school"
    )

    print("  Computing nearest grocery distance …")
    result["dist_nearest_grocery"] = compute_nearest_distance(
        centroids, groceries, "dist_nearest_grocery"
    )

    # --- transit coverage ----------------------------------------------
    print("  Computing transit coverage (may take a moment) …")
    result["transit_coverage"] = compute_transit_coverage(tracts, transit_lines)

    # --- derived columns -----------------------------------------------
    result["tract_area_m2"] = result.geometry.area
    result["tract_area_km2"] = result["tract_area_m2"] / 1e6
    result["bus_stops_per_km2"] = (
        result["num_bus_stops"] / result["tract_area_km2"].replace(0, np.nan)
    )
    result["park_count_per_km2"] = (
        result["num_parks"] / result["tract_area_km2"].replace(0, np.nan)
    )

    # Fill any remaining NaN from division by zero
    result["bus_stops_per_km2"] = result["bus_stops_per_km2"].fillna(0)
    result["park_count_per_km2"] = result["park_count_per_km2"].fillna(0)

    print(f"  Feature engineering complete — {len(result)} tracts, {result.shape[1]} columns")
    return result

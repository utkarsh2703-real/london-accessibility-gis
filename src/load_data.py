"""
load_data.py
============
Functions to load, clean, and reproject every spatial data source
for the London, Ontario accessibility analysis.

All loaders return GeoDataFrames in **EPSG:32617** (UTM Zone 17N)
so that downstream distance / area calculations are in metres.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_CRS = "EPSG:32617"
WGS84 = "EPSG:4326"
LONDON_CTUID_PREFIX = "555"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_output_dirs(project_root: Path) -> Path:
    """Create the output/ directory if it does not exist and return its path."""
    out = project_root / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _find_shapefile(directory: Path) -> Path:
    """Recursively search *directory* for the first .shp file."""
    shp_files = list(directory.rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(
            f"No .shp file found under {directory}. "
            "Check that the census archive was extracted correctly."
        )
    print(f"  Found shapefile: {shp_files[0]}")
    return shp_files[0]


def _extract_zip_if_needed(zip_path: Path, extract_to: Path) -> Path:
    """Extract a zip archive if *extract_to* does not already exist.

    Returns the directory that contains the extracted contents.
    """
    if extract_to.exists() and any(extract_to.iterdir()):
        print(f"  Already extracted: {extract_to}")
        return extract_to
    print(f"  Extracting {zip_path} → {extract_to} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    return extract_to


def _find_gtfs_dir(data_dir: Path) -> Path:
    """Locate the GTFS directory or zip inside *data_dir*.

    Handles both an already-extracted folder (``google_transit/``) and
    a ``google_transit.zip`` archive.
    """
    folder = data_dir / "google_transit"
    if folder.is_dir() and (folder / "shapes.txt").exists():
        return folder

    zip_path = data_dir / "google_transit.zip"
    if zip_path.exists():
        dest = data_dir / "google_transit"
        _extract_zip_if_needed(zip_path, dest)
        return dest

    raise FileNotFoundError(
        "Could not find google_transit/ directory or google_transit.zip "
        f"in {data_dir}"
    )


# ---------------------------------------------------------------------------
# Census tracts
# ---------------------------------------------------------------------------

def load_london_census_tracts(data_dir: Path) -> gpd.GeoDataFrame:
    """Load the national census-tract shapefile and filter to London CMA.

    Parameters
    ----------
    data_dir : Path
        Directory that contains **either** the extracted census folder
        (``DLI_2021_Census_DBF_Eng_Nat_ct/``) **or** the zip archive
        (``DLI_2021_Census_DBF_Eng_Nat_ct.zip``).

    Returns
    -------
    GeoDataFrame in EPSG:32617 with only London census tracts
    (CTUID starting with '555').
    """
    ct_dir = data_dir / "DLI_2021_Census_DBF_Eng_Nat_ct"
    ct_zip = data_dir / "DLI_2021_Census_DBF_Eng_Nat_ct.zip"

    if ct_dir.is_dir():
        shp_path = _find_shapefile(ct_dir)
    elif ct_zip.exists():
        dest = _extract_zip_if_needed(ct_zip, ct_dir)
        shp_path = _find_shapefile(dest)
    else:
        raise FileNotFoundError(
            f"Census tract data not found in {data_dir}. "
            "Expected DLI_2021_Census_DBF_Eng_Nat_ct/ or .zip"
        )

    print("  Loading national census tracts …")
    gdf = gpd.read_file(shp_path)
    print(f"  Loaded {len(gdf)} national tracts  |  CRS = {gdf.crs}")

    if "CTUID" not in gdf.columns:
        raise KeyError(
            "Expected column 'CTUID' not found. "
            f"Available columns: {list(gdf.columns)}"
        )

    london = gdf[gdf["CTUID"].astype(str).str.startswith(LONDON_CTUID_PREFIX)].copy()
    print(f"  Filtered to {len(london)} London CMA tracts (prefix '{LONDON_CTUID_PREFIX}')")

    if london.empty:
        raise ValueError("No London tracts found — check CTUID prefix filter.")

    london = london.to_crs(TARGET_CRS)
    return london


# ---------------------------------------------------------------------------
# Bus stops
# ---------------------------------------------------------------------------

def load_bus_stops(data_dir: Path) -> gpd.GeoDataFrame:
    """Load the London Transit bus-stop inventory CSV.

    Expects columns ``Latitude`` and ``Longitude``.
    """
    csv_path = data_dir / "Open-Data-BSI-Jan-2025.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Bus stop file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} bus-stop rows from CSV")

    for col in ("Latitude", "Longitude"):
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' in bus stop CSV. Got: {list(df.columns)}")

    df = df.dropna(subset=["Latitude", "Longitude"]).copy()
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"])
    print(f"  {len(df)} stops with valid coordinates")

    geometry = [Point(lon, lat) for lon, lat in zip(df["Longitude"], df["Latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=WGS84)
    gdf = gdf.to_crs(TARGET_CRS)
    return gdf


# ---------------------------------------------------------------------------
# GeoJSON loaders (schools, hospitals, parks)
# ---------------------------------------------------------------------------

def _load_geojson(data_dir: Path, filename: str, label: str) -> gpd.GeoDataFrame:
    """Generic GeoJSON loader with CRS validation."""
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    gdf = gpd.read_file(path)
    print(f"  Loaded {len(gdf)} {label} features  |  CRS = {gdf.crs}")
    gdf = gdf.to_crs(TARGET_CRS)
    return gdf


def load_schools(data_dir: Path) -> gpd.GeoDataFrame:
    """Load Schools.geojson → EPSG:32617."""
    return _load_geojson(data_dir, "Schools.geojson", "school")


def load_hospitals(data_dir: Path) -> gpd.GeoDataFrame:
    """Load Hospitals.geojson → EPSG:32617."""
    return _load_geojson(data_dir, "Hospitals.geojson", "hospital")


def load_parks(data_dir: Path) -> gpd.GeoDataFrame:
    """Load Parks.geojson → EPSG:32617."""
    return _load_geojson(data_dir, "Parks.geojson", "park")


# ---------------------------------------------------------------------------
# Grocery stores via OpenStreetMap
# ---------------------------------------------------------------------------

def fetch_grocery_stores_osm(
    place: str = "London, Ontario, Canada",
) -> gpd.GeoDataFrame:
    """Query OSM for grocery / supermarket locations in *place*.

    Combines results from multiple tag queries and deduplicates by
    geometry to avoid double-counting.
    """
    import osmnx as ox

    tag_sets = [
        {"amenity": "supermarket"},
        {"shop": "supermarket"},
        {"shop": "grocery"},
    ]

    frames: list[gpd.GeoDataFrame] = []
    for tags in tag_sets:
        try:
            result = ox.features_from_place(place, tags=tags)
            if not result.empty:
                frames.append(result)
                print(f"  OSM query {tags}: {len(result)} features")
        except Exception as exc:
            print(f"  OSM query {tags} failed: {exc}")

    if not frames:
        print(
            "  WARNING: No grocery/supermarket features returned from OSM. "
            "Returning empty GeoDataFrame."
        )
        return gpd.GeoDataFrame(
            columns=["geometry"], geometry="geometry", crs=TARGET_CRS
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=frames[0].crs)

    # Represent every feature as its centroid point for distance calculations
    combined["geometry"] = combined.geometry.centroid
    combined = combined.drop_duplicates(subset=["geometry"])
    print(f"  Combined & deduplicated groceries: {len(combined)} points")

    combined = combined.to_crs(TARGET_CRS)
    return combined


# ---------------------------------------------------------------------------
# GTFS transit shapes
# ---------------------------------------------------------------------------

def load_gtfs_shapes(data_dir: Path) -> gpd.GeoDataFrame:
    """Build transit-route LineStrings from GTFS shapes.txt.

    Uses trips.txt to identify unique shape_ids per route so that
    duplicate shapes (same geometry used by many trips) are not
    double-counted.

    Returns a GeoDataFrame with one row per unique shape_id in
    EPSG:32617.
    """
    gtfs_dir = _find_gtfs_dir(data_dir)

    shapes_path = gtfs_dir / "shapes.txt"
    trips_path = gtfs_dir / "trips.txt"
    for p in (shapes_path, trips_path):
        if not p.exists():
            raise FileNotFoundError(f"GTFS file not found: {p}")

    shapes_df = pd.read_csv(shapes_path)
    trips_df = pd.read_csv(trips_path)

    required_shape_cols = {"shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"}
    missing = required_shape_cols - set(shapes_df.columns)
    if missing:
        raise KeyError(f"shapes.txt missing columns: {missing}")

    # Keep only shape_ids actually referenced by trips (dedup at route level)
    unique_shapes = trips_df.drop_duplicates(subset=["shape_id"])["shape_id"].unique()
    shapes_df = shapes_df[shapes_df["shape_id"].isin(unique_shapes)].copy()
    print(f"  GTFS: {len(unique_shapes)} unique shape_ids from trips.txt")

    # Sort and build LineStrings
    shapes_df = shapes_df.sort_values(["shape_id", "shape_pt_sequence"])

    lines: list[dict] = []
    for sid, grp in shapes_df.groupby("shape_id"):
        coords = list(zip(grp["shape_pt_lon"], grp["shape_pt_lat"]))
        if len(coords) >= 2:
            lines.append({"shape_id": sid, "geometry": LineString(coords)})

    gdf = gpd.GeoDataFrame(lines, crs=WGS84)
    gdf = gdf.to_crs(TARGET_CRS)
    print(f"  Built {len(gdf)} route LineStrings")
    return gdf

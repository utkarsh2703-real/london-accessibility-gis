# Urban Accessibility Clustering for London, Ontario

## Project Overview

This project identifies and ranks the most underserved census tracts in London, Ontario by measuring accessibility to five categories of essential services: hospitals, schools, grocery stores, public transit, and parks. Using unsupervised machine learning (KMeans clustering), it groups tracts by their accessibility profile and produces a continuous underserved score to recommend priority areas for infrastructure investment.

## Why This Project Matters

Equitable access to essential services is a cornerstone of healthy, liveable cities. Gaps in accessibility disproportionately affect vulnerable populations — including the elderly, low-income families, and people with disabilities. By quantifying service accessibility at the census-tract level and identifying the specific gaps in each neighbourhood, this analysis provides data-driven evidence that urban planners, transit authorities, and policy-makers can use to allocate resources where they are needed most.

## Objective

1. Compute tract-level accessibility features (distances to services, counts of nearby amenities, transit route coverage).
2. Cluster census tracts into groups with similar accessibility profiles using KMeans.
3. Rank clusters from most accessible to least accessible.
4. Produce a continuous underserved score for every tract and identify each tract's single biggest service gap.
5. Visualise results on an interactive Folium map and in static charts.

## Data Sources

| Layer | Source | Format |
|---|---|---|
| Census tracts | Statistics Canada 2021 Census — DLI national CT shapefile | `.shp` (zipped) |
| Bus stops | City of London Open Data — Bus Stop Inventory (Jan 2025) | `.csv` |
| Transit routes | City of London GTFS feed | `google_transit.zip` |
| Schools | City of London Open Data | `.geojson` |
| Hospitals | City of London Open Data | `.geojson` |
| Parks | City of London Open Data | `.geojson` |
| Grocery stores | OpenStreetMap via OSMnx | queried at runtime |

## Methodology

### 1. Data Loading
- Load the national census-tract shapefile and filter to London CMA tracts (CTUID prefix `555`).
- Load bus stops from CSV, schools/hospitals/parks from GeoJSON.
- Query OSM for grocery/supermarket locations.
- Build transit-route LineStrings from GTFS `shapes.txt`, deduplicated via `trips.txt`.

### 2. Spatial Processing
- Reproject every layer to **EPSG:32617** (UTM Zone 17N) for metric distance and area calculations.
- Compute tract centroids for nearest-service distance measurements.

### 3. Feature Engineering
For each census tract:
- `num_bus_stops` — count of bus stops within the tract.
- `dist_nearest_hospital` — distance (m) from tract centroid to nearest hospital.
- `dist_nearest_school` — distance (m) from tract centroid to nearest school.
- `dist_nearest_grocery` — distance (m) from tract centroid to nearest grocery store.
- `num_parks` — count of parks intersecting the tract.
- `transit_coverage` — total length (m) of GTFS route geometry inside the tract.
- Derived density metrics: `bus_stops_per_km2`, `park_count_per_km2`.

### 4. Clustering
- Standardise features with `StandardScaler`.
- Evaluate KMeans for k = 2 .. 8 using the elbow method and silhouette scores.
- Select optimal k (highest silhouette, with manual override option).
- Rank clusters by a composite disadvantage score so that label 0 = most accessible.

### 5. Ranking Underserved Tracts
- Compute a continuous `underserved_score` per tract using standardised deficits.
- Identify the `biggest_accessibility_gap` for each tract (the single worst deficit).

### 6. Visualisation
- Interactive Folium choropleth map with popups (`output/index.html`).
- Elbow curve and silhouette score plots.
- Cluster feature heatmap.
- Top 10 underserved tracts bar chart.

## Project Structure

```
london-accessibility-gis/
├── data/                       # Raw input data files
├── output/                     # Generated outputs
│   ├── index.html              # Interactive Folium map
│   ├── london_accessibility_clusters.geojson
│   ├── cluster_summary.csv
│   └── top_5_underserved.csv
├── notebooks/
│   └── analysis.ipynb          # Main analysis notebook
├── src/
│   ├── load_data.py            # Data loading functions
│   ├── feature_engineering.py  # Spatial feature computation
│   ├── clustering.py           # ML pipeline
│   └── visualize.py            # Plotting & mapping
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone or download the repository
cd london-accessibility-gis

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

## How to Run Locally

1. Place the input data files in the `data/` directory (or adjust `DATA_DIR` in the notebook).
2. Launch Jupyter:
   ```bash
   jupyter notebook notebooks/analysis.ipynb
   ```
3. Run all cells. Outputs will be written to the `output/` directory.

## How to Run in Google Colab

1. Upload the repository (or mount your Google Drive containing it).
2. Open `notebooks/analysis.ipynb` in Colab.
3. In the first cell, install dependencies:
   ```python
   !pip install -r ../requirements.txt
   ```
4. Update `DATA_DIR` to point to your uploaded data files.
5. Run all cells.

## Outputs

| File | Description |
|---|---|
| `output/london_accessibility_clusters.geojson` | Full tract GeoDataFrame with features, scores, and cluster labels |
| `output/cluster_summary.csv` | Mean/median feature values per cluster |
| `output/top_5_underserved.csv` | Top 5 most underserved tracts |
| `output/index.html` | Interactive Folium map |
| `output/elbow_curve.png` | Elbow plot |
| `output/silhouette_scores.png` | Silhouette plot |
| `output/cluster_heatmap.png` | Cluster feature heatmap |
| `output/top_underserved.png` | Top underserved bar chart |

## Example Results

*After running the notebook, paste screenshots or key numbers here.*

- Number of London census tracts analysed: ~56
- Optimal k selected: (determined at runtime)
- Most underserved tract: (determined at runtime)
- Most common biggest gap: (determined at runtime)

## Possible Future Improvements

- Incorporate population and demographic data from the census to weight accessibility by need.
- Add walk-time isochrones (network distance) instead of straight-line centroid distances.
- Include additional service categories (libraries, pharmacies, childcare centres).
- Use DBSCAN or hierarchical clustering for comparison.
- Integrate real-time transit frequency data from GTFS `stop_times.txt`.
- Build a Streamlit or Dash dashboard for interactive exploration.

## Notes, Assumptions, and Limitations

- London CMA census tracts are filtered from the national CT shapefile using CTUID values starting with `555`.
- Parks are sourced from the uploaded `Parks.geojson` file, **not** from OpenStreetMap.
- Grocery stores are queried from OSM at runtime; results may vary over time.
- Transit coverage is based on GTFS route-shape geometry length, not trip frequency.
- All distances are Euclidean centroid-to-nearest-service distances in EPSG:32617 (metres), not network/walking distances.
- KMeans labels are arbitrary — the project includes a ranking step to ensure label 0 = most accessible.
- A fixed `random_state=42` is used for reproducibility.

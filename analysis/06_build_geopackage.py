"""
06_build_geopackage.py
----------------------
Assemble the final map dataset for the app: every pedestrian-involved
fatal crash (2016-2024) with coordinates, display factor, narrative,
and popup details, saved as a GeoPackage (+ a compact parquet fallback).

The Streamlit app reads ONLY this file -- retraining the model and
rerunning 03/05 then this script updates the app with no code changes.

Usage: python 06_build_geopackage.py <data_dir>
Requires: pandas geopandas shapely pyarrow
"""

import sys, os
import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "."

pq, cs = os.path.join(DATA, "crash_level.parquet"), os.path.join(DATA, "crash_level.csv")
df = pd.read_parquet(pq) if os.path.exists(pq) else pd.read_csv(cs, low_memory=False)
tf = pd.read_csv(os.path.join(DATA, "ped_top_factors.csv"))
nar = pd.read_csv(os.path.join(DATA, "narratives.csv"))

d = (df[df.ped_involved == 1]
     .merge(tf, on="CRASH_ID", how="left")
     .merge(nar, on="CRASH_ID", how="left"))

# valid coordinates only
d = d[d.LATITUDE.between(24, 72) & d.LONGITUD.between(-180, -60)].copy()

# color per display factor family (map legend)
BEHAVIOR = {"hit_run", "speeding_involved", "drinking_driver",
            "suv_involved", "pickup_involved", "heavy_truck_involved"}
def color(f):
    if not isinstance(f, str): return "#718096"          # gray  - other
    if f in BEHAVIOR: return "#1a202c"                   # black - driver behavior
    if f.startswith("lighting"): return "#2c5282"        # blue  - lighting
    if f.startswith("junction"): return "#c53030"        # red   - crossing/junction
    if f.startswith("road_class"): return "#dd6b20"      # orange- road design
    if f == "hour": return "#6b46c1"                     # purple- time of day
    if f.startswith("weather"): return "#2f855a"         # green - weather/exposure
    return "#718096"

d["factor_color"] = d.display_factor.map(color)
d["factor_label"] = d.display_factor_label.fillna("Other")

keep = ["CRASH_ID", "YEAR", "STATENAME", "LATITUDE", "LONGITUD",
        "lighting", "road_class", "junction", "ped_age", "ped_sex",
        "ped_loc_type", "factor_label", "factor_color", "narrative"]
out = d[keep].rename(columns={"LONGITUD": "LONGITUDE"})

out.to_parquet(os.path.join(DATA, "ped_map_data.parquet"), index=False)
print(f"{len(out):,} mappable pedestrian crashes -> ped_map_data.parquet")

try:
    import geopandas as gpd
    from shapely.geometry import Point
    gdf = gpd.GeoDataFrame(
        out, geometry=[Point(xy) for xy in zip(out.LONGITUDE, out.LATITUDE)],
        crs="EPSG:4326")
    gdf.to_file(os.path.join(DATA, "GeoPedFatalities.gpkg"),
                layer="ped_fatalities", driver="GPKG")
    print("-> GeoPedFatalities.gpkg (layer: ped_fatalities)")
except ImportError:
    print("geopandas not installed - skipped gpkg (parquet is enough for the app)")
print(out.factor_label.value_counts().head(8).to_string())

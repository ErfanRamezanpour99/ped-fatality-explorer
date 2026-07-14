"""
backend_route.py -- geocoding + routing + corridor filtering.
Adapted from the original Route Fatality Explorer backend (same
Nominatim -> ORS -> buffer -> spatial filter pattern), with:
  - selectable routing profile: foot-walking or driving-car
  - shapely STRtree spatial filter on the pedestrian dataset
    (no geopandas dependency)
"""

import json
import os
import requests
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
from shapely.ops import transform
import pyproj

_TO_MERC = pyproj.Transformer.from_crs(4326, 3857, always_xy=True).transform


def geocode(address):
    """Return (lon, lat) or None."""
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": address, "format": "json", "limit": 1},
            headers={"User-Agent": "PedSafetyExplorer/1.0"},
            timeout=10,
        )
        data = r.json()
        if not data:
            return None
        return float(data[0]["lon"]), float(data[0]["lat"])
    except Exception:
        return None


def get_route_crashes(df, origin, destination, profile="foot-walking", buffer_m=100):
    """
    df: pandas DataFrame with LONGITUDE/LATITUDE columns (ped crashes).
    Returns dict with route_coords + filtered crash records, or {'error': ...}.
    """
    o, d = geocode(origin), geocode(destination)
    if o is None:
        return {"error": f"Could not find location: {origin}"}
    if d is None:
        return {"error": f"Could not find location: {destination}"}

    key = os.environ.get("ORS_API_KEY")
    if not key:                      # Streamlit Cloud stores secrets in st.secrets
        try:
            import streamlit as st
            key = st.secrets.get("ORS_API_KEY")
        except Exception:
            key = None
    if not key:
        return {"error": "ORS_API_KEY not configured (add it in app secrets)."}

    try:
        r = requests.post(
            f"https://api.openrouteservice.org/v2/directions/{profile}",
            headers={"Authorization": key, "Content-Type": "application/json"},
            data=json.dumps({"coordinates": [list(o), list(d)]}),
            timeout=25,
        )
        if r.status_code != 200:
            return {"error": "Routing failed.", "details": r.text[:300]}
        routes = r.json().get("routes") or []
        if not routes:
            return {"error": "No route found."}
        import polyline
        latlon = polyline.decode(routes[0]["geometry"])
    except Exception as e:
        return {"error": "Routing failed.", "details": str(e)}

    route_coords = [(lon, lat) for lat, lon in latlon]
    line_merc = transform(_TO_MERC, LineString(route_coords))
    corridor = line_merc.buffer(buffer_m)

    pts = [transform(_TO_MERC, Point(xy))
           for xy in zip(df.LONGITUDE.values, df.LATITUDE.values)]
    tree = STRtree(pts)
    hit_idx = [i for i in tree.query(corridor) if corridor.contains(pts[i])]
    hits = df.iloc[sorted(hit_idx)]

    return {
        "route_coords": route_coords,
        "crashes": hits.to_dict(orient="records"),
        "distance_km": round(line_merc.length / 1000, 1),
    }

# 🚶 Pedestrian Fatality Explorer

**Interactive map of 60,022 pedestrian deaths on U.S. roads (2016–2024) — each one explained.**

Companion app to the ITE 2026 Annual Meeting poster *Where Pedestrians Die — and Why* (Detroit, MI).

🗺️ **Live app:** https://ped-fatality-explorergit-5kadirlmyieldrgyjeq7xj.streamlit.app/
🔬 **Analysis pipeline (full reproducibility):** https://github.com/ErfanRamezanpour99/fars-pedestrian-analysis

---

## What it does

Every pedestrian-involved fatal crash from NHTSA's FARS database (2016–2024) is plotted on an interactive map. Each dot is colored by the condition that most set that crash apart — according to SHAP explanations from an XGBoost involvement model (ROC-AUC 0.885) — and clicking it shows a plain-language narrative of what happened plus a candidate countermeasure.

Two modes:

- **🗺️ Explore the map** — pan/zoom through all 60,022 crashes, filter, and tap any dot for its story.
- **🚶 Check a walking route** — enter two addresses; the app geocodes them, routes between them, and shows every pedestrian fatality within a corridor along the way.

## How it works

```
app.py                 Streamlit UI (map, filters, narratives)
backend_route.py       geocoding (Nominatim) + routing (OpenRouteService) + corridor filter (shapely STRtree)
ped_map_data.parquet   the ONLY data file the app reads (~3.4 MB)
requirements.txt
```

**Data contract:** the app reads a single file, `ped_map_data.parquet`, produced by script `06_build_geopackage.py` in the [analysis repo](https://github.com/ErfanRamezanpour99/fars-pedestrian-analysis). To update the app after a model change: rerun the pipeline, replace the parquet, push. The app code never changes.

## Run locally

```bash
git clone https://github.com/ErfanRamezanpour99/ped-fatality-explorer.git
cd ped-fatality-explorer
pip install -r requirements.txt
streamlit run app.py
```

The map mode works out of the box. The walking-route mode needs a free [OpenRouteService](https://openrouteservice.org/) API key, provided either as an environment variable (`ORS_API_KEY`) or in Streamlit secrets:

```toml
# .streamlit/secrets.toml
ORS_API_KEY = "your-key-here"
```

## Data & model

- **Data:** NHTSA Fatality Analysis Reporting System (FARS), 2016–2024 (public domain)
- **Model:** XGBoost pedestrian-involvement classifier, explained per-crash with SHAP; full methodology, validation, and reproduction steps in the [analysis repo](https://github.com/ErfanRamezanpour99/fars-pedestrian-analysis)

## Citation

> Ramezanpour, E. (2026). *Where Pedestrians Die — and Why: Explainable Machine Learning on FARS 2016–2024.* Poster, ITE 2026 Annual Meeting, Detroit, MI.

## Contact

Erfan Ramezanpour — eramezanpour1@ua.edu — The University of Alabama

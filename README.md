# Pedestrian Fatality Explorer

**Where Pedestrians Die, and Why** — explainable machine learning on 60,000
U.S. pedestrian fatalities (FARS 2016–2024), translated into plain-language,
map-based decision support.

Poster: ITE 2026 Annual Meeting · Erfan Ramezanpour, Alex Hainen · The University of Alabama

**Live app:** https://ped-fatality-explorergit-5kadirlmyieldrgyjeq7xj.streamlit.app/

## What's here

| Path | What it is |
|------|-----------|
| `app.py`, `backend_route.py`, `ped_map_data.parquet` | The Streamlit map app (deployed on Streamlit Community Cloud) |
| `analysis/` | The full reproducible pipeline: 11 numbered scripts, one command (`run_all.py`) rebuilds every number, figure, narrative, and the app dataset from raw FARS |
| `poster/` | The conference poster (PDF) |

## Headline findings

- Nearly **1 in 3** pedestrian deaths occur in a single scenario — dark, urban,
  arterial, away from an intersection — where 45.5% of fatal crashes involve a
  pedestrian (2.5× the average).
- Risk peaks **5–9 PM in darkness**, not late at night.
- **64%** of hit-and-run fatal crashes kill a pedestrian.
- Model: XGBoost, ROC-AUC 0.885 (logistic baseline 0.869), 0.888 on a
  2023–24 temporal holdout. Pre-crash features only; leakage documented
  and excluded.

## Reproduce everything

Download FARS National CSV zips (2016–2024) from NHTSA into `raw_fars/`
(see `analysis/README.md`), then:

```
pip install pandas numpy scikit-learn xgboost matplotlib pyarrow qrcode[pil]
python analysis/run_all.py raw_fars analysis/my_run
```

Data: NHTSA FARS (public domain) + U.S. Census ACS 5-year (2023).

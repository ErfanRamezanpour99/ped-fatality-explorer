# ITE 2026 Poster - Reproducible Analysis Pipeline

**Project:** Where Pedestrians Die - and Why (FARS 2016-2024, XGBoost + SHAP,
narratives, interactive map).

## One command reproduces everything

    pip install pandas numpy scikit-learn xgboost matplotlib pyarrow qrcode[pil]
    python run_all.py "D:/UPDATED FARS/raw_fars" "D:/UPDATED FARS/analysis/my_run"

Every poster number is printed to the console; every figure, table, narrative,
and the app dataset is written to the output folder.

## Script map (each is standalone; run in this order)

| # | script | produces |
|---|--------|----------|
| 01 | 01_build_dataset.py | crash_level dataset (1 row/fatal crash, target ped_involved) |
| 09 | 09_enrich_vehicle_features.py | speeding/impaired/hit-run/vehicle-type flags |
| 10 | 10_acs_join.py | county income + car access (needs acs_counties.json) |
| 02 | 02_train_model.py | XGBoost model + metrics (ROC-AUC 0.885) |
| 03 | 03_shap_analysis.py | global importance + per-crash top-3 factors |
| 04 | 04_actionable_findings.py | archetype ladder, labels, vehicle trend, ages, countermeasures |
| 05 | 05_narratives.py | 60,562 plain-language narratives |
| 06 | 06_build_geopackage.py | ped_map_data.parquet (the app reads only this) |
| 07 | 07_poster_figures.py | all poster figures + QR + feature dictionary |
| 11 | 11_model_validation.py | logistic baseline, temporal holdout, interaction findings |
| 08 | 08_build_static_site.py | (contingency) static-site data files |

## Key modeling decisions (defend these)

- Target: ped_involved = any PERSON.PER_TYP==5. Valid because both classes
  exist in FARS (unlike fatal-vs-nonfatal, which FARS cannot support).
- Feature rule: value must be knowable BEFORE the crash. Excluded as leakage:
  HARM_EV, MAN_COLL, PEDS counts, VE_TOTAL, REL_ROAD, HIT_RUN (post-crash
  action), all PBTYPE fields (pedestrian-only). Hit-run is reported
  descriptively (64% of hit-run fatal crashes involve a pedestrian).
- Class imbalance (18.6% positive): scale_pos_weight ~ 4.4 -> recall 0.82 at
  precision 0.49 (screening trade-off; threshold adjustable).
- Validation: random split ROC-AUC 0.885; temporal split (train 16-22,
  test 23-24) 0.888; logistic baseline for comparison in script 11.

## The app

`../ped_app/` - Streamlit app deployed on Streamlit Community Cloud.
To update after a model change: rerun pipeline, copy ped_map_data.parquet
into the GitHub repo, push. The app code never changes.

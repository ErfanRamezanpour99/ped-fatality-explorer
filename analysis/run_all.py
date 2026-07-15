"""
run_all.py -- reproduce the entire ITE poster analysis with one command.

    python run_all.py "D:/UPDATED FARS/raw_fars" "D:/UPDATED FARS/analysis/my_run"

Prints every number, table, and finding on the poster, and writes every
figure, the narratives, and the app's map dataset to <out_dir>.
Runtime: roughly 15-30 minutes depending on machine.

Requires: pip install pandas numpy scikit-learn xgboost matplotlib pyarrow qrcode[pil]
(geopandas+shapely optional, only for the GeoPackage export)
"""
import subprocess, sys, os, time

RAW = sys.argv[1] if len(sys.argv) > 1 else "../raw_fars"
OUT = sys.argv[2] if len(sys.argv) > 2 else "my_run"
HERE = os.path.dirname(os.path.abspath(__file__))
ACS = os.path.join(HERE, "acs_counties.json")
os.makedirs(OUT, exist_ok=True)

STEPS = [
    ("01  build crash-level dataset",      ["01_build_dataset.py", RAW, OUT]),
    ("09  vehicle/driver behavior flags",  ["09_enrich_vehicle_features.py", OUT, RAW]),
    ("10  ACS county join",                ["10_acs_join.py", OUT, ACS]),
    ("02  train XGBoost",                  ["02_train_model.py", OUT]),
    ("03  SHAP global + per-crash",        ["03_shap_analysis.py", OUT]),
    ("04  actionable findings",            ["04_actionable_findings.py", OUT, RAW]),
    ("05  narratives",                     ["05_narratives.py", OUT]),
    ("06  map dataset (app input)",        ["06_build_geopackage.py", OUT]),
    ("07  poster figures + QR + dictionary", ["07_poster_figures.py", OUT, OUT]),
    ("11  validation (baseline/temporal/interactions)", ["11_model_validation.py", OUT]),
]

for name, args in STEPS:
    print(f"\n{'='*70}\n>>> {name}\n{'='*70}", flush=True)
    t = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, args[0])] + args[1:])
    if r.returncode != 0:
        print(f"STEP FAILED: {name} -- fix and rerun; later steps depend on it.")
        sys.exit(1)
    print(f"[{name}: {time.time()-t:.0f}s]")

print("\nAll steps complete. Poster inputs are in:", OUT)

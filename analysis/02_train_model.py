"""
02_train_model.py
-----------------
Train the XGBoost pedestrian-involvement classifier.

  Dependent variable:   ped_involved (1 = fatal crash involved a pedestrian)
  Independent variables (pre-crash context only):
      categorical: lighting, weather, road_class, junction  (one-hot encoded)
      numeric/binary: urban, work_zone, hour, month, weekend
  NOTE: rel_road exists in the dataset but is EXCLUDED from the model --
  it describes where the harmful event occurred (on roadway vs roadside),
  which is determined by the crash outcome itself (leakage).

Outputs (in <data_dir>):
  ped_model.json      trained XGBoost model
  feature_cols.csv    one-hot feature names (order matters for SHAP)
  metrics.json        test-set performance
  test_idx.csv        row indices of the held-out test set

Usage:  python 02_train_model.py <data_dir>
        <data_dir> must contain crash_level.csv (or .parquet)
        e.g.  python 02_train_model.py "D:/UPDATED FARS/analysis/output"

Requires: pip install pandas numpy scikit-learn xgboost
"""

import sys, os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, confusion_matrix)
import xgboost as xgb

DATA = sys.argv[1] if len(sys.argv) > 1 else "."

# ---------- load ----------
pq = os.path.join(DATA, "crash_level.parquet")
cs = os.path.join(DATA, "crash_level.csv")
df = pd.read_parquet(pq) if os.path.exists(pq) else pd.read_csv(cs, low_memory=False)
print(f"Loaded {len(df):,} fatal crashes | pedestrian-involved: "
      f"{df.ped_involved.sum():,} ({df.ped_involved.mean()*100:.1f}%)")

# ---------- features ----------
CATS = ["lighting", "weather", "road_class", "junction"]
NUMS = ["urban", "work_zone", "hour", "month", "weekend",
        "speeding_involved",  "drinking_driver",
        "suv_involved", "pickup_involved", "heavy_truck_involved", "county_median_income", "county_pct_no_vehicle"]
X = pd.get_dummies(df[CATS + NUMS], columns=CATS, dummy_na=False)
y = df["ped_involved"].values
print(f"Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} one-hot columns")

# ---------- split (stratified 80/20) ----------
Xtr, Xte, ytr, yte, itr, ite = train_test_split(
    X, y, df.index, test_size=0.2, stratify=y, random_state=42)

# ---------- train ----------
spw = (ytr == 0).sum() / (ytr == 1).sum()   # class imbalance weight (~4.4)
model = xgb.XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.1,
    subsample=0.9, colsample_bytree=0.9, tree_method="hist",
    scale_pos_weight=spw, eval_metric=["logloss", "auc"],
    early_stopping_rounds=25, n_jobs=-1, random_state=42)
model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)

# ---------- evaluate ----------
p = model.predict_proba(Xte)[:, 1]
pred = (p >= 0.5).astype(int)
metrics = {
    "roc_auc":  round(roc_auc_score(yte, p), 4),
    "pr_auc":   round(average_precision_score(yte, p), 4),
    "baseline_pr": round(yte.mean(), 4),
    "f1":       round(f1_score(yte, pred), 4),
    "precision": round(precision_score(yte, pred), 4),
    "recall":   round(recall_score(yte, pred), 4),
    "trees_used": int(model.best_iteration),
    "n_train": int(len(ytr)), "n_test": int(len(yte)),
    "scale_pos_weight": round(spw, 3),
}
print("\n=== Test-set performance ===")
print(json.dumps(metrics, indent=2))
tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
print(f"\nConfusion matrix (threshold 0.5):")
print(f"  actual non-ped: {tn:>6} correct | {fp:>6} false alarm")
print(f"  actual ped:     {fn:>6} missed  | {tp:>6} caught")

# ---------- save ----------
model.save_model(os.path.join(DATA, "ped_model.json"))
X.columns.to_series().to_csv(os.path.join(DATA, "feature_cols.csv"), index=False)
pd.Series(ite).to_csv(os.path.join(DATA, "test_idx.csv"), index=False)
json.dump(metrics, open(os.path.join(DATA, "metrics.json"), "w"), indent=2)
print(f"\nSaved model + metrics -> {DATA}")

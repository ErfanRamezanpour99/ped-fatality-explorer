"""
11_model_validation.py
----------------------
Validation analyses reported on the poster:
  1. Logistic-regression baseline vs XGBoost (same features, same split)
  2. Temporal holdout: train 2016-2022, test 2023-2024
  3. Conditional SHAP ("interaction") findings:
       - dark-unlit contribution by road class
       - evening-hours contribution, dark vs daylight
       - arterial contribution by lighting

Usage: python 11_model_validation.py <data_dir>
Requires: pandas numpy scikit-learn xgboost
"""
import sys, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score
import xgboost as xgb

DATA = sys.argv[1] if len(sys.argv) > 1 else "."
pq, cs = os.path.join(DATA, "crash_level.parquet"), os.path.join(DATA, "crash_level.csv")
df = pd.read_parquet(pq) if os.path.exists(pq) else pd.read_csv(cs, low_memory=False)

CATS = ["lighting", "weather", "road_class", "junction"]
NUMS = ["urban", "work_zone", "hour", "month", "weekend", "speeding_involved",
        "drinking_driver", "suv_involved", "pickup_involved", "heavy_truck_involved",
        "county_median_income", "county_pct_no_vehicle"]
X = pd.get_dummies(df[CATS + NUMS], columns=CATS, dummy_na=False)
y = df.ped_involved.values

def report(tag, yte, p, thresh=0.5):
    pred = (p >= thresh).astype(int)
    print(f"{tag:38s} ROC-AUC {roc_auc_score(yte,p):.4f} | PR-AUC "
          f"{average_precision_score(yte,p):.4f} | recall {recall_score(yte,pred):.3f} "
          f"| precision {precision_score(yte,pred):.3f}")

def xgb_model(ytr):
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    return xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, tree_method="hist",
        scale_pos_weight=spw, eval_metric="auc", early_stopping_rounds=25,
        n_jobs=-1, random_state=42)

# ---------- 1. baseline vs XGBoost (random split) ----------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("=== 1. Model comparison (random 80/20 split) ===")
logit = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(max_iter=2000, class_weight="balanced"))
logit.fit(Xtr, ytr)
report("Logistic regression (baseline)", yte, logit.predict_proba(Xte)[:, 1])
m = xgb_model(ytr); m.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
report("XGBoost (final model)", yte, m.predict_proba(Xte)[:, 1])

# ---------- 2. temporal holdout ----------
print("\n=== 2. Temporal validation ===")
tr, te = df.YEAR <= 2022, df.YEAR >= 2023
mt = xgb_model(y[tr]); mt.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=False)
report("XGBoost train 2016-22 -> test 2023-24", y[te], mt.predict_proba(X[te])[:, 1])

# ---------- 3. conditional SHAP (interactions) ----------
print("\n=== 3. Conditional SHAP contributions ===")
model = xgb.XGBClassifier(); model.load_model(os.path.join(DATA, "ped_model.json"))
rng = np.random.RandomState(0)
idx = rng.choice(len(X), min(15000, len(X)), replace=False)
dm = xgb.DMatrix(X.iloc[idx], feature_names=list(X.columns))
sv = pd.DataFrame(model.get_booster().predict(dm, pred_contribs=True)[:, :-1],
                  columns=X.columns)
sub = df.iloc[idx].reset_index(drop=True)
Xs = X.iloc[idx].reset_index(drop=True)

print("\n(a) Dark-unlit contribution by road class (present only):")
mask = Xs["lighting_Dark - Not Lighted"] == 1
print(sv.loc[mask, "lighting_Dark - Not Lighted"]
      .groupby(sub.loc[mask, "road_class"]).agg(["mean", "count"]).round(3).to_string())

print("\n(b) 'hour' contribution during 5-9 PM, dark vs daylight:")
ev = sub.hour.between(17, 21)
dk = sub.lighting.fillna("").str.startswith("Dark")
print(f"  evening & dark:     {sv.loc[ev & dk, 'hour'].mean():.3f} (n={int((ev&dk).sum())})")
print(f"  evening & daylight: {sv.loc[ev & ~dk, 'hour'].mean():.3f} (n={int((ev&~dk).sum())})")

print("\n(c) Principal-arterial contribution by lighting (present only):")
art = Xs["road_class_Principal Arterial"] == 1
print(sv.loc[art, "road_class_Principal Arterial"]
      .groupby(sub.loc[art, "lighting"]).agg(["mean", "count"]).round(3).to_string())

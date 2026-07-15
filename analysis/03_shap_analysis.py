"""
03_shap_analysis.py
-------------------
Explain the trained model with SHAP values (computed via XGBoost's native
`pred_contribs=True`, which implements the exact TreeSHAP algorithm -- no
extra `shap` package needed).

Produces (in <data_dir>):
  shap_global_importance.csv   mean |SHAP| per one-hot feature + per parent variable
  ped_top_factors.csv          per pedestrian crash: top-3 positive contributors,
                               plus a 'display_factor' (top ACTIONABLE factor,
                               excluding generic urban/month/weekend) and
                               readable labels for map popups
  fig_global_importance.png    bar chart, grouped by parent variable
  fig_top_factors.png          bar chart of display factors across ped crashes
  fig_ped_rates.png            descriptive: ped share of fatal crashes by
                               lighting / road class / urban

Usage:  python 03_shap_analysis.py <data_dir>
        (run 01 and 02 first; <data_dir> holds crash_level + ped_model.json)

Requires: pip install pandas numpy xgboost matplotlib
"""

import sys, os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = sys.argv[1] if len(sys.argv) > 1 else "."
PART = sys.argv[2] if len(sys.argv) > 2 else "all"   # all | global | percrash | figures

# ---------- readable names for one-hot features ----------
FRIENDLY = {
    "urban": "Urban area", "work_zone": "Work zone", "hour": "Time of day",
    "hit_run": "Hit-and-run driver", "speeding_involved": "Speeding involved",
    "drinking_driver": "Impaired driver", "suv_involved": "SUV involved",
    "pickup_involved": "Pickup involved", "heavy_truck_involved": "Heavy truck involved",
    "county_median_income": "County income", "county_pct_no_vehicle": "County car access",
    "month": "Month", "weekend": "Weekend",
    "lighting_Daylight": "Daylight", "lighting_Dark - Lighted": "Dark but lighted road",
    "lighting_Dark - Not Lighted": "Dark, unlit road", "lighting_Dawn": "Dawn",
    "lighting_Dusk": "Dusk", "lighting_Dark - Unknown Lighting": "Dark (lighting unknown)",
    "road_class_Principal Arterial": "Principal arterial road",
    "road_class_Minor Arterial": "Minor arterial road",
    "road_class_Interstate": "Interstate", "road_class_Freeway/Expressway": "Freeway",
    "road_class_Major Collector": "Major collector road",
    "road_class_Minor Collector": "Minor collector road",
    "road_class_Local Road": "Local road",
    "junction_Intersection": "At intersection",
    "junction_Intersection-Related": "Intersection-related location",
    "junction_Non-Junction": "Away from intersection (midblock)",
    "junction_Through Roadway": "Through roadway",
    "junction_Entrance/Exit Ramp": "Highway ramp",
    "junction_Driveway Access": "Driveway access",
}
def nice(col):
    if col in FRIENDLY: return FRIENDLY[col]
    return col.replace("_", ": ", 1) if "_" in col else col

CATS = ["lighting", "weather", "road_class", "junction"]
NUMS = ["urban", "work_zone", "hour", "month", "weekend",
        "speeding_involved",  "drinking_driver",
        "suv_involved", "pickup_involved", "heavy_truck_involved", "county_median_income", "county_pct_no_vehicle"]
GENERIC = {"urban", "month", "weekend", "county_median_income", "county_pct_no_vehicle"}   # true-but-unactionable; kept out of map colors

# ---------- load data + model ----------
pq, cs = os.path.join(DATA, "crash_level.parquet"), os.path.join(DATA, "crash_level.csv")
df = pd.read_parquet(pq) if os.path.exists(pq) else pd.read_csv(cs, low_memory=False)
X = pd.get_dummies(df[CATS + NUMS], columns=CATS, dummy_na=False)
model = xgb.XGBClassifier(); model.load_model(os.path.join(DATA, "ped_model.json"))
booster = model.get_booster()
cols = np.array(X.columns)

def shap_values(Xsub, batch=20000):
    """Exact TreeSHAP contributions, batched to limit memory."""
    parts = []
    for s in range(0, len(Xsub), batch):
        dm = xgb.DMatrix(Xsub.iloc[s:s+batch], feature_names=list(cols))
        parts.append(booster.predict(dm, pred_contribs=True)[:, :-1])  # drop bias col
    return np.vstack(parts)

# ================= PART 1: global importance =================
if PART in ("all", "global"):
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X), min(30000, len(X)), replace=False)
    sv = shap_values(X.iloc[idx])
    imp = pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False)
    parent = {c: next((p for p in CATS if c.startswith(p + "_")), c) for c in cols}
    grouped = imp.groupby(pd.Series(parent)).sum().sort_values(ascending=False)
    out = pd.DataFrame({"mean_abs_shap": imp.round(4)})
    out["parent"] = [parent[c] for c in imp.index]
    out.to_csv(os.path.join(DATA, "shap_global_importance.csv"))
    grouped.round(4).to_csv(os.path.join(DATA, "shap_grouped_importance.csv"))
    print("=== Global importance (mean |SHAP|, grouped) ===")
    print(grouped.round(3).to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped.sort_values().plot.barh(ax=ax, color="#2b6cb0")
    ax.set_xlabel("Mean |SHAP| (impact on pedestrian-involvement prediction)", fontsize=12)
    ax.set_title("What distinguishes pedestrian fatal crashes?\nGlobal feature importance (2016-2024 FARS)", fontsize=13)
    ax.tick_params(labelsize=12)
    plt.tight_layout(); plt.savefig(os.path.join(DATA, "fig_global_importance.png"), dpi=200)
    print("saved fig_global_importance.png")

# ================= PART 2: per-crash top factors =================
if PART in ("all", "percrash"):
    mask = (df.ped_involved == 1).values
    Xp = X[mask]
    sv = shap_values(Xp)
    vals = Xp.values.astype(float)
    present = np.where(vals != 0, 1.0, np.nan)     # feature must apply to this crash
    sv_pos = np.where(sv > 0, sv, np.nan) * present  # must push TOWARD pedestrian

    def top_k(matrix, colnames, k=3):
        order = np.argsort(np.nan_to_num(matrix, nan=-9e9), axis=1)[:, ::-1]
        tops, valid = [], []
        for j in range(k):
            v = np.take_along_axis(matrix, order[:, j:j+1], 1).ravel()
            t = colnames[order[:, j]].astype(object)
            t[~np.isfinite(v)] = None
            tops.append(t); valid.append(v)
        return tops, valid

    (t1, t2, t3), (v1, _, _) = top_k(sv_pos, cols)
    keep = np.array([c not in GENERIC for c in cols])
    (d1, _, _), (dv1, _, _) = top_k(sv_pos[:, keep], cols[keep])

    out = df.loc[mask, ["CRASH_ID"]].copy()
    out["top1"], out["top2"], out["top3"] = t1, t2, t3
    out["top1_shap"] = np.round(np.nan_to_num(v1), 4)
    out["display_factor"] = d1
    out["display_factor_label"] = [nice(c) if c else None for c in d1]
    out.to_csv(os.path.join(DATA, "ped_top_factors.csv"), index=False)
    print("\n=== Top actionable factor across pedestrian crashes ===")
    print(out.display_factor_label.value_counts().head(12).to_string())
    print(f"rows: {len(out):,} -> ped_top_factors.csv")

    counts = out.display_factor_label.value_counts().head(10)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    counts.sort_values().plot.barh(ax=ax, color="#c53030")
    ax.set_xlabel("Pedestrian fatal crashes (2016-2024)", fontsize=12)
    ax.set_title("Most influential context factor per pedestrian fatal crash\n(SHAP, excluding generic urban/seasonal effects)", fontsize=13)
    ax.tick_params(labelsize=12)
    plt.tight_layout(); plt.savefig(os.path.join(DATA, "fig_top_factors.png"), dpi=200)
    print("saved fig_top_factors.png")

# ================= PART 3: descriptive rates figure =================
if PART in ("all", "figures"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    specs = [("lighting", "Lighting"), ("road_class", "Road class"), ("urban", "Area type")]
    for ax, (col, title) in zip(axes, specs):
        r = df.groupby(col).ped_involved.mean().mul(100).sort_values()
        if col == "urban": r.index = ["Rural", "Urban"]
        r.plot.barh(ax=ax, color="#2f855a")
        ax.set_title(f"Ped share of fatal crashes by {title.lower()}", fontsize=12)
        ax.set_xlabel("% pedestrian-involved", fontsize=11)
        ax.tick_params(labelsize=10); ax.set_ylabel("")
    plt.tight_layout(); plt.savefig(os.path.join(DATA, "fig_ped_rates.png"), dpi=200)
    print("saved fig_ped_rates.png")

print("\nDone.")

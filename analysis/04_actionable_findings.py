"""
04_actionable_findings.py
-------------------------
Sharpen the analysis into actionable, poster-ready findings.

Parts (run all, or pass one as 3rd argument):
  labels     -> resolve vague display factors: 'hour' becomes the crash's actual
                time window (e.g. 'Late night (12-4 AM)'); clear weather is
                marked as an exposure effect. Rewrites ped_top_factors.csv.
  archetype  -> risk-archetype coverage & lift: what share of pedestrian fatal
                crashes match {dark + urban + arterial + non-intersection}?
  vehicles   -> striking-vehicle body type per year (SUV/pickup trend).
                Needs vehicle.csv per year in <raw_dir>.
  ages       -> pedestrian age distribution vs other fatal-crash victims.
  depplot    -> SHAP dependence: hour vs SHAP(hour), colored by lighting.
  counter    -> writes factor -> FHWA countermeasure mapping (countermeasures.csv)

Usage:  python 04_actionable_findings.py <data_dir> <raw_dir> [part]
  e.g.  python 04_actionable_findings.py "D:/UPDATED FARS/analysis/my_run" "D:/UPDATED FARS/raw_fars"

Requires: pandas numpy xgboost matplotlib
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = sys.argv[1] if len(sys.argv) > 1 else "."
RAW  = sys.argv[2] if len(sys.argv) > 2 else "raw_fars"
PART = sys.argv[3] if len(sys.argv) > 3 else "all"

pq, cs = os.path.join(DATA, "crash_level.parquet"), os.path.join(DATA, "crash_level.csv")
df = pd.read_parquet(pq) if os.path.exists(pq) else pd.read_csv(cs, low_memory=False)

def hour_window(h):
    if pd.isna(h): return None
    h = int(h)
    if h < 5:  return "Late night (12-4 AM)"
    if h < 10: return "Morning (5-9 AM)"
    if h < 16: return "Midday (10 AM-3 PM)"
    if h < 20: return "Evening (4-7 PM)"
    return "Night (8-11 PM)"

# ---------------------------------------------------------------- labels
if PART in ("all", "labels"):
    tf = pd.read_csv(os.path.join(DATA, "ped_top_factors.csv"))
    tf = tf.merge(df[["CRASH_ID", "hour"]], on="CRASH_ID", how="left")
    is_hour = tf.display_factor == "hour"
    tf.loc[is_hour, "display_factor_label"] = tf.loc[is_hour, "hour"].map(hour_window)
    tf.loc[tf.display_factor == "weather_Clear", "display_factor_label"] = \
        "Clear weather (more walking exposure)"
    tf.drop(columns=["hour"]).to_csv(os.path.join(DATA, "ped_top_factors.csv"), index=False)
    print("=== Display factors after label fix ===")
    print(tf.display_factor_label.value_counts().head(14).to_string())

# ---------------------------------------------------------------- archetype
if PART in ("all", "archetype"):
    dark = df.lighting.isin(["Dark - Lighted", "Dark - Not Lighted", "Dark - Unknown Lighting"])
    urban = df.urban == 1
    arterial = df.road_class.isin(["Principal Arterial", "Minor Arterial"])
    nonint = ~df.junction.isin(["Intersection", "Intersection-Related"])
    base = df.ped_involved.mean()

    print("\n=== Risk archetypes (pedestrian share of fatal crashes: "
          f"{base*100:.1f}% baseline) ===")
    rows = []
    combos = {
        "Dark": dark,
        "Dark + Urban": dark & urban,
        "Dark + Urban + Arterial": dark & urban & arterial,
        "Dark + Urban + Arterial + Non-intersection": dark & urban & arterial & nonint,
    }
    for name, m in combos.items():
        cov = df.loc[m, "ped_involved"].sum() / df.ped_involved.sum()   # % of ped deaths captured
        rate = df.loc[m, "ped_involved"].mean()                          # ped share within profile
        rows.append({"profile": name,
                     "share_of_all_fatal_crashes": f"{m.mean()*100:.1f}%",
                     "pct_of_ped_crashes_captured": f"{cov*100:.1f}%",
                     "ped_share_within_profile": f"{rate*100:.1f}%",
                     "lift_vs_baseline": f"{rate/base:.1f}x"})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA, "archetypes.csv"), index=False)
    print(out.to_string(index=False))

# ---------------------------------------------------------------- vehicles
BODY_GROUP = [
    ((1, 13),  "Passenger car"),
    ((14, 19), "SUV / crossover"),
    ((20, 29), "Van / minivan"),
    ((30, 39), "Pickup truck"),
    ((50, 59), "Bus"),
    ((60, 79), "Med/heavy truck"),
    ((80, 89), "Motorcycle"),
]
def body_group(code):
    try: code = float(code)
    except (TypeError, ValueError): return None
    for (lo, hi), name in BODY_GROUP:
        if lo <= code <= hi: return name
    return "Other/Unknown"

def find_file(year, name):
    for d in (f"{RAW}/{year}", f"{RAW}/FARS{year}NationalCSV",
              f"{RAW}/FARS{year}NationalCSV/FARS{year}NationalCSV"):
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower() == f"{name}.csv":
                    return os.path.join(d, f)
    return None

if PART in ("all", "vehicles"):
    yearly = []
    for y in range(2016, 2025):
        pf, vf = find_file(y, "person"), find_file(y, "vehicle")
        if not pf or not vf:
            print(f"{y}: missing person/vehicle file, skipped"); continue
        per = pd.read_csv(pf, encoding="latin-1", low_memory=False,
                          usecols=lambda c: c.upper().replace("﻿","") in
                          ("ST_CASE","PER_TYP","STR_VEH","INJ_SEV"))
        per.columns = [c.upper().replace("﻿","") for c in per.columns]
        veh = pd.read_csv(vf, encoding="latin-1", low_memory=False,
                          usecols=lambda c: c.upper().replace("﻿","") in
                          ("ST_CASE","VEH_NO","BODY_TYP"))
        veh.columns = [c.upper().replace("﻿","") for c in veh.columns]
        peds = per[(pd.to_numeric(per.PER_TYP, errors="coerce") == 5)
                   & (pd.to_numeric(per.INJ_SEV, errors="coerce") == 4)].copy()
        peds["VEH_NO"] = pd.to_numeric(peds.STR_VEH, errors="coerce")
        j = peds.merge(veh, on=["ST_CASE", "VEH_NO"], how="left")
        j["group"] = j.BODY_TYP.map(body_group)
        g = j.group.value_counts(normalize=True).mul(100).round(1)
        g.name = y
        yearly.append(g)
        print(f"{y}: {len(j)} ped fatalities joined to striking vehicle")
    trend = pd.DataFrame(yearly).fillna(0)
    trend.to_csv(os.path.join(DATA, "striking_vehicle_trend.csv"))
    print("\n=== Striking vehicle type, % of pedestrian fatalities ===")
    print(trend.round(1).to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    for colname in ["Passenger car", "SUV / crossover", "Pickup truck"]:
        if colname in trend.columns:
            ax.plot(trend.index, trend[colname], marker="o", linewidth=2.5, label=colname)
    ax.set_ylabel("% of pedestrian fatalities", fontsize=12)
    ax.set_title("Striking vehicle type in pedestrian fatalities (FARS 2016-2024)", fontsize=13)
    ax.legend(fontsize=11); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(DATA, "fig_striking_vehicle.png"), dpi=200)
    print("saved fig_striking_vehicle.png")

# ---------------------------------------------------------------- ages
if PART in ("all", "ages"):
    ped_ages = df.loc[(df.ped_involved == 1) & (df.ped_inj_sev == 4), "ped_age"].dropna()
    print(f"\n=== Pedestrian fatality ages (n={len(ped_ages):,}) ===")
    print(f"median {ped_ages.median():.0f} | 65+: {(ped_ages>=65).mean()*100:.1f}%"
          f" | under 18: {(ped_ages<18).mean()*100:.1f}%")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.hist(ped_ages, bins=range(0, 101, 5), color="#6b46c1", edgecolor="white")
    ax.set_xlabel("Pedestrian age", fontsize=12); ax.set_ylabel("Fatalities", fontsize=12)
    ax.set_title("Age distribution of pedestrian fatalities (2016-2024)", fontsize=13)
    plt.tight_layout(); plt.savefig(os.path.join(DATA, "fig_ped_ages.png"), dpi=200)
    print("saved fig_ped_ages.png")

# ---------------------------------------------------------------- depplot
if PART in ("all", "depplot"):
    import xgboost as xgb
    CATS = ["lighting", "weather", "road_class", "junction"]
    NUMS = ["urban", "work_zone", "hour", "month", "weekend", "speeding_involved",  "drinking_driver", "suv_involved", "pickup_involved", "heavy_truck_involved", "county_median_income", "county_pct_no_vehicle"]
    X = pd.get_dummies(df[CATS + NUMS], columns=CATS, dummy_na=False)
    model = xgb.XGBClassifier(); model.load_model(os.path.join(DATA, "ped_model.json"))
    rng = np.random.RandomState(1)
    idx = rng.choice(len(X), 20000, replace=False)
    dm = xgb.DMatrix(X.iloc[idx], feature_names=list(X.columns))
    sv = model.get_booster().predict(dm, pred_contribs=True)[:, :-1]
    hcol = list(X.columns).index("hour")
    hrs = X.iloc[idx]["hour"].values
    dark = df.iloc[idx].lighting.str.startswith("Dark").fillna(False).values
    fig, ax = plt.subplots(figsize=(9, 5))
    for m, lab, c in [(~dark, "Daylight/dawn/dusk", "#f6ad55"), (dark, "Dark", "#2c5282")]:
        ax.scatter(hrs[m] + rng.uniform(-.3,.3,m.sum()), sv[m, hcol], s=4, alpha=.25, label=lab, color=c)
    ax.axhline(0, color="gray", lw=1)
    ax.set_xlabel("Hour of day", fontsize=12)
    ax.set_ylabel("SHAP value of 'hour'\n(push toward pedestrian involvement)", fontsize=12)
    ax.set_title("When does pedestrian risk concentrate?", fontsize=13)
    ax.legend(fontsize=11, markerscale=4)
    plt.tight_layout(); plt.savefig(os.path.join(DATA, "fig_hour_dependence.png"), dpi=200)
    print("saved fig_hour_dependence.png")

# ---------------------------------------------------------------- counter
if PART in ("all", "counter"):
    cm = pd.DataFrame([
        ["Dark, unlit road", "Continuous roadway lighting; crosswalk-focused lighting", "FHWA Proven Safety Countermeasure: Lighting"],
        ["Dark but lighted road", "Lighting quality/uniformity audit; brighter crossings", "FHWA Lighting; NCHRP 498"],
        ["Late night (12-4 AM)", "Lighting + speed management on night-activity corridors", "FHWA Speed Management"],
        ["Night (8-11 PM)", "Lighting + speed management on night-activity corridors", "FHWA Speed Management"],
        ["Away from intersection (midblock)", "Pedestrian hybrid beacons; median refuge islands; midblock crosswalks", "FHWA PHB; STEP program"],
        ["Intersection-related location", "Leading pedestrian intervals; signal timing; curb extensions", "FHWA LPI"],
        ["Principal arterial road", "Road diets; speed management; crossing opportunities every 200-300 ft", "FHWA Road Diet"],
        ["Minor arterial road", "Road diets; speed management", "FHWA Road Diet"],
        ["Through roadway", "Access management; midblock crossing treatments", "FHWA STEP"],
        ["Highway ramp", "Channelization; pedestrian fencing/wayfinding at interchanges", "AASHTO"],
        ["Clear weather (more walking exposure)", "Exposure indicator - prioritize corridors with high foot traffic", "-"],
    ], columns=["model_factor", "candidate_countermeasures", "reference"])
    cm.to_csv(os.path.join(DATA, "countermeasures.csv"), index=False)
    print("\nsaved countermeasures.csv (factor -> intervention mapping)")

print("\nDone.")

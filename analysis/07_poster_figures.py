"""
07_poster_figures.py -- print-quality figures for the ITE poster.
Style: navy/amber 'night street' palette, large type (legible at 4-5 ft
when printed at 200%), no chartjunk.

Usage: python 07_poster_figures.py <data_dir> <out_dir>
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else DATA
os.makedirs(OUT, exist_ok=True)

NAVY, AMBER, RED, GRAY = "#1a2b4a", "#f6a623", "#c53030", "#8a94a6"
plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 19, "axes.labelsize": 17,
    "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#4a5568", "figure.facecolor": "white",
})

pq, cs = os.path.join(DATA, "crash_level.parquet"), os.path.join(DATA, "crash_level.csv")
df = pd.read_parquet(pq) if os.path.exists(pq) else pd.read_csv(cs, low_memory=False)

# ---- 1. trend: pedestrian share of fatal crashes by year -------------
t = df.groupby("YEAR").ped_involved.agg(["mean", "sum"])
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(t.index, t["mean"]*100, marker="o", lw=3.5, color=AMBER, ms=9,
        markerfacecolor=NAVY, markeredgecolor=AMBER, mew=2)
ax.set_ylabel("% of fatal crashes\ninvolving a pedestrian")
ax.set_ylim(16, 20.5); ax.set_xticks(range(2016, 2025, 2))
ax.grid(axis="y", alpha=0.3)
for x, y in [(2016, t.loc[2016,"mean"]*100), (2024, t.loc[2024,"mean"]*100)]:
    ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=15, fontweight="bold", color=NAVY)
plt.tight_layout(); plt.savefig(f"{OUT}/p_trend.png", dpi=250); plt.close()

# ---- 2. ped share by lighting (the exposure gradient) ----------------
r = df.groupby("lighting").ped_involved.mean().mul(100)
order = ["Daylight", "Dawn", "Dusk", "Dark - Not Lighted", "Dark - Lighted"]
r = r[order]
labels = ["Daylight", "Dawn", "Dusk", "Dark, unlit", "Dark, lighted"]
colors = [GRAY, GRAY, GRAY, NAVY, AMBER]
fig, ax = plt.subplots(figsize=(7.5, 4.4))
bars = ax.barh(labels, r.values, color=colors, height=0.62)
for b, v in zip(bars, r.values):
    ax.text(v + 0.7, b.get_y() + b.get_height()/2, f"{v:.0f}%",
            va="center", fontsize=16, fontweight="bold", color=NAVY)
ax.set_xlabel("% of fatal crashes involving a pedestrian")
ax.set_xlim(0, 40)
plt.tight_layout(); plt.savefig(f"{OUT}/p_lighting.png", dpi=250); plt.close()

# ---- 3. global SHAP importance (environment vs behavior colored) ----
gi = pd.read_csv(os.path.join(DATA, "shap_grouped_importance.csv"), index_col=0).iloc[:, 0]
NAMES = {"lighting":"Lighting","junction":"Junction context","drinking_driver":"Impaired driver",
         "speeding_involved":"Speeding","urban":"Urban vs rural","road_class":"Road class",
         "county_pct_no_vehicle":"County car access","county_median_income":"County income",
         "hour":"Hour of day","heavy_truck_involved":"Heavy truck","suv_involved":"SUV",
         "pickup_involved":"Pickup","month":"Month","weather":"Weather","weekend":"Weekend",
         "work_zone":"Work zone"}
gi.index = [NAMES.get(i, i) for i in gi.index]
gi = gi.head(12)
BEHAV = {"Impaired driver","Speeding","Heavy truck","SUV","Pickup"}
from matplotlib.patches import Patch
fig, ax = plt.subplots(figsize=(7.5, 5.2))
gi.sort_values().plot.barh(ax=ax, color=[AMBER if i in BEHAV else NAVY for i in gi.sort_values().index])
ax.set_xlabel("Mean |SHAP| (influence on prediction)")
ax.legend(handles=[Patch(color=NAVY, label="Environment & place"),
                   Patch(color=AMBER, label="Driver behavior / vehicle")],
          loc="lower right", fontsize=13)
plt.tight_layout(); plt.savefig(f"{OUT}/p_importance.png", dpi=250); plt.close()

# ---- 4. hour dependence (restyled) ------------------------------------
import xgboost as xgb
CATS = ["lighting","weather","road_class","junction"]; NUMS = ["urban","work_zone","hour","month","weekend","speeding_involved","drinking_driver","suv_involved","pickup_involved","heavy_truck_involved", "county_median_income", "county_pct_no_vehicle"]
X = pd.get_dummies(df[CATS+NUMS], columns=CATS, dummy_na=False)
model = xgb.XGBClassifier(); model.load_model(os.path.join(DATA, "ped_model.json"))
rng = np.random.RandomState(1); idx = rng.choice(len(X), 15000, replace=False)
dm = xgb.DMatrix(X.iloc[idx], feature_names=list(X.columns))
sv = model.get_booster().predict(dm, pred_contribs=True)[:, :-1]
h = X.iloc[idx]["hour"].values; hcol = list(X.columns).index("hour")
dark = df.iloc[idx].lighting.str.startswith("Dark").fillna(False).values
fig, ax = plt.subplots(figsize=(8.2, 4.4))
ax.scatter(h[~dark]+rng.uniform(-.35,.35,(~dark).sum()), sv[~dark,hcol], s=6, alpha=.35, color=AMBER, label="Daylight / dawn / dusk")
ax.scatter(h[dark]+rng.uniform(-.35,.35,dark.sum()), sv[dark,hcol], s=6, alpha=.3, color=NAVY, label="Dark")
ax.axhline(0, color="#4a5568", lw=1)
ax.axvspan(16.5, 21.5, color=RED, alpha=0.08)
ax.text(19, ax.get_ylim()[1]*0.88, "5–9 PM\npeak", ha="center", fontsize=15, fontweight="bold", color=RED)
ax.set_xlabel("Hour of day"); ax.set_ylabel("Push toward pedestrian\ninvolvement (SHAP)")
leg = ax.legend(loc="lower right", markerscale=4, framealpha=0.9)
plt.tight_layout(); plt.savefig(f"{OUT}/p_hour.png", dpi=250); plt.close()

# ---- 5. striking vehicle trend ----------------------------------------
tr = pd.read_csv(os.path.join(DATA, "striking_vehicle_trend.csv"), index_col=0)
fig, ax = plt.subplots(figsize=(7.5, 4.4))
style = {"Passenger car": (GRAY, "--"), "SUV / crossover": (AMBER, "-"), "Pickup truck": (NAVY, "-")}
for colname, (c, ls) in style.items():
    ax.plot(tr.index, tr[colname], marker="o", lw=3, ls=ls, color=c, label=colname, ms=6)
ax.annotate(f"{tr['SUV / crossover'].iloc[-1]:.0f}%", (2024, tr['SUV / crossover'].iloc[-1]),
            textcoords="offset points", xytext=(8, 6), fontsize=15, fontweight="bold", color=AMBER)
ax.annotate(f"{tr['Passenger car'].iloc[-1]:.0f}%", (2024, tr['Passenger car'].iloc[-1]),
            textcoords="offset points", xytext=(8, -4), fontsize=15, fontweight="bold", color=GRAY)
ax.set_ylabel("% of pedestrian fatalities"); ax.set_xticks(range(2016, 2025, 2))
ax.grid(axis="y", alpha=0.3); ax.legend(loc="center left")
plt.tight_layout(); plt.savefig(f"{OUT}/p_vehicle.png", dpi=250); plt.close()

print("saved:", ", ".join(sorted(os.listdir(OUT))))


# ---- 6. US dot map: every pedestrian death draws the country ---------
md_path = os.path.join(DATA, "ped_map_data.parquet")
if os.path.exists(md_path):
    d = pd.read_parquet(md_path)
    m = d[(d.LATITUDE.between(24.3, 49.5)) & (d.LONGITUDE.between(-125, -66))]
    fig, ax = plt.subplots(figsize=(13, 8), facecolor="#101c33")
    ax.set_facecolor("#101c33")
    ax.scatter(m.LONGITUDE, m.LATITUDE, s=0.9, c=AMBER, alpha=0.45, linewidths=0)
    ax.set_aspect(1.25); ax.axis("off")
    ax.text(0.012, 0.03, f"Each dot: one pedestrian killed, 2016\u20132024 (FARS)  \u00b7  {len(d):,} deaths",
            transform=ax.transAxes, color="#cbd5e0", fontsize=20)
    plt.tight_layout(pad=0.4)
    plt.savefig(f"{OUT}/p_usmap.png", dpi=250, facecolor="#101c33"); plt.close()
    print("saved p_usmap.png")

# ---- 7. QR code for the deployed app ---------------------------------
APP_URL = "https://ped-fatality-explorergit-5kadirlmyieldrgyjeq7xj.streamlit.app/"
try:
    import qrcode
    q = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=20, border=2)
    q.add_data(APP_URL); q.make(fit=True)
    q.make_image(fill_color="#9E1B32", back_color="white").save(f"{OUT}/qr_app.png")
    print("saved qr_app.png ->", APP_URL)
except ImportError:
    print("pip install qrcode[pil] to generate the QR code")

# ---- 8. feature dictionary (poster/appendix table) --------------------
rows = [
 ["lighting","categorical (6)","ACCIDENT.LGT_COND","Light condition: Daylight, Dark-Lighted, Dark-Not Lighted, Dawn, Dusk, Dark-Unknown"],
 ["weather","categorical (10)","ACCIDENT.WEATHER","Atmospheric conditions: Clear, Rain, Snow, Fog, Cloudy, ..."],
 ["road_class","categorical (7)","ACCIDENT.FUNC_SYS","Functional class: Interstate ... Principal/Minor Arterial ... Local"],
 ["junction","categorical (12)","ACCIDENT.RELJCT2","Relation to junction: Intersection(-related), Non-Junction, Driveway, Ramp, ..."],
 ["urban","binary","ACCIDENT.RUR_URB","Urban (1) vs rural (0) roadway segment"],
 ["work_zone","binary","ACCIDENT.WRK_ZONE","Crash occurred in a work zone"],
 ["hour","numeric 0-23","ACCIDENT.HOUR","Hour of day (99=unknown -> missing)"],
 ["month","numeric 1-12","ACCIDENT.MONTH","Calendar month"],
 ["weekend","binary","ACCIDENT.DAY_WEEK","Saturday or Sunday"],
 ["speeding_involved","binary","VEHICLE.SPEEDREL (any)","Any vehicle coded speeding-related"],
 ["drinking_driver","binary","VEHICLE.DR_DRINK (any)","Any driver coded as drinking"],
 ["suv_involved","binary","VEHICLE.BODY_TYP 14-19","Any SUV/crossover in the crash"],
 ["pickup_involved","binary","VEHICLE.BODY_TYP 30-39","Any pickup truck in the crash"],
 ["heavy_truck_involved","binary","VEHICLE.BODY_TYP 60-79","Any medium/heavy truck in the crash"],
 ["county_median_income","numeric $","ACS 5-yr B19013 via FIPS","Median household income of crash county"],
 ["county_pct_no_vehicle","numeric %","ACS 5-yr B08201 via FIPS","% of county households without a vehicle"],
]
pd.DataFrame(rows, columns=["feature","type","source","description"]) \
  .to_csv(os.path.join(DATA, "feature_dictionary.csv"), index=False)
print("saved feature_dictionary.csv (16 raw features -> 48 one-hot columns)")

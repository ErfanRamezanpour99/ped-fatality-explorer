"""
09_enrich_vehicle_features.py
-----------------------------
Per-crash driver/vehicle behavior features from FARS vehicle files.
Valid for BOTH classes (exist for every fatal crash -> no leakage):

  speeding_involved   any vehicle coded speeding-related (SPEEDREL 2-5)
  hit_run             any vehicle hit-and-run (HIT_RUN 1)
  drinking_driver     any driver coded drinking (DR_DRINK 1)
  suv_involved        any SUV/crossover in crash (BODY_TYP 14-19)
  pickup_involved     any pickup (BODY_TYP 30-39)
  heavy_truck_involved any medium/heavy truck (BODY_TYP 60-79)

Also, for pedestrian-involved crashes, joins the STRIKING vehicle's
body-type group and posted speed limit (VSPD_LIM) for use in narratives:
  striking_veh_type    e.g. "SUV / crossover", "Passenger car"
  speed_limit_mph      posted limit of striking vehicle's trafficway

Merges everything into crash_level.parquet/csv IN PLACE (adds columns).

Usage: python 09_enrich_vehicle_features.py <data_dir> <raw_dir> [year_start year_end]
"""
import sys, os
import numpy as np
import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "."
RAW  = sys.argv[2] if len(sys.argv) > 2 else "raw_fars"
Y0 = int(sys.argv[3]) if len(sys.argv) > 3 else 2016
Y1 = int(sys.argv[4]) if len(sys.argv) > 4 else 2024

def find_file(year, name):
    for d in (f"{RAW}/{year}", f"{RAW}/FARS{year}NationalCSV",
              f"{RAW}/FARS{year}NationalCSV/FARS{year}NationalCSV"):
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower() == f"{name}.csv":
                    return os.path.join(d, f)
    return None

BODY_GROUP = [((1, 13), "passenger car"), ((14, 19), "SUV"),
              ((20, 29), "van"), ((30, 39), "pickup truck"),
              ((50, 59), "bus"), ((60, 79), "heavy truck"),
              ((80, 89), "motorcycle")]
def body_group(code):
    try: code = float(code)
    except (TypeError, ValueError): return None
    for (lo, hi), name in BODY_GROUP:
        if lo <= code <= hi: return name
    return None

frames = []
strike_frames = []
for y in range(Y0, Y1 + 1):
    vf = find_file(y, "vehicle")
    if not vf:
        print(f"{y}: no vehicle file"); continue
    v = pd.read_csv(vf, encoding="latin-1", low_memory=False,
                    usecols=lambda c: c.upper().replace("﻿", "").strip() in
                    ("ST_CASE", "VEH_NO", "SPEEDREL", "HIT_RUN", "DR_DRINK",
                     "BODY_TYP", "VSPD_LIM"))
    v.columns = [c.upper().replace("﻿", "").strip() for c in v.columns]
    for c in v.columns:
        v[c] = pd.to_numeric(v[c], errors="coerce")
    bt = v["BODY_TYP"]
    g = v.assign(
        speeding=v["SPEEDREL"].between(2, 5),
        hitrun=v["HIT_RUN"] == 1,
        drink=v.get("DR_DRINK", pd.Series(np.nan, index=v.index)) == 1,
        suv=bt.between(14, 19), pickup=bt.between(30, 39), truck=bt.between(60, 79),
    ).groupby("ST_CASE")[["speeding", "hitrun", "drink", "suv", "pickup", "truck"]].max()
    g = g.astype(int).reset_index()
    g["CRASH_ID"] = g.ST_CASE.astype(str) + "_" + str(y)
    frames.append(g.drop(columns="ST_CASE"))

    # striking vehicle for pedestrian crashes (via PERSON.STR_VEH)
    pf = find_file(y, "person")
    if pf:
        per = pd.read_csv(pf, encoding="latin-1", low_memory=False,
                          usecols=lambda c: c.upper().replace("﻿", "").strip() in
                          ("ST_CASE", "PER_TYP", "STR_VEH"))
        per.columns = [c.upper().replace("﻿", "").strip() for c in per.columns]
        peds = per[pd.to_numeric(per.PER_TYP, errors="coerce") == 5].copy()
        peds["VEH_NO"] = pd.to_numeric(peds.STR_VEH, errors="coerce")
        peds = peds.dropna(subset=["VEH_NO"]).drop_duplicates("ST_CASE")
        j = peds.merge(v[["ST_CASE", "VEH_NO", "BODY_TYP", "VSPD_LIM"]]
                       if "VSPD_LIM" in v.columns else
                       v[["ST_CASE", "VEH_NO", "BODY_TYP"]],
                       on=["ST_CASE", "VEH_NO"], how="left")
        j["striking_veh_type"] = j.BODY_TYP.map(body_group)
        sl = pd.to_numeric(j.get("VSPD_LIM"), errors="coerce")
        j["speed_limit_mph"] = sl.where((sl > 0) & (sl < 96))
        j["CRASH_ID"] = j.ST_CASE.astype(str) + "_" + str(y)
        strike_frames.append(j[["CRASH_ID", "striking_veh_type", "speed_limit_mph"]])
    print(f"{y}: {len(g)} crashes")

feat = pd.concat(frames, ignore_index=True).rename(columns={
    "speeding": "speeding_involved", "hitrun": "hit_run", "drink": "drinking_driver",
    "suv": "suv_involved", "pickup": "pickup_involved", "truck": "heavy_truck_involved"})
out = os.path.join(DATA, f"vehicle_features_{Y0}_{Y1}.parquet")
feat.to_parquet(out, index=False)
print("saved", out, feat.shape)

# ---- merge flags into crash_level IN PLACE (idempotent) ----------------
pq = os.path.join(DATA, "crash_level.parquet")
cs = os.path.join(DATA, "crash_level.csv")
cl = pd.read_parquet(pq) if os.path.exists(pq) else pd.read_csv(cs, low_memory=False)
flag_cols = [c for c in feat.columns if c != "CRASH_ID"]
cl = cl.drop(columns=[c for c in flag_cols if c in cl.columns])
cl = cl.merge(feat, on="CRASH_ID", how="left")
for c in flag_cols:
    cl[c] = cl[c].fillna(0).astype(int)
if strike_frames:
    strike = pd.concat(strike_frames, ignore_index=True).drop_duplicates("CRASH_ID")
    cl = cl.drop(columns=[c for c in ["striking_veh_type", "speed_limit_mph"]
                          if c in cl.columns])
    cl = cl.merge(strike, on="CRASH_ID", how="left")
    print(f"striking-vehicle info: {cl.striking_veh_type.notna().sum():,} crashes | "
          f"speed limit known: {cl.speed_limit_mph.notna().sum():,}")
if os.path.exists(pq):
    cl.to_parquet(pq, index=False)
cl.to_csv(cs, index=False)
print(f"merged {len(flag_cols)} behavior flags into crash_level ({cl.shape})")
print("\n=== ped share by flag (sanity check) ===")
for c in flag_cols:
    r = cl.groupby(c).ped_involved.mean().mul(100).round(1)
    print(f"{c:22s} 0: {r.get(0)}%   1: {r.get(1)}%")

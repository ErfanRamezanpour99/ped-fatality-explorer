"""
10_acs_join.py
--------------
Join county-level ACS 2023 5-year context onto the crash-level dataset:

  county_median_income   B19013_001E  (median household income, $)
  county_pct_no_vehicle  B08201_002E / B08201_001E * 100  (% households w/o car)

County key: FARS STATE (2-digit) + COUNTY (3-digit) FIPS.
Merges IN PLACE into crash_level.parquet/csv.

Usage: python 10_acs_join.py <data_dir> <acs_counties.json>
"""
import sys, os, json
import numpy as np
import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "."
ACS = sys.argv[2] if len(sys.argv) > 2 else "acs_counties.json"

raw = json.load(open(ACS))
acs = pd.DataFrame(raw[1:], columns=raw[0])
for c in ["B19013_001E", "B08201_001E", "B08201_002E", "B01003_001E"]:
    acs[c] = pd.to_numeric(acs[c], errors="coerce")
acs.loc[acs.B19013_001E < 0, "B19013_001E"] = np.nan   # census sentinel -666666666
acs["fips"] = acs.state + acs.county
acs["county_median_income"] = acs.B19013_001E
acs["county_pct_no_vehicle"] = (acs.B08201_002E / acs.B08201_001E * 100).round(2)
acs = acs[["fips", "county_median_income", "county_pct_no_vehicle"]]

pq, cs = os.path.join(DATA, "crash_level.parquet"), os.path.join(DATA, "crash_level.csv")
df = pd.read_parquet(pq) if os.path.exists(pq) else pd.read_csv(cs, low_memory=False)
df = df.drop(columns=[c for c in ["county_median_income", "county_pct_no_vehicle"] if c in df.columns])
df["fips"] = (df.STATE.astype(int).astype(str).str.zfill(2)
              + df.COUNTY.astype(int).astype(str).str.zfill(3))
df = df.merge(acs, on="fips", how="left").drop(columns="fips")
df.to_parquet(pq, index=False)
print(f"joined: {df.county_median_income.notna().mean()*100:.1f}% of crashes matched a county")

print("\n=== ped share by county income quintile ===")
q = pd.qcut(df.county_median_income, 5, labels=["poorest 20%","Q2","Q3","Q4","richest 20%"])
print(df.groupby(q, observed=True).ped_involved.mean().mul(100).round(1).to_string())
print("\n=== ped share by county %-no-vehicle quintile ===")
q2 = pd.qcut(df.county_pct_no_vehicle, 5, labels=["fewest carless","Q2","Q3","Q4","most carless"])
print(df.groupby(q2, observed=True).ped_involved.mean().mul(100).round(1).to_string())

"""
01_build_dataset.py
-------------------
Build a CRASH-LEVEL dataset from raw FARS National CSV files (2016-2024)
for the ITE poster project:

    Target:  ped_involved = 1 if the fatal crash involved >=1 pedestrian
             (PERSON.PER_TYP == 5), else 0.

    Features: PRE-CRASH CONTEXT ONLY (no leakage):
        - LGT_COND   -> lighting category
        - WEATHER    -> weather category
        - RUR_URB    -> urban flag
        - FUNC_SYS   -> road functional class
        - RELJCT2    -> junction context
        - REL_ROAD   -> location relative to trafficway (flagged: review at checkpoint)
        - WRK_ZONE   -> work-zone flag
        - HOUR       -> numeric hour (99 -> NaN)
        - DAY_WEEK   -> weekend flag
        - MONTH      -> numeric month

    Explicitly EXCLUDED (leakage / trivially determined by outcome):
        HARM_EV, MAN_COLL, PEDS, PERNOTMVIT, VE_TOTAL, VE_FORMS,
        PERSONS, FATALS, notification/arrival/hospital times.

    Also carried through (NOT model features - for mapping, popups,
    narratives, descriptive stats):
        LATITUDE, LONGITUD, STATE, STATENAME, COUNTY, CITY, YEAR,
        FATALS, and for pedestrian crashes: pedestrian age/sex/injury,
        LOCATION (person file), PBTYPE crossing details (PEDCTYPE,
        PEDLOC, PEDPOS, MOTMAN).

Usage:  python 01_build_dataset.py <raw_dir> <out_dir>
        raw_dir contains subfolders 2016..2024 each with accident.csv,
        person.csv, pbtype.csv
"""

import sys
import numpy as np
import pandas as pd

RAW = sys.argv[1] if len(sys.argv) > 1 else "fars_local"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
YEARS = list(range(2016, 2025))

# ----------------------------------------------------------------------
# Recode maps (FARS Analytical User's Manual codes, consistent 2016+)
# ----------------------------------------------------------------------
LGT_MAP = {
    1: "Daylight", 2: "Dark - Not Lighted", 3: "Dark - Lighted",
    4: "Dawn", 5: "Dusk", 6: "Dark - Unknown Lighting",
}
WEATHER_MAP = {
    1: "Clear", 2: "Rain", 3: "Sleet/Hail", 4: "Snow", 5: "Fog/Smog",
    6: "Severe Crosswinds", 7: "Blowing Sand/Dirt", 10: "Cloudy",
    11: "Blowing Snow", 12: "Freezing Rain",
}
FUNC_MAP = {
    1: "Interstate", 2: "Freeway/Expressway", 3: "Principal Arterial",
    4: "Minor Arterial", 5: "Major Collector", 6: "Minor Collector",
    7: "Local Road",
}
RELJCT2_MAP = {
    1: "Non-Junction", 2: "Intersection", 3: "Intersection-Related",
    4: "Driveway Access", 5: "Entrance/Exit Ramp Related",
    6: "Railway Crossing", 7: "Crossover-Related",
    8: "Driveway Access Related", 16: "Shared-Use Path Crossing",
    17: "Acceleration/Deceleration Lane", 18: "Through Roadway",
    19: "Other Location in Interchange", 20: "Entrance/Exit Ramp",
}
REL_ROAD_MAP = {
    1: "On Roadway", 2: "On Shoulder", 3: "On Median", 4: "On Roadside",
    5: "Outside Trafficway", 6: "Off Roadway - Unknown", 7: "In Parking Lane",
    8: "Gore", 10: "Separator", 11: "Continuous Left-Turn Lane",
}
# PBTYPE descriptive maps (for popups/narratives on ped crashes)
PEDLOC_MAP = {
    1: "At Intersection", 2: "Intersection-Related", 3: "Not At Intersection (Midblock)",
    4: "Non-Trafficway Location",
}
MOTMAN_MAP = {
    1: "Going Straight", 2: "Turning Left", 3: "Turning Right",
    4: "Backing", 5: "Other Maneuver",
}


def clean_codes(s, valid_max=97):
    """Set FARS unknown/not-reported codes to NaN (98, 99, 998, 999...)."""
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s < valid_max, np.nan)


def find_file(year, name):
    """Locate <name>.csv for a year under RAW, handling layouts:
       RAW/<year>/, RAW/FARS<year>NationalCSV/, and the double-nested
       RAW/FARS<year>NationalCSV/FARS<year>NationalCSV/. Case-insensitive."""
    import glob, os
    candidates = [f"{RAW}/{year}", f"{RAW}/FARS{year}NationalCSV",
                  f"{RAW}/FARS{year}NationalCSV/FARS{year}NationalCSV"]
    for d in candidates:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower() == f"{name}.csv":
                return os.path.join(d, f)
    return None


def load_year(year):
    acc = pd.read_csv(find_file(year, "accident"), encoding="latin-1", low_memory=False)
    per = pd.read_csv(find_file(year, "person"), encoding="latin-1", low_memory=False)
    pb_path = find_file(year, "pbtype")
    pb = pd.read_csv(pb_path, encoding="latin-1", low_memory=False) if pb_path else None
    def _clean_cols(df):
        df.columns = (df.columns.str.replace("﻿", "", regex=False)
                      .str.replace("ï»¿", "", regex=False).str.strip().str.upper())
        return df

    acc, per = _clean_cols(acc), _clean_cols(per)
    if pb is not None:
        pb = _clean_cols(pb)

    # ---------------- target: any pedestrian (PER_TYP==5) ----------------
    per["is_ped"] = (pd.to_numeric(per["PER_TYP"], errors="coerce") == 5).astype(int)
    ped_flag = per.groupby("ST_CASE")["is_ped"].max().rename("ped_involved")

    # ---------------- pedestrian details (first/most-severe ped) ---------
    peds = per[per["is_ped"] == 1].copy()
    if len(peds):
        peds["INJ_SEV_n"] = pd.to_numeric(peds["INJ_SEV"], errors="coerce")
        peds = peds.sort_values("INJ_SEV_n", ascending=False).drop_duplicates("ST_CASE")
        ped_cols = {"ST_CASE": "ST_CASE"}
        for c_src, c_dst in [("AGE", "ped_age"), ("SEX", "ped_sex"),
                             ("INJ_SEV_n", "ped_inj_sev"), ("LOCATION", "ped_location"),
                             ("VEH_NO", "VEH_NO"), ("PER_NO", "PER_NO")]:
            if c_src in peds.columns:
                ped_cols[c_src] = c_dst
        ped_info = peds[list(ped_cols)].rename(columns=ped_cols)
        ped_info["ped_age"] = clean_codes(ped_info.get("ped_age"), 150)
        ped_info["ped_sex"] = pd.to_numeric(ped_info.get("ped_sex"), errors="coerce") \
            .map({1: "Male", 2: "Female"})
    else:
        ped_info = pd.DataFrame(columns=["ST_CASE"])

    # ---------------- PBTYPE details for that pedestrian -----------------
    if pb is not None and len(peds):
        pbp = pb[pd.to_numeric(pb["PBPTYPE"], errors="coerce") == 5].copy()
        keep = ["ST_CASE", "VEH_NO", "PER_NO"]
        for c in ["PEDLOC", "MOTMAN", "PEDCTYPE", "PEDPOS"]:
            if c in pbp.columns:
                keep.append(c)
        pbp = pbp[keep].drop_duplicates(["ST_CASE", "VEH_NO", "PER_NO"])
        ped_info = ped_info.merge(pbp, on=["ST_CASE", "VEH_NO", "PER_NO"], how="left")

    # ---------------- crash-level features --------------------------------
    a = pd.DataFrame()
    a["ST_CASE"] = acc["ST_CASE"]
    a["YEAR"] = year
    a["STATE"] = acc["STATE"]
    a["STATENAME"] = acc.get("STATENAME")
    a["COUNTY"] = acc.get("COUNTY")
    a["CITY"] = acc.get("CITY")
    a["CITYNAME"] = acc.get("CITYNAME")
    a["LATITUDE"] = pd.to_numeric(acc["LATITUDE"], errors="coerce")
    a["LONGITUD"] = pd.to_numeric(acc["LONGITUD"], errors="coerce")
    a["FATALS"] = pd.to_numeric(acc["FATALS"], errors="coerce")

    a["lighting"] = clean_codes(acc["LGT_COND"]).map(LGT_MAP)
    a["weather"] = clean_codes(acc["WEATHER"]).map(WEATHER_MAP)
    a["road_class"] = clean_codes(acc["FUNC_SYS"]).map(FUNC_MAP)
    a["junction"] = clean_codes(acc["RELJCT2"]).map(RELJCT2_MAP)
    a["rel_road"] = clean_codes(acc["REL_ROAD"]).map(REL_ROAD_MAP)
    a["urban"] = clean_codes(acc["RUR_URB"], 3).map({1: 0, 2: 1})
    a["work_zone"] = (pd.to_numeric(acc["WRK_ZONE"], errors="coerce")
                      .fillna(0) > 0).astype(int)
    a["hour"] = clean_codes(acc["HOUR"], 24)
    a["month"] = clean_codes(acc["MONTH"], 13)
    dw = clean_codes(acc["DAY_WEEK"], 8)
    a["weekend"] = dw.isin([1, 7]).astype(int)          # FARS: 1=Sunday, 7=Saturday
    a.loc[dw.isna(), "weekend"] = np.nan

    # merge target + ped details
    a = a.merge(ped_flag, on="ST_CASE", how="left")
    a["ped_involved"] = a["ped_involved"].fillna(0).astype(int)
    a = a.merge(ped_info.drop(columns=["VEH_NO", "PER_NO"], errors="ignore"),
                on="ST_CASE", how="left")

    # readable PBTYPE labels
    if "PEDLOC" in a.columns:
        a["ped_loc_type"] = clean_codes(a["PEDLOC"], 8).map(PEDLOC_MAP)
    if "MOTMAN" in a.columns:
        a["motorist_maneuver"] = clean_codes(a["MOTMAN"], 8).map(MOTMAN_MAP)

    # unique crash id across years
    a["CRASH_ID"] = a["ST_CASE"].astype(str) + "_" + str(year)
    return a


if __name__ == "__main__":
    frames = []
    for y in YEARS:
        try:
            df = load_year(y)
            frames.append(df)
            print(f"{y}: {len(df):>6} crashes | ped_involved: "
                  f"{df.ped_involved.sum():>5} ({df.ped_involved.mean()*100:.1f}%)")
        except (FileNotFoundError, ValueError, TypeError) as e:
            print(f"{y}: MISSING ({e})")

    full = pd.concat(frames, ignore_index=True)
    print(f"\nTOTAL: {len(full)} fatal crashes, "
          f"{full.ped_involved.sum()} pedestrian-involved "
          f"({full.ped_involved.mean()*100:.1f}%)")

    import os
    os.makedirs(OUT, exist_ok=True)
    full.to_csv(f"{OUT}/crash_level.csv", index=False)
    try:
        full.to_parquet(f"{OUT}/crash_level.parquet", index=False)
        print(f"Saved -> {OUT}/crash_level.csv and .parquet")
    except ImportError:
        print(f"Saved -> {OUT}/crash_level.csv (install pyarrow for parquet)")

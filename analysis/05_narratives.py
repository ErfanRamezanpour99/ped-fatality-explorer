"""
05_narratives.py
----------------
Generate a short plain-language narrative for every pedestrian-involved
fatal crash, combining:
  - who (pedestrian age/sex, from PERSON)
  - where/when (PBTYPE crossing location, lighting, year)
  - why the model flags it (top SHAP factors, via ped_top_factors.csv)
  - what could help (countermeasure vocabulary)

The wording bank lives in narrative_phrases.json (LLM-authored, editable).
New model factors without a phrase fall back to a generic sentence, so
the pipeline never breaks when the model changes.

Outputs: narratives.csv  (CRASH_ID, narrative)

Usage: python 05_narratives.py <data_dir> [phrases_json]
Requires: pandas numpy
"""

import sys, os, json
import numpy as np
import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "."
PHRASES = sys.argv[2] if len(sys.argv) > 2 else \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "narrative_phrases.json")

bank = json.load(open(PHRASES, encoding="utf-8"))
FP, CP = bank["factor_phrases"], bank["countermeasure_phrases"]

pq, cs = os.path.join(DATA, "crash_level.parquet"), os.path.join(DATA, "crash_level.csv")
df = pd.read_parquet(pq) if os.path.exists(pq) else pd.read_csv(cs, low_memory=False)
tf = pd.read_csv(os.path.join(DATA, "ped_top_factors.csv"))

d = df[df.ped_involved == 1].merge(tf, on="CRASH_ID", how="left")

def hour_window(h):
    if pd.isna(h): return "an unknown time"
    h = int(h)
    if h < 5:  return "the late-night hours"
    if h < 10: return "the morning"
    if h < 16: return "midday"
    if h < 20: return "the evening"
    return "the night"

def who(r):
    age = "" if pd.isna(r.ped_age) else f"{int(r.ped_age)}-year-old "
    sex = {"Male": "man", "Female": "woman"}.get(r.ped_sex, "pedestrian")
    noun = sex if sex != "pedestrian" else "pedestrian"
    fatal = "was killed" if r.ped_inj_sev == 4 else "was struck"
    veh = getattr(r, "striking_veh_type", None)
    by = (f" by {'an' if veh == 'SUV' else 'a'} {veh}") if isinstance(veh, str) else ""
    sl = getattr(r, "speed_limit_mph", None)
    road = f" on a {int(sl)}-mph road" if pd.notna(sl) else ""
    return f"A {age}{noun} on foot {fatal}{by}{road} here in {int(r.YEAR)}"

def where(r):
    loc = r.ped_loc_type if isinstance(r.ped_loc_type, str) else None
    if loc == "Not At Intersection (Midblock)": return ", crossing midblock"
    if loc == "At Intersection": return ", at an intersection"
    if loc == "Intersection-Related": return ", near an intersection"
    return ""

def factor_sentence(r):
    loc = r.ped_loc_type if isinstance(r.ped_loc_type, str) else ""
    skip = set()
    if "Midblock" in loc:
        # PBTYPE says midblock: suppress contradicting intersection phrases
        # and redundant midblock phrasing (already stated in where())
        skip |= {"junction_Intersection", "junction_Intersection-Related",
                 "junction_Non-Junction"}
    elif "Intersection" in loc:
        skip |= {"junction_Non-Junction", "junction_Through Roadway"}
    phrases = []
    for f in [r.top1, r.top2]:
        if not isinstance(f, str) or f in ("urban", "month", "weekend", "county_median_income", "county_pct_no_vehicle") or f in skip:
            continue
        p = FP.get(f)
        if p is None: p = FP["_default"]
        if "{time_window}" in p: p = p.format(time_window=hour_window(r.hour))
        if p not in phrases: phrases.append(p)
        if len(phrases) == 2: break
    if not phrases: return ""
    joined = phrases[0] if len(phrases) == 1 else f"{phrases[0]}, and {phrases[1]}"
    return f" What set this crash apart: {joined}."

def fix_sentence(r):
    loc = r.ped_loc_type if isinstance(r.ped_loc_type, str) else ""
    skip = set()
    if "Midblock" in loc:
        skip |= {"junction_Intersection", "junction_Intersection-Related"}
    elif "Intersection" in loc:
        skip |= {"junction_Non-Junction", "junction_Through Roadway"}
    for f in [r.top1, r.top2, r.top3]:
        if isinstance(f, str) and f in CP and f not in skip:
            return f" Candidate fix: {CP[f]}."
    if "Midblock" in loc:   # sensible default for midblock deaths
        return " Candidate fix: midblock crossings or pedestrian hybrid beacons."
    return ""

def hitrun_sentence(r):
    return " The driver fled the scene." if getattr(r, "hit_run", 0) == 1 else ""

def narrate(r):
    return who(r) + where(r) + "." + hitrun_sentence(r) + factor_sentence(r) + fix_sentence(r)

d["narrative"] = d.apply(narrate, axis=1)
out = d[["CRASH_ID", "narrative"]]
out.to_csv(os.path.join(DATA, "narratives.csv"), index=False)
print(f"Wrote {len(out):,} narratives -> narratives.csv\n")
print("=== 6 random samples ===")
for s in out.sample(6, random_state=7).narrative:
    print("-", s)

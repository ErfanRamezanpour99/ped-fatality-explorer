"""
08_build_static_site.py
-----------------------
Build the static (HTML + Leaflet) Pedestrian Fatality Explorer for a
free Hugging Face **Static** Space (no server, never sleeps).

Emits into <site_dir>:
  data/states.json     [{s: "Alabama", n: 1052, lat, lon}, ...]
  data/<State>.json    per state: {years:[...], recs:[[lat,lon,year,colorIdx,label,narrative],...]}
  data/us_points.json  all points as [[lat,lon],...] for the clustered US view
(index.html is written separately.)

Usage: python 08_build_static_site.py <data_dir> <site_dir>
"""
import sys, os, json, re
import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "."
SITE = sys.argv[2] if len(sys.argv) > 2 else "ped_app_static"
os.makedirs(os.path.join(SITE, "data"), exist_ok=True)

df = pd.read_parquet(os.path.join(DATA, "ped_map_data.parquet"))
COLORS = ["#c53030", "#dd6b20", "#6b46c1", "#2c5282", "#2f855a", "#718096"]
cmap = {c: i for i, c in enumerate(COLORS)}
df["cidx"] = df.factor_color.map(cmap).fillna(5).astype(int)
df["label"] = df.factor_label.fillna("Other")
df["narrative"] = df.narrative.fillna("").str.replace(r"\s+", " ", regex=True)

states = []
for st, g in df.groupby("STATENAME"):
    fn = re.sub(r"[^A-Za-z]", "_", st)
    recs = [[round(r.LATITUDE, 5), round(r.LONGITUDE, 5), int(r.YEAR), int(r.cidx),
             r.label, r.narrative] for r in g.itertuples()]
    json.dump({"recs": recs}, open(f"{SITE}/data/{fn}.json", "w"), separators=(",", ":"))
    states.append({"s": st, "f": fn, "n": len(g),
                   "lat": round(g.LATITUDE.median(), 3), "lon": round(g.LONGITUDE.median(), 3)})
states.sort(key=lambda x: x["s"])
json.dump(states, open(f"{SITE}/data/states.json", "w"), separators=(",", ":"))

pts = [[round(la, 4), round(lo, 4)] for la, lo in zip(df.LATITUDE, df.LONGITUDE)]
json.dump(pts, open(f"{SITE}/data/us_points.json", "w"), separators=(",", ":"))

import glob
sizes = {os.path.basename(p): os.path.getsize(p)//1024 for p in glob.glob(f"{SITE}/data/*.json")}
big = sorted(sizes.items(), key=lambda x: -x[1])[:5]
print(f"{len(states)} states | biggest files (KB): {big}")
print("total data:", sum(sizes.values())//1024, "MB")

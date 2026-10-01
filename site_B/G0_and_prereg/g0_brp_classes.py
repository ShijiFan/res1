"""Site-2 G0: BRP 2024 crop-class composition per candidate AOI (and Twente for calibration).

Uses the national BRP 2024 GeoPackage already on disk and the A1 code mapping
(runs/20260923_0140_G0R/qc/crop_code_mapping_2024_2025.csv) with the strict season protocol
(ambiguous cereal codes -> Other), as in A1. Eligibility proxy: area after a 20*sqrt(2) m inward
buffer >= 2 x 1600 m^2 (A1's exact rule counts >= 2 pure 40 m cells; the proxy is calibrated on Twente).
Out: brp2024_class_counts.csv
"""
import csv
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

SAR = Path(__file__).resolve().parents[1]
GPKG = SAR / "sar_v2" / "labels" / "nl_brp" / "brpgewaspercelen_definitief_2024.gpkg"
MAP = SAR / "A1_observation_budget_20260923" / "runs" / "20260923_0140_G0R" / "qc" / "crop_code_mapping_2024_2025.csv"
AMBIG = {238, 314, 382, 670, 2652, 6636, 7130}
SITES = {"twente": (6.00, 51.98, 6.54, 52.38), "flevoland": (5.40, 52.35, 5.95, 52.70),
         "groningen_drenthe": (6.60, 52.75, 7.15, 53.10), "brabant_east": (5.45, 51.35, 6.00, 51.70)}

rule = {int(r["gewascode"]): r["assigned_8class"] for r in csv.DictReader(open(MAP, encoding="utf-8"))}
rows = []
for site, bb in SITES.items():
    aoi = gpd.GeoSeries([box(*bb)], crs=4326).to_crs(28992)
    g = gpd.read_file(GPKG, bbox=tuple(aoi.total_bounds))
    g = g[g.intersects(aoi.iloc[0])]
    code = pd.to_numeric(g["gewascode"], errors="coerce").fillna(-1).astype(int)
    cls = code.map(lambda c: "Other" if c in AMBIG else rule.get(c, "Other"))
    elig = g.geometry.buffer(-20 * 2 ** 0.5).area >= 3200
    for c in sorted(cls.unique()):
        m = cls == c
        rows.append({"site": site, "class": c, "parcels": int(m.sum()), "eligible_proxy": int((m & elig).sum()),
                     "area_km2": round(float(g.geometry[m].area.sum() / 1e6), 1)})
    print(site, len(g), "parcels")
pd.DataFrame(rows).to_csv("brp2024_class_counts.csv", index=False)

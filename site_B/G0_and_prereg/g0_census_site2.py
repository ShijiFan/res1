"""Site-2 G0: Sentinel-1 IW SLC census per relative orbit for candidate regions (public ASF search).

Derived from A1_observation_budget_20260923/scripts/g0_asf_census.py (same logic, new candidates,
compact output). Counts, per orbit, the dates with >= 95% AOI coverage in Mar-Oct, their platforms,
and the number of disjoint 12-day S1A/S1A pairs that can be formed greedily in time order.
Out: census_<site>_<year>.json
"""
import json
import sys
from collections import defaultdict
from datetime import date

import asf_search as asf
from shapely.geometry import box, shape

SITES = {  # lon_min, lat_min, lon_max, lat_max; about 35 x 40 km each
    "flevoland": (5.40, 52.35, 5.95, 52.70),
    "groningen_drenthe": (6.60, 52.75, 7.15, 53.10),
    "brabant_east": (5.45, 51.35, 6.00, 51.70),
}


def census(site, year):
    aoi = box(*SITES[site])
    res = asf.search(platform=[asf.PLATFORM.SENTINEL1], processingLevel=asf.PRODUCT_TYPE.SLC,
                     beamMode=asf.BEAMMODE.IW, intersectsWith=aoi.wkt,
                     start=f"{year}-03-01T00:00:00Z", end=f"{year}-10-31T23:59:59Z")
    per = defaultdict(lambda: defaultdict(lambda: {"cov": 0.0, "plat": set(), "scenes": []}))
    for r in res:
        p = r.properties
        d = per[int(p["pathNumber"])][p["startTime"][:10]]
        d["cov"] = min(1.0, d["cov"] + aoi.intersection(shape(r.geometry)).area / aoi.area)
        d["plat"].add(p.get("platform", "?"))
        d["scenes"].append(p["sceneName"])
    out = {}
    for orb, days in sorted(per.items()):
        full = {k: v for k, v in days.items() if v["cov"] >= 0.95}
        s1a = sorted(k for k, v in full.items() if "Sentinel-1A" in v["plat"])
        ds = [date.fromisoformat(k) for k in s1a]
        pairs, i = [], 0
        while i < len(ds) - 1:  # greedy disjoint 12-day S1A pairs
            j = next((j for j in range(i + 1, len(ds)) if (ds[j] - ds[i]).days == 12), None)
            if j is None:
                i += 1
                continue
            pairs.append(f"{ds[i]}_{ds[j]}")
            i = j + 1
        out[orb] = {"dates_any_cov": len(days), "dates_full_cov": len(full),
                    "platforms": sorted({p for v in full.values() for p in v["plat"]}),
                    "s1a_full_dates": len(s1a), "disjoint_12d_s1a_pairs": len(pairs), "pairs": pairs,
                    "mean_cov_all_dates": round(sum(v["cov"] for v in days.values()) / len(days), 3)}
    return out


if __name__ == "__main__":
    for site in sys.argv[1:] or SITES:
        for year in (2024, 2025):
            o = census(site, year)
            json.dump(o, open(f"census_{site}_{year}.json", "w"), indent=1)
            print(site, year, {k: (v["dates_full_cov"], v["disjoint_12d_s1a_pairs"], v["mean_cov_all_dates"]) for k, v in o.items()})

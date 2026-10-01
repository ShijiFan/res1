"""A1 G0: count Sentinel-1 IW SLC acquisitions per relative orbit over an AOI.

Usage:
    pip install asf_search shapely
    python g0_asf_census.py --year 2025                 # Twente AOI
    python g0_asf_census.py --year 2024
    python g0_asf_census.py --year 2025 --site flevoland

Outputs g0_census_<site>_<year>.json next to this script:
per relative orbit -> dates, platform counts (S1A/S1C/...), min/median gap in days,
AOI coverage fraction per date, and the list of 6-day pairs (if any).
No credentials needed (ASF search is public).
"""
import argparse, json, statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import asf_search as asf
from shapely.geometry import box, shape

SITES = {
    "twente": (6.00, 51.98, 6.54, 52.38),
    "flevoland": (5.40, 52.40, 5.90, 52.70),  # candidate, adjust in G0
}
TRACKS = {"twente": [15, 37, 88, 139], "flevoland": None}  # None = report all orbits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--site", default="twente", choices=SITES)
    ap.add_argument("--start", default="03-01")
    ap.add_argument("--end", default="10-31")
    a = ap.parse_args()

    aoi = box(*SITES[a.site])
    kw = dict(
        platform=[asf.PLATFORM.SENTINEL1],
        processingLevel=asf.PRODUCT_TYPE.SLC,
        beamMode=asf.BEAMMODE.IW,
        intersectsWith=aoi.wkt,
        start=f"{a.year}-{a.start}T00:00:00Z",
        end=f"{a.year}-{a.end}T23:59:59Z",
    )
    if TRACKS[a.site]:
        kw["relativeOrbit"] = TRACKS[a.site]
    res = asf.search(**kw)

    per = defaultdict(lambda: defaultdict(lambda: {"cov": 0.0, "platforms": set(), "scenes": []}))
    meta = {}
    for r in res:
        p = r.properties
        orb = int(p["pathNumber"])
        day = p["startTime"][:10]
        cov = aoi.intersection(shape(r.geometry)).area / aoi.area
        d = per[orb][day]
        d["cov"] = min(1.0, d["cov"] + cov)  # frames on the same pass add up
        d["platforms"].add(p.get("platform", "?"))
        d["scenes"].append(p["sceneName"])
        meta[orb] = p.get("flightDirection", "?")

    out = {"site": a.site, "year": a.year, "window": [a.start, a.end], "aoi": SITES[a.site], "orbits": {}}
    for orb in sorted(per):
        days = sorted(per[orb])
        full = [d for d in days if per[orb][d]["cov"] >= 0.95]
        dts = [datetime.fromisoformat(x) for x in full]
        gaps = [(b - c).days for c, b in zip(dts, dts[1:])]
        plat = defaultdict(int)
        for d in full:
            for pl in per[orb][d]["platforms"]:
                plat[pl] += 1
        six = [(full[i], full[i + 1]) for i, g in enumerate(gaps) if g == 6]
        out["orbits"][orb] = {
            "direction": meta[orb],
            "n_dates_any": len(days),
            "n_dates_cov95": len(full),
            "platform_counts": dict(plat),
            "min_gap_days": min(gaps) if gaps else None,
            "median_gap_days": statistics.median(gaps) if gaps else None,
            "n_6day_pairs": len(six),
            "dates_cov95": full,
            "coverage_by_date": {d: round(per[orb][d]["cov"], 3) for d in days},
        }
        print(f"orbit {orb:>3} {meta[orb]:<10} dates(cov>=95%)={len(full):>3} "
              f"platforms={dict(plat)} min_gap={out['orbits'][orb]['min_gap_days']} 6d_pairs={len(six)}")

    f = Path(__file__).with_name(f"g0_census_{a.site}_{a.year}.json")
    f.write_text(json.dumps(out, indent=1))
    print("written", f)


if __name__ == "__main__":
    main()

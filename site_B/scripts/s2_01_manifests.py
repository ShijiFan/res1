"""Site 2 (Groningen-Drenthe) scene/pair manifests, budget design, Study-1 pairs, HyP3 job list.

Re-implements, for a new AOI and without touching site-1 folders, the rules of
A1_observation_budget_20260923/scripts/repair_g0r_catalog_20260923.py (L98-254):
  * per (track, date) the single SLC frame with the largest AOI coverage;
  * 12-day S1A/S1A pairs whose two frames overlap >= 0.85 of the AOI;
  * disjoint ladder in date order (>= 9 pairs required), uniform nested design with first/last
    pair as M=2 anchors and repeated largest-gap midpoint insertion, M=1 = pair nearest 30 June;
  * 2024 matched to the 2025 slots by day-of-year within 14 days (non-overlapping DP).
Site-specific additions (PREREG_SITE2.md): T88 prefers the site-1 SLC frame when it covers
>= 0.85 of the site-2 AOI, so already-downloaded site-1 T88 products are reused; Study-1 pairs
use the site-1 dates (spring and autumn 2024) on T15/T37/T88/T139.
Out: ../manifests/{scene_manifest_{y}.csv, pair_manifest_{y}.csv, budget_design.csv,
     study1_pairs.csv, job_list.csv, reuse_map.csv, MANIFEST_REPORT.json}
"""
import csv
import json
from datetime import date
from functools import lru_cache
from pathlib import Path

import asf_search as asf
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.ops import transform

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manifests"
SAR = ROOT.parent
S1_MAN = SAR / "A1_observation_budget_20260923" / "runs" / "20260923_G0R_CATALOG_REPAIR" / "manifests"
S1_DL = SAR / "A1_observation_budget_20260923" / "runs" / "20260924_A1_MAIN" / "downloads"
S1_P0 = SAR / "A1_observation_budget_20260923" / "runs" / "20260923_P0_PILOT" / "downloads"  # pilot batch (2025 June pairs)
AOI = box(6.55, 52.75, 7.05, 53.10)
TO_M = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True).transform
AOI_M = transform(TO_M, AOI)
M_VALUES = (1, 2, 3, 4, 6, 8, 9)
DOY_TOL = 14
MIN_COV = 0.85
STUDY1 = {  # site-1 Study-1 dates (A0 Table I and tgrs_revision autumn); (date_1, date_2)
    "spring": {15: ("2024-05-10", "2024-05-22"), 37: ("2024-05-12", "2024-05-24"),
               88: ("2024-05-15", "2024-05-27"), 139: ("2024-05-07", "2024-05-19")},
    "autumn": {15: ("2024-10-01", "2024-10-13"), 37: ("2024-10-03", "2024-10-15"),
               88: ("2024-10-06", "2024-10-18"), 139: ("2024-10-10", "2024-10-22")},
}


def wcsv(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def rcsv(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def site1_t88_scenes():
    ids = set()
    for y in (2024, 2025):
        for r in rcsv(S1_MAN / f"scene_manifest_twente_{y}_SLC.csv"):
            if int(r["relative_orbit"]) == 88:
                ids.add(r["scene_id"])
    return ids


def search(year, tracks, start, end):
    res = asf.search(platform=[asf.PLATFORM.SENTINEL1], processingLevel=asf.PRODUCT_TYPE.SLC,
                     beamMode=asf.BEAMMODE.IW, intersectsWith=AOI.wkt, relativeOrbit=list(tracks),
                     start=f"{year}-{start}T00:00:00Z", end=f"{year}-{end}T23:59:59Z")
    rows, fp = [], {}
    for r in res:
        p = r.properties
        g = transform(TO_M, shape(r.geometry))
        fp[p["sceneName"]] = g
        rows.append({"year": year, "track": int(p["pathNumber"]), "date": p["startTime"][:10],
                     "platform": p.get("platform", "?"), "scene_id": p["sceneName"],
                     "coverage": round(g.intersection(AOI_M).area / AOI_M.area, 6)})
    return rows, fp


def main():
    t88_site1 = site1_t88_scenes()
    scenes, fp = [], {}
    for y in (2024, 2025):
        r, f = search(y, (15, 37, 88, 139), "03-01", "10-31")
        scenes += r
        fp.update(f)
    # best frame per (year, track, date, platform); T88 prefers the site-1 frame if it covers enough
    best = {}
    for r in scenes:
        k = (r["year"], r["track"], r["date"], r["platform"])
        pref = (r["track"] == 88 and r["scene_id"] in t88_site1 and r["coverage"] >= MIN_COV)
        score = (pref, r["coverage"], r["scene_id"])
        if k not in best or score > best[k][0]:
            best[k] = (score, r)
    chosen = [v[1] for v in best.values()]
    for y in (2024, 2025):
        wcsv(OUT / f"scene_manifest_{y}.csv", sorted([r for r in chosen if r["year"] == y],
                                                     key=lambda r: (r["track"], r["date"])))

    pairs = []
    for (y, t) in {(r["year"], r["track"]) for r in chosen}:
        s = sorted([r for r in chosen if r["year"] == y and r["track"] == t and r["platform"] == "Sentinel-1A"],
                   key=lambda r: r["date"])
        for a in s:
            for b in s:
                lag = (date.fromisoformat(b["date"]) - date.fromisoformat(a["date"])).days
                if lag != 12:
                    continue
                ov = fp[a["scene_id"]].intersection(fp[b["scene_id"]]).intersection(AOI_M).area / AOI_M.area
                pairs.append({"pair_id": f"PAIR_{y}_T{t}_S1A_S1A_{a['date'].replace('-', '')}_{b['date'].replace('-', '')}_12D",
                              "year": y, "track": t, "date_1": a["date"], "date_2": b["date"], "lag_days": 12,
                              "scene_id_1": a["scene_id"], "scene_id_2": b["scene_id"], "overlap": round(ov, 6),
                              "status": "CANDIDATE" if ov >= MIN_COV else "REJECTED"})
    for y in (2024, 2025):
        wcsv(OUT / f"pair_manifest_{y}.csv", sorted([p for p in pairs if p["year"] == y],
                                                    key=lambda p: (p["track"], p["date_1"])))
    by_id = {p["pair_id"]: p for p in pairs}
    mid = lambda p: (date.fromisoformat(p["date_1"]).toordinal() + date.fromisoformat(p["date_2"]).toordinal()) / 2  # noqa: E731
    doy = lambda p: date.fromisoformat(p["date_1"]).timetuple().tm_yday  # noqa: E731

    def ladder(y, t):
        cand = sorted([p for p in pairs if p["year"] == y and p["track"] == t and p["status"] == "CANDIDATE"],
                      key=lambda p: (p["date_1"], p["pair_id"]))
        dis, used = [], set()
        for p in cand:
            e = {p["scene_id_1"], p["scene_id_2"]}
            if not used & e:
                dis.append(p)
                used |= e
        if len(dis) < max(M_VALUES):
            raise SystemExit(f"STOP: only {len(dis)} disjoint pairs for {y} T{t}")
        return dis

    design = {}
    for t in (37, 88):
        dis = ladder(2025, t)
        nested = [dis[0], dis[-1]]
        sel = {2: list(nested)}
        while len(nested) < 9:
            o = sorted(nested, key=mid)
            gaps = [(mid(b) - mid(a), a, b) for a, b in zip(o[:-1], o[1:])]
            g = max(x[0] for x in gaps)
            a, b = sorted([(a, b) for gg, a, b in gaps if gg == g], key=lambda x: (x[0]["pair_id"], x[1]["pair_id"]))[0]
            tm = (mid(a) + mid(b)) / 2
            inside = [p for p in dis if p not in nested and mid(a) < mid(p) < mid(b)] or [p for p in dis if p not in nested]
            nested.append(sorted(inside, key=lambda p: (abs(mid(p) - tm), p["pair_id"]))[0])
            sel[len(nested)] = sorted(nested, key=lambda p: p["date_1"])
        sel[1] = [sorted(dis, key=lambda p: (abs(mid(p) - date(2025, 6, 30).toordinal()), p["pair_id"]))[0]]
        for m in M_VALUES:
            design[(2025, t, m)] = sel[m]
        # 2024: DOY match to the 2025 M=9 slots, non-overlapping, within 14 days
        ref = sel[9]
        opts = sorted([p for p in pairs if p["year"] == 2024 and p["track"] == t and p["status"] == "CANDIDATE"],
                      key=lambda p: (p["date_1"], p["pair_id"]))

        @lru_cache(None)
        def match(i, prev):
            if i == len(ref):
                return (0, ())
            bestm = None
            for j, o in enumerate(opts):
                if prev >= 0 and o["date_1"] <= opts[prev]["date_2"]:
                    continue
                d = abs(doy(ref[i]) - doy(o))
                if d > DOY_TOL:
                    continue
                tail = match(i + 1, j)
                if tail is not None and (bestm is None or (d + tail[0], (j,) + tail[1]) < bestm):
                    bestm = (d + tail[0], (j,) + tail[1])
            return bestm
        sol = match(0, -1)
        if sol is None:
            raise SystemExit(f"STOP: no 2024 DOY match for T{t}")
        mp = {r["pair_id"]: opts[j] for r, j in zip(ref, sol[1])}
        single = sorted([o for o in opts if abs(doy(sel[1][0]) - doy(o)) <= DOY_TOL],
                        key=lambda o: (abs(doy(sel[1][0]) - doy(o)), o["pair_id"]))[0]
        for m in M_VALUES:
            design[(2024, t, m)] = [single] if m == 1 else sorted([mp[p["pair_id"]] for p in design[(2025, t, m)]],
                                                                   key=lambda p: p["date_1"])
    rows = []
    for (y, t, m), ps in sorted(design.items()):
        sc = sorted({s for p in ps for s in (p["scene_id_1"], p["scene_id_2"])})
        assert len(sc) == 2 * m
        rows.append({"year": y, "track": t, "branch": "MAIN_12D_AA", "M": m, "pair_ids": ";".join(p["pair_id"] for p in ps),
                     "scene_ids": ";".join(sc), "first_date": ps[0]["date_1"], "last_date": ps[-1]["date_2"]})
    wcsv(OUT / "budget_design.csv", rows)

    # Study-1 pairs (site-1 dates); must exist as candidate 12-day S1A pairs at site 2
    s1 = []
    for season, d in STUDY1.items():
        for t, (d1, d2) in d.items():
            m = [p for p in pairs if p["track"] == t and p["date_1"] == d1 and p["date_2"] == d2 and p["status"] == "CANDIDATE"]
            if not m:
                raise SystemExit(f"STOP: Study-1 {season} T{t} {d1}/{d2} not available at site 2")
            s1.append({"season": season, "track": t, **{k: m[0][k] for k in ("pair_id", "date_1", "date_2", "scene_id_1", "scene_id_2", "overlap")}})
    wcsv(OUT / "study1_pairs.csv", s1)

    # job list: RTC20 per scene, INSAR per pair; mark products reusable from site-1 downloads
    have = {z.name.split("_")[2] if z.name.startswith("S1A_IW_") else z.name.split("_")[1] + "_" + z.name.split("_")[2]: z.name
            for z in list(S1_DL.glob("*.zip")) + list(S1_P0.glob("*.zip"))}
    need_scenes, need_pairs = set(), set()
    for r in rows:
        need_scenes |= set(r["scene_ids"].split(";"))
        need_pairs |= set(r["pair_ids"].split(";"))
    for r in s1:
        need_scenes |= {r["scene_id_1"], r["scene_id_2"]}
        need_pairs.add(r["pair_id"])
    jobs, reuse = [], []
    key = lambda sid: sid.split("_")[5]  # noqa: E731  start time YYYYMMDDTHHMMSS
    for sid in sorted(need_scenes):
        z = have.get(key(sid)) if sid in t88_site1 else None
        (reuse.append({"kind": "RTC", "input": sid, "site1_zip": z}) if z else
         jobs.append({"job_type": "RTC_GAMMA", "inputs": sid, "name": f"S2_RTC_{key(sid)}"}))
    for pid in sorted(need_pairs):
        p = by_id[pid]
        k = f"{key(p['scene_id_1'])}_{key(p['scene_id_2'])}"
        z = have.get(k) if p["scene_id_1"] in t88_site1 and p["scene_id_2"] in t88_site1 else None
        (reuse.append({"kind": "INSAR", "input": pid, "site1_zip": z}) if z else
         jobs.append({"job_type": "INSAR_GAMMA", "inputs": f"{p['scene_id_1']}&{p['scene_id_2']}", "name": f"S2_INS_{k}"}))
    wcsv(OUT / "job_list.csv", jobs)
    if reuse:
        wcsv(OUT / "reuse_map.csv", reuse)
    rep = {"aoi": list(AOI.bounds), "n_scenes": len(chosen), "n_pairs_candidate": sum(p["status"] == "CANDIDATE" for p in pairs),
           "jobs_new": len(jobs), "jobs_new_rtc": sum(j["job_type"] == "RTC_GAMMA" for j in jobs),
           "jobs_new_insar": sum(j["job_type"] == "INSAR_GAMMA" for j in jobs), "reused_site1": len(reuse),
           "credits_new": 15 * len(jobs),
           "t88_design_equals_site1": None}
    s1d = {(int(r["year"]), int(r["track"]), int(r["M"])): r for r in rcsv(S1_MAN / "budget_design.csv") if r["branch"] == "MAIN_12D_AA"}
    rep["t88_design_equals_site1"] = all(
        sorted(s1d[(y, 88, m)]["scene_ids"].split(";")) == sorted(next(r for r in rows if r["year"] == y and r["track"] == 88 and r["M"] == m)["scene_ids"].split(";"))
        for y in (2024, 2025) for m in M_VALUES)
    (OUT / "MANIFEST_REPORT.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()

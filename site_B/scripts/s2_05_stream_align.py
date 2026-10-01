"""Site 2: download HyP3 products as they finish, align them to the site-2 40 m grid, delete own zips.

Alignment is identical to A1 a1_s4_build_cube.py (L46-58, L137-159): linear gamma0 VV/VH, COH (corr)
and THETA (inc_map_ell) area-averaged (Resampling.average); PHASE as mean phasor (cos, sin bands).
Differences, required by the 77 GB of free disk (G0_SITE2_REPORT.md):
  * each product's index record is written to ../runs/MAIN/cube/product_index.jsonl as soon as it is
    aligned, so parcel tables can be rebuilt after the zip is deleted;
  * zips downloaded here (../runs/MAIN/downloads) are deleted after alignment; site-1 zips listed in
    ../manifests/reuse_map.csv are only read, never deleted.
Usage: python s2_05_stream_align.py            (loop until every job and reuse product is aligned)
"""
import csv
import hashlib
import json
import time
import zipfile
from pathlib import Path

import hyp3_sdk
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "MAIN"
DL, AL, CUBE = RUN / "downloads", RUN / "aligned_40m", RUN / "cube"
GRID = ROOT / "runs" / "G0L_COMMON_GRID" / "grid_spec.json"
SAR = ROOT.parent
SITE1_DIRS = [SAR / "A1_observation_budget_20260923" / "runs" / "20260924_A1_MAIN" / "downloads",
              SAR / "A1_observation_budget_20260923" / "runs" / "20260923_P0_PILOT" / "downloads"]
INDEX = CUBE / "product_index.jsonl"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(4 << 20), b""):
            h.update(c)
    return h.hexdigest()


def catalogue():
    cat = {}
    for y in (2024, 2025):
        for r in csv.DictReader(open(ROOT / "manifests" / f"scene_manifest_{y}.csv", encoding="utf-8")):
            cat[r["scene_id"].split("_")[5]] = {"year": int(r["year"]), "track": int(r["track"]), "date": r["date"],
                                                "platform": r["platform"].replace("Sentinel-", "S")}
    return cat


def parse(stem):
    p = stem.split("_")
    if stem.startswith("S1") and "_RTC" in stem:
        return {"kind": "RTC", "key1": p[2]}
    if "_INT40_" in stem:
        return {"kind": "INSAR", "key1": p[1], "key2": p[2], "lag": int(p[3][3:6])}
    return None


def align(src, crs, w, h, tr, fn=None):
    with rasterio.open(src) as s:
        a = s.read(1).astype("float32")
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        a[~np.isfinite(a)] = np.nan
        if fn is not None:
            a = fn(a)
        d = np.full((h, w), np.nan, dtype="float32")
        reproject(a, d, src_transform=s.transform, src_crs=s.crs, src_nodata=np.nan, dst_transform=tr,
                  dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.average)
    return d


def save(path, arrs, crs, tr):
    """Atomic: write to .tmp.tif then rename, so an interrupted run never leaves a truncated raster."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.tif")
    h, w = arrs[0].shape
    with rasterio.open(tmp, "w", driver="GTiff", dtype="float32", nodata=np.nan, width=w, height=h,
                       count=len(arrs), crs=crs, transform=tr, compress="deflate") as d:
        for i, a in enumerate(arrs, 1):
            d.write(a.astype("float32"), i)
    tmp.replace(path)


def cleanup(done_records):
    """Startup: drop rasters not backed by an index record (possibly truncated by an interruption) and
    any own zip left in downloads (partial or unprocessed; it is re-downloaded). Site-1 zips are untouched."""
    keep = {AL / str(r["year"]) / f"T{r['track']}" / f"{lay}_{r['key']}.tif"
            for r in done_records for lay in filter(None, r["layers"].split(";"))}
    removed = [p for p in AL.rglob("*.tif") if p not in keep]
    for p in removed:
        p.unlink()
    zips = list(DL.glob("*.zip"))
    for z in zips:
        z.unlink()
    print(f"cleanup: removed {len(removed)} unindexed rasters, {len(zips)} leftover own zips", flush=True)


def uid_of(stem, cat):
    """Product uid (kind|year|track|key) from a product name, without opening the zip."""
    info = parse(stem)
    s1 = cat.get(info["key1"]) if info else None
    if s1 is None:
        return None
    key = f"{s1['date']}_{cat[info['key2']]['date']}" if info["kind"] == "INSAR" else s1["date"]
    return f"{info['kind']}|{s1['year']}|{s1['track']}|{key}"


def process(zp, cat, g, done, own):
    if uid_of(zp.stem, cat) in done:  # already aligned (e.g. re-downloaded after a restart)
        if own:
            zp.unlink()
        return
    info = parse(zp.stem)
    s1 = cat.get(info["key1"]) if info else None
    if s1 is None:
        raise SystemExit(f"STOP: {zp.name} not in the site-2 scene manifest")
    rec = {"product": zp.stem, "kind": info["kind"], "year": s1["year"], "track": s1["track"], "date1": s1["date"],
           "platform1": s1["platform"], "date2": "", "platform2": "", "lag_days": "", "zip_sha256": sha256(zp),
           "source": "site2_download" if own else "site1_reuse"}
    if info["kind"] == "INSAR":
        s2 = cat[info["key2"]]
        rec.update(date2=s2["date"], platform2=s2["platform"], lag_days=info["lag"])
        key = f"{s1['date']}_{s2['date']}"
    else:
        key = s1["date"]
    rec["key"] = key
    uid = f"{rec['kind']}|{rec['year']}|{rec['track']}|{key}"
    if uid in done:
        return
    crs, W, H, TR = g
    with zipfile.ZipFile(zp) as z:
        names = [m.filename for m in z.infolist() if not m.is_dir()]
    sufs = (("VV", "_VV.tif"), ("VH", "_VH.tif")) if info["kind"] == "RTC" else \
        (("COH", "_corr.tif"), ("THETA", "_inc_map_ell.tif"), ("PHASE", "_wrapped_phase.tif"))
    layers = []
    for lay, suf in sufs:
        m = next((n for n in names if n.endswith(suf)), None)
        if m is None:
            continue
        src = f"/vsizip/{zp.resolve().as_posix()}/{m}"
        out = AL / str(rec["year"]) / f"T{rec['track']}" / f"{lay}_{key}.tif"
        if not out.exists():
            arrs = [align(src, crs, W, H, TR, np.cos), align(src, crs, W, H, TR, np.sin)] if lay == "PHASE" \
                else [align(src, crs, W, H, TR)]
            save(out, arrs, crs, TR)
        with rasterio.open(out) as s:
            a = s.read(1)
        v = a[np.isfinite(a)]
        rec[f"{lay}_valid_frac"] = round(v.size / a.size, 4)
        if lay in ("VV", "VH") and v.size:
            rec[f"{lay}_median_db"] = round(float(10 * np.log10(np.median(v[v > 0]))), 3)
        if lay == "COH" and v.size:
            rec["COH_median"] = round(float(np.median(v)), 4)
        layers.append(lay)
    rec["layers"] = ";".join(layers)
    rec["uid"] = uid
    with open(INDEX, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    done.add(uid)
    print("aligned", zp.name, rec["layers"], flush=True)
    if own:
        zp.unlink()


def main():
    for d in (DL, AL, CUBE):
        d.mkdir(parents=True, exist_ok=True)
    gs = json.loads(GRID.read_text())
    g = (gs["crs"], int(gs["width"]), int(gs["height"]), Affine.from_gdal(*gs["affine_gdal"]))
    cat = catalogue()
    records = [json.loads(l) for l in open(INDEX, encoding="utf-8")] if INDEX.exists() else []
    cleanup(records)
    done = {r["uid"] for r in records}
    # 1) site-1 products reused in place (read-only)
    for r in csv.DictReader(open(ROOT / "manifests" / "reuse_map.csv", encoding="utf-8")):
        zp = next(d / r["site1_zip"] for d in SITE1_DIRS if (d / r["site1_zip"]).exists())
        process(zp, cat, g, done, own=False)
    # 2) own jobs: download as they succeed, align, delete
    rec = json.loads(sorted((ROOT / "submit").glob("submitted_*.json"))[-1].read_text(encoding="utf-8"))
    hyp3 = hyp3_sdk.HyP3()
    ids = [j["job_id"] for j in rec["jobs"]]
    handled = set()
    while True:
        batch = hyp3_sdk.Batch([hyp3.get_job_by_id(i) for i in ids])
        failed = [j.name for j in batch if j.failed()]
        if failed:
            raise SystemExit(f"STOP: failed jobs {failed}")
        for j in batch:
            if j.succeeded() and j.job_id not in handled:
                stems = [Path(f["filename"]).stem for f in j.files]
                if stems and all(uid_of(s, cat) in done for s in stems):  # aligned in an earlier run
                    handled.add(j.job_id)
                    continue
                for zp in j.download_files(DL):
                    process(Path(zp), cat, g, done, own=True)
                handled.add(j.job_id)
        n_ok = sum(j.succeeded() for j in batch)
        print(f"{time.strftime('%H:%M')} succeeded {n_ok}/{len(batch)}, handled {len(handled)}", flush=True)
        if len(handled) == len(batch):
            break
        time.sleep(120)
    print("ALL PRODUCTS ALIGNED", flush=True)


if __name__ == "__main__":
    main()

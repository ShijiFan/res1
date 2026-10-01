"""Step 4 (E0): align every downloaded HyP3 product to the common 40 m grid and build parcel tables.

Reads products directly from the zips (GDAL /vsizip), so nothing large is extracted to disk.
Sources: RUN/downloads/*.zip and runs/20260923_P0_PILOT/downloads/*.zip.

Per product:
  RTC  -> VV, VH (linear gamma0 power, area-average to 40 m)
  InSAR-> COH (corr, area-average), THETA (inc_map_ell, if present), PHASE (wrapped phase as mean phasor,
          2 bands cos/sin, if present)
Outputs:
  RUN/aligned_40m/{year}/T{track}/{layer}_{key}.tif
  RUN/cube/PRODUCT_INDEX.csv         product -> year, track, dates, platforms, QC
  RUN/cube/parcel_long_{year}.csv.gz long table: parcel x layer x key (mean over pure pixels, n valid px)
  RUN/reports/E0_QC.md
Idempotent: aligned rasters that already exist are not recomputed.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

from a1_common import (D, G0L, GRID_SPEC, P0, ensure_dirs, get_logger, load_scene_catalogue, match_scene, now,
                       parse_product, read_json, sha256_file, update_status, write_json)

log = get_logger("s4_build_cube")


def grid():
    g = read_json(GRID_SPEC)
    return g["crs"], int(g["width"]), int(g["height"]), Affine.from_gdal(*g["affine_gdal"])


def vsizip(zp: Path, member: str) -> str:
    return f"/vsizip/{zp.resolve().as_posix()}/{member}"


def align(src_path: str, crs, w, h, tr, transform_fn=None) -> np.ndarray:
    with rasterio.open(src_path) as s:
        a = s.read(1).astype("float32")
        nod = s.nodata
        if nod is not None:
            a[a == nod] = np.nan
        a[~np.isfinite(a)] = np.nan
        if transform_fn is not None:
            a = transform_fn(a)
        dst = np.full((h, w), np.nan, dtype="float32")
        reproject(a, dst, src_transform=s.transform, src_crs=s.crs, src_nodata=np.nan,
                  dst_transform=tr, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.average)
    return dst


def save(path: Path, arrs, crs, tr):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs = [arrs] if isinstance(arrs, np.ndarray) else arrs
    h, w = arrs[0].shape
    with rasterio.open(path, "w", driver="GTiff", dtype="float32", nodata=np.nan, width=w, height=h,
                       count=len(arrs), crs=crs, transform=tr, compress="deflate") as d:
        for i, a in enumerate(arrs, 1):
            d.write(a.astype("float32"), i)


def load(path: Path):
    with rasterio.open(path) as s:
        return [s.read(i).astype("float64") for i in range(1, s.count + 1)]


def members(zp: Path):
    with zipfile.ZipFile(zp) as z:
        return [m.filename for m in z.infolist() if not m.is_dir()]


def pick(names, suffix):
    c = [n for n in names if n.endswith(suffix)]
    return c[0] if c else None


def parcel_means(arr, labels, n_lab):
    ok = np.isfinite(arr) & (labels > 0)
    s = np.bincount(labels[ok], weights=arr[ok], minlength=n_lab)
    c = np.bincount(labels[ok], minlength=n_lab)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / c
    return m, c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-hash", action="store_true", help="skip SHA-256 of zips (faster)")
    ap.add_argument("--extra-zip-dir", action="append", default=[], help="additional folder with product zips")
    a = ap.parse_args()
    ensure_dirs()
    crs, W, H, TR = grid()
    cat = load_scene_catalogue()
    zips = sorted(D["downloads"].glob("*.zip")) + sorted((P0 / "downloads").glob("*.zip"))
    for extra in a.extra_zip_dir:
        zips += sorted(Path(extra).glob("*.zip"))
    log.info("found %d zips", len(zips))

    index, seen = [], set()
    for zp in zips:
        info = parse_product(zp.stem)
        if info is None:
            log.warning("unrecognised product name: %s", zp.name)
            continue
        s1 = match_scene(info["key1"], cat)
        if s1 is None:
            log.warning("no scene match for %s", zp.name)
            continue
        rec = {"product": zp.stem, "zip": str(zp), "kind": info["kind"], "year": s1["year"], "track": s1["track"],
               "date1": s1["date"], "platform1": s1["platform"], "date2": "", "platform2": "", "lag_days": ""}
        if info["kind"] == "INSAR":
            s2 = match_scene(info["key2"], cat)
            if s2 is None:
                log.warning("no scene match for secondary of %s", zp.name)
                continue
            rec.update(date2=s2["date"], platform2=s2["platform"], lag_days=info["lag"])
            key = f"{rec['date1']}_{rec['date2']}"
        else:
            key = rec["date1"]
        uid = (rec["kind"], rec["year"], rec["track"], key)
        if uid in seen:
            log.warning("duplicate acquisition %s (%s); keeping the first", uid, zp.name)
            continue
        seen.add(uid)
        rec["key"] = key
        rec["zip_sha256"] = "" if a.no_hash else sha256_file(zp)

        out_dir = D["aligned"] / str(rec["year"]) / f"T{rec['track']}"
        names = members(zp)
        layers = {}
        if info["kind"] == "RTC":
            for pol in ("VV", "VH"):
                m = pick(names, f"_{pol}.tif")
                if m:
                    layers[pol] = (m, None)
        else:
            for lay, suf in (("COH", "_corr.tif"), ("THETA", "_inc_map_ell.tif"), ("PHASE", "_wrapped_phase.tif")):
                m = pick(names, suf)
                if m:
                    layers[lay] = (m, None)
        for lay, (m, _) in layers.items():
            p = out_dir / f"{lay}_{key}.tif"
            if not p.exists():
                if lay == "PHASE":
                    c = align(vsizip(zp, m), crs, W, H, TR, np.cos)
                    s = align(vsizip(zp, m), crs, W, H, TR, np.sin)
                    save(p, [c, s], crs, TR)
                else:
                    save(p, align(vsizip(zp, m), crs, W, H, TR), crs, TR)
                log.info("aligned %s %s", zp.stem, lay)
            arr = load(p)[0]
            v = arr[np.isfinite(arr)]
            rec[f"{lay}_valid_frac"] = round(v.size / arr.size, 4)
            if lay in ("VV", "VH") and v.size:
                rec[f"{lay}_median_db"] = round(float(10 * np.log10(np.median(v[v > 0]))), 3)
            if lay == "COH" and v.size:
                rec["COH_median"] = round(float(np.median(v)), 4)
        rec["layers"] = ";".join(layers)
        index.append(rec)

    fields = sorted({k for r in index for k in r}, key=lambda k: (k not in ("product", "kind", "year", "track", "key"), k))
    with open(D["cube"] / "PRODUCT_INDEX.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(index)

    # parcel long tables
    for year in sorted({r["year"] for r in index}):
        with rasterio.open(G0L / f"parcel_raster_{year}.tif") as s:
            lab = s.read(1).astype("int64")
            assert s.width == W and s.height == H, "parcel raster does not match grid"
        lab[lab < 0] = 0
        n_lab = int(lab.max()) + 1
        elig = {int(r["source_row"]): r for r in csv.DictReader(open(G0L / f"parcel_table_{year}.csv", encoding="utf-8"))
                if r["eligible_ge2"] == "1"}
        rows_idx = np.array(sorted(elig), dtype="int64")
        outp = D["cube"] / f"parcel_long_{year}.csv.gz"
        with gzip.open(outp, "wt", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["year", "track", "source_row", "parcel_id", "layer", "key", "date1", "date2", "lag_days",
                        "platform1", "platform2", "value", "n_valid_px"])
            for r in (x for x in index if x["year"] == year):
                for lay in r["layers"].split(";"):
                    if not lay:
                        continue
                    p = D["aligned"] / str(year) / f"T{r['track']}" / f"{lay}_{r['key']}.tif"
                    bands = load(p)
                    if lay == "PHASE":
                        mc, cnt = parcel_means(bands[0], lab, n_lab)
                        ms, _ = parcel_means(bands[1], lab, n_lab)
                        vals = np.arctan2(ms, mc)
                    else:
                        vals, cnt = parcel_means(bands[0], lab, n_lab)
                    for sr in rows_idx:
                        if sr < n_lab and cnt[sr] > 0 and np.isfinite(vals[sr]):
                            w.writerow([year, r["track"], sr, elig[sr]["parcel_id"], lay, r["key"], r["date1"],
                                        r["date2"], r["lag_days"], r["platform1"], r["platform2"],
                                        f"{vals[sr]:.6g}", int(cnt[sr])])
        log.info("wrote %s", outp)

    # QC
    bad = []
    for r in index:
        for lay in r["layers"].split(";"):
            vf = r.get(f"{lay}_valid_frac")
            if vf is not None and vf < 0.90:
                bad.append(f"{r['product']} {lay} valid_frac={vf}")
    by = {}
    for r in index:
        if r["kind"] == "RTC" and "VV_median_db" in r:
            by.setdefault((r["year"], r["track"]), []).append(r["VV_median_db"])
    for r in index:
        if r["kind"] == "RTC" and "VV_median_db" in r:
            med = float(np.median(by[(r["year"], r["track"])]))
            if abs(r["VV_median_db"] - med) > 3:
                bad.append(f"{r['product']} VV median {r['VV_median_db']} dB vs track-year median {med:.2f}")
    counts = {}
    for r in index:
        k = f"{r['year']}_T{r['track']}_{r['kind']}"
        counts[k] = counts.get(k, 0) + 1
    md = [f"# E0 QC ({now()})", "", "## Products per year/track/kind", "", "| key | n |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in sorted(counts.items())]
    md += ["", "## Flags (valid fraction < 0.90 of the grid, or VV median > 3 dB from the track-year median)", ""]
    md += [f"- {b}" for b in bad] or ["- none"]
    md += ["", "Grid valid fraction is relative to the whole 966x1148 grid; values near 1 are expected for T37/T88.",
           "A flagged product is kept in the index; step 5 drops parcels with missing values per configuration."]
    (D["reports"] / "E0_QC.md").write_text("\n".join(md), encoding="utf-8")
    write_json(D["cube"] / "E0_SUMMARY.json", {"utc": now(), "counts": counts, "flags": bad})
    update_status("s4_build_cube", "DONE", products=len(index), flags=len(bad))


if __name__ == "__main__":
    main()

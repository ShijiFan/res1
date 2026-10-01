"""Site 2: PRODUCT_INDEX.csv, parcel long tables and E0 QC from the aligned rasters.

Same computation as A1 a1_s4_build_cube.py L170-237, but driven by runs/MAIN/cube/product_index.jsonl
(written by s2_05_stream_align.py) because site-2 zips are deleted after alignment.
Out: runs/MAIN/cube/{PRODUCT_INDEX.csv, parcel_long_{year}.csv.gz, E0_SUMMARY.json}, runs/MAIN/reports/E0_QC.md
"""
import csv
import gzip
import json

import numpy as np
import rasterio

from a1_common import D, G0L, ensure_dirs, now, write_json


def load(p):
    with rasterio.open(p) as s:
        return [s.read(i).astype("float64") for i in range(1, s.count + 1)]


def parcel_means(arr, labels, n_lab):
    ok = np.isfinite(arr) & (labels > 0)
    s = np.bincount(labels[ok], weights=arr[ok], minlength=n_lab)
    c = np.bincount(labels[ok], minlength=n_lab)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s / c, c


def main():
    ensure_dirs()
    index = [json.loads(l) for l in open(D["cube"] / "product_index.jsonl", encoding="utf-8")]
    uids = [r["uid"] for r in index]
    assert len(uids) == len(set(uids)), "duplicate products in index"
    fields = sorted({k for r in index for k in r}, key=lambda k: (k not in ("product", "kind", "year", "track", "key"), k))
    with open(D["cube"] / "PRODUCT_INDEX.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(index)
    for year in sorted({r["year"] for r in index}):
        with rasterio.open(G0L / f"parcel_raster_{year}.tif") as s:
            lab = s.read(1).astype("int64")
        lab[lab < 0] = 0
        n_lab = int(lab.max()) + 1
        elig = {int(r["source_row"]): r for r in csv.DictReader(open(G0L / f"parcel_table_{year}.csv", encoding="utf-8"))
                if r["eligible_ge2"] == "1"}
        rows_idx = np.array(sorted(elig), dtype="int64")
        with gzip.open(D["cube"] / f"parcel_long_{year}.csv.gz", "wt", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["year", "track", "source_row", "parcel_id", "layer", "key", "date1", "date2", "lag_days",
                        "platform1", "platform2", "value", "n_valid_px"])
            for r in (x for x in index if x["year"] == year):
                for lay in filter(None, r["layers"].split(";")):
                    bands = load(D["aligned"] / str(year) / f"T{r['track']}" / f"{lay}_{r['key']}.tif")
                    if lay == "PHASE":
                        mc, cnt = parcel_means(bands[0], lab, n_lab)
                        ms, _ = parcel_means(bands[1], lab, n_lab)
                        vals = np.arctan2(ms, mc)
                    else:
                        vals, cnt = parcel_means(bands[0], lab, n_lab)
                    for sr in rows_idx:
                        if sr < n_lab and cnt[sr] > 0 and np.isfinite(vals[sr]):
                            w.writerow([year, r["track"], sr, elig[sr]["parcel_id"], lay, r["key"], r["date1"], r["date2"],
                                        r["lag_days"], r["platform1"], r["platform2"], f"{vals[sr]:.6g}", int(cnt[sr])])
        print("wrote parcel_long", year, flush=True)
    bad, by = [], {}
    for r in index:
        for lay in filter(None, r["layers"].split(";")):
            vf = r.get(f"{lay}_valid_frac")
            if vf is not None and vf < 0.90:
                bad.append(f"{r['product']} {lay} valid_frac={vf}")
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
    md = [f"# Site-2 E0 QC ({now()})", "", "| key | n |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sorted(counts.items())]
    md += ["", "## Flags", ""] + ([f"- {b}" for b in bad] or ["- none"])
    (D["reports"] / "E0_QC.md").write_text("\n".join(md), encoding="utf-8")
    write_json(D["cube"] / "E0_SUMMARY.json", {"utc": now(), "counts": counts, "flags": bad})
    print(json.dumps(counts), "flags:", len(bad))


if __name__ == "__main__":
    main()

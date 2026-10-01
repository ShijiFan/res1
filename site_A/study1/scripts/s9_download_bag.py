"""Download BAG building footprints (pand) for the A0 AOI from the PDOK OGC API v2.

Source: https://api.pdok.nl/kadaster/bag/ogc/v2/collections/pand (Public Domain Mark 1.0).
The API serves the CURRENT register; construction year (bouwjaar) and status are kept so that
buildings present in May 2024 can be selected later (bouwjaar <= 2024).
Out: ../labels/bag_pand_aoi.geojsonl (one feature per line, deduplicated) + download_meta.json
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "labels"
BASE = "https://api.pdok.nl/kadaster/bag/ogc/v2/collections/pand/items"
AOI = [6.00, 51.98, 6.54, 52.38]
STEP = 0.02


def fetch_tile(bbox):
    url = f"{BASE}?bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}&limit=1000&f=json"
    feats = []
    while url:
        for attempt in range(6):
            try:
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                d = r.json()
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"failed {url}")
        feats += d["features"]
        url = next((l["href"] for l in d["links"] if l["rel"] == "next"), None)
    return feats


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    xs = np.arange(AOI[0], AOI[2], STEP)
    ys = np.arange(AOI[1], AOI[3], STEP)
    tiles = [(round(x, 4), round(y, 4), round(min(x + STEP, AOI[2]), 4), round(min(y + STEP, AOI[3]), 4))
             for x in xs for y in ys]
    tiles = [t for t in tiles if t[2] - t[0] > 1e-6 and t[3] - t[1] > 1e-6]  # drop zero-area edge tiles
    seen, n_raw = set(), 0
    with (OUT / "bag_pand_aoi.geojsonl").open("w", encoding="utf-8") as f, ThreadPoolExecutor(6) as ex:
        for i, feats in enumerate(ex.map(fetch_tile, tiles)):
            n_raw += len(feats)
            for ft in feats:
                k = ft["properties"]["identificatie"]
                if k in seen:
                    continue
                seen.add(k)
                p = ft["properties"]
                f.write(json.dumps({"type": "Feature", "geometry": ft["geometry"],
                                    "properties": {"id": k, "bouwjaar": p.get("bouwjaar"),
                                                   "status": p.get("status"), "documentdatum": p.get("documentdatum"),
                                                   "gebruiksdoel": p.get("gebruiksdoel")}}) + "\n")
            if i % 50 == 0:
                print(f"tile {i}/{len(tiles)} unique {len(seen)}", flush=True)
    meta = {"downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": BASE,
            "license": "Public Domain Mark 1.0", "aoi": AOI, "tile_deg": STEP, "n_tiles": len(tiles),
            "features_raw": n_raw, "features_unique": len(seen)}
    (OUT / "download_meta.json").write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    main()

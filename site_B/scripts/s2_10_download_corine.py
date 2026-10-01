"""Site 2: CORINE Land Cover 2018 polygons for the site-2 bbox (Study-1 replication labels).

Same EEA ArcGIS REST query, paging and completeness check as sar_v2/labels/download_labels_FIXED.py
(L48-98); only the bbox and output folder differ.
Out: ../labels/corine2018/{clc2018_site2.geojson, clc2018_site2_meta.json}
"""
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "labels" / "corine2018"
AOI_BBOX = [6.55, 52.75, 7.05, 53.10]
BASE = "https://image.discomap.eea.europa.eu/arcgis/rest/services/Corine/CLC2018_WM/MapServer/0/query?"
PAGE, MAX_PAGES = 1000, 200


def get_json(url, tries=6):
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # network retry
            last = e
            time.sleep(5 * (i + 1))
    raise SystemExit(f"FATAL: request failed after {tries} tries: {last}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    geo = {"geometry": ",".join(map(str, AOI_BBOX)), "geometryType": "esriGeometryEnvelope", "inSR": 4326,
           "spatialRel": "esriSpatialRelIntersects"}
    total = int(get_json(BASE + urllib.parse.urlencode({**geo, "returnCountOnly": "true", "f": "json"})).get("count", -1))
    if total < 0:
        raise SystemExit("FATAL: no count returned")
    feats, offset = [], 0
    for _ in range(MAX_PAGES):
        d = get_json(BASE + urllib.parse.urlencode({**geo, "outSR": 4326, "outFields": "Code_18,Remark,ID",
                                                     "returnGeometry": "true", "resultOffset": offset,
                                                     "resultRecordCount": PAGE, "f": "geojson"}))
        got = d.get("features", [])
        feats += got
        exceeded = bool(d.get("exceededTransferLimit") or d.get("properties", {}).get("exceededTransferLimit"))
        if not got or (not exceeded and len(got) < PAGE):
            break
        offset += len(got)
    if len(feats) != total:
        raise SystemExit(f"FATAL: retrieved {len(feats)} of {total}")
    json.dump({"type": "FeatureCollection", "features": feats}, open(OUT / "clc2018_site2.geojson", "w", encoding="utf-8"))
    meta = {"dataset": "CORINE Land Cover 2018", "source": BASE, "aoi_bbox_WSEN": AOI_BBOX,
            "download_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "server_reported_count": total, "features_retrieved": len(feats),
            "codes": sorted({f["properties"]["Code_18"] for f in feats})}
    (OUT / "clc2018_site2_meta.json").write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    main()

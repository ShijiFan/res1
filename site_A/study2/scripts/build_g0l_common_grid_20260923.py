"""Build a predeclared 40 m BRP pure-pixel grid, without any SAR product.

This is label-side preflight. SAR products must later be resampled onto this
common grid with their real affine transforms; this script never claims RTC QC.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from pyproj import Transformer
from rasterio.enums import MergeAlg
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box
from shapely.ops import transform


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "20260923_G0L_COMMON_GRID"
SOURCE = Path(r"E:\research\SAR\sar_v2\labels\nl_brp")
FILES = {2024: SOURCE / "brp_parcels_2024_aoi.geojson", 2025: SOURCE / "brp_parcels_aoi.geojson"}
MAPPING = ROOT / "runs" / "20260923_0140_G0R" / "qc" / "crop_code_mapping_2024_2025.csv"
CLASSES = ("PermGrass", "TempGrass", "Maize", "WinterCereal", "SpringCereal", "Potato", "Beet", "Other")
GRID_CRS = "EPSG:32632"
PIXEL_M = 40
EROSION_M = math.sqrt(2) * PIXEL_M / 2  # farthest point from a 40 m pixel center
AOI = box(6.00, 51.98, 6.54, 52.38)
AOI_M = transform(Transformer.from_crs("EPSG:4326", GRID_CRS, always_xy=True).transform, AOI)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    if RUN.exists():
        raise SystemExit(f"Refusing to overwrite run: {RUN}")
    RUN.mkdir(parents=True)
    rule = {int(row["gewascode"]): row["assigned_8class"] for row in csv_rows(MAPPING)}
    ambiguous_codes = {238, 314, 382, 670, 2652, 6636, 7130}
    minx, miny, maxx, maxy = AOI_M.bounds
    left = math.floor(minx / PIXEL_M) * PIXEL_M
    top = math.ceil(maxy / PIXEL_M) * PIXEL_M
    width = math.ceil((maxx - left) / PIXEL_M)
    height = math.ceil((top - miny) / PIXEL_M)
    affine = from_origin(left, top, PIXEL_M, PIXEL_M)
    shape = (height, width)
    aoi_center_mask = rasterize([(AOI_M.buffer(-EROSION_M), 1)], out_shape=shape, transform=affine, fill=0, dtype="uint8") != 0
    grid = {
        "crs": GRID_CRS, "pixel_m": PIXEL_M, "width": width, "height": height,
        "affine_gdal": list(affine.to_gdal()), "origin_rule": "project AOI EPSG:4326 -> EPSG:32632; left=floor(minx/40)*40; top=ceil(maxy/40)*40",
        "pixel_purity_rule": "rasterize negative sqrt(2)*20 m polygon buffer at pixel centers; remove multi-parcel conflicts",
        "aoi_pixel_rule": "same negative buffer for AOI boundary",
        "grid_status": "PREDECLARED_COMMON_GRID; SAR product alignment pending",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (RUN / "grid_spec.json").write_text(json.dumps(grid, ensure_ascii=False, indent=2), encoding="utf-8")
    summaries: dict[int, dict] = {}

    for year, path in FILES.items():
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            raise RuntimeError(f"Source CRS missing: {path}")
        expected_crs = "EPSG:4326" if year == 2024 else "EPSG:28992"
        if gdf.crs.to_string() != expected_crs:
            raise RuntimeError(f"Unexpected {year} CRS: {gdf.crs}")
        if set(gdf["jaar"].astype(str)) != {str(year)} or set(gdf["status"].astype(str)) != {"Definitief"}:
            raise RuntimeError(f"Unexpected year/status in {path}")
        gdf = gdf.to_crs(GRID_CRS)
        geoms = np.asarray(gdf.geometry.values, dtype=object)
        valid = shapely.is_valid(geoms)
        empty = shapely.is_empty(geoms) | shapely.is_missing(geoms)
        invalid_count = int((~valid & ~empty).sum())
        if invalid_count:
            geoms[~valid & ~empty] = shapely.make_valid(geoms[~valid & ~empty])
        if not bool(np.all(shapely.is_valid(geoms[~empty]))):
            raise RuntimeError(f"Unrepaired geometry in {year}")

        codes = [int(value) if value is not None and str(value).strip().isdigit() else -1 for value in gdf["gewascode"]]
        labels = [rule.get(code, "Other") for code in codes]
        buffered = shapely.buffer(geoms, -EROSION_M)
        keep = ~(shapely.is_empty(buffered) | shapely.is_missing(buffered))
        shapes = [(geom, index + 1) for index, geom in enumerate(buffered) if keep[index]]
        counts = rasterize(((geom, 1) for geom, _ in shapes), out_shape=shape, transform=affine, fill=0, dtype="uint16", merge_alg=MergeAlg.add)
        ids = rasterize(shapes, out_shape=shape, transform=affine, fill=0, dtype="uint32")
        conflict = counts > 1
        ids[conflict | ~aoi_center_mask] = 0
        per_parcel = np.bincount(ids.ravel(), minlength=len(gdf) + 1)
        assert len(per_parcel) == len(gdf) + 1
        assert int(per_parcel[1:].sum()) == int((ids > 0).sum())
        assert int(ids.max()) <= len(gdf)

        raster_path = RUN / f"parcel_raster_{year}.tif"
        with rasterio.open(raster_path, "w", driver="GTiff", height=height, width=width,
                           count=1, dtype="uint32", crs=GRID_CRS, transform=affine,
                           nodata=0, compress="LZW", tiled=True) as output:
            output.write(ids, 1)

        areas = shapely.area(geoms) / 10000
        centroids = shapely.centroid(geoms)
        centers_x = shapely.get_x(centroids)
        centers_y = shapely.get_y(centroids)
        table_path = RUN / f"parcel_table_{year}.csv"
        class_support = Counter()
        block_support: dict[str, Counter[str]] = defaultdict(Counter)
        geometry_fingerprints = Counter()
        with table_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("year", "parcel_id", "source_row", "gewascode", "class", "area_ha", "pure_pixels_40m", "eligible_ge2", "block_5km", "geometry_sha256_16"))
            for index, geom in enumerate(geoms):
                fingerprint = hashlib.sha256(shapely.to_wkb(shapely.normalize(geom))).hexdigest()[:16]
                geometry_fingerprints[fingerprint] += 1
                parcel_id = f"{year}_{index + 1:06d}_{fingerprint}"
                pixels = int(per_parcel[index + 1])
                block = f"{math.floor(float(centers_x[index])/5000)}_{math.floor(float(centers_y[index])/5000)}"
                eligible = pixels >= 2
                writer.writerow((year, parcel_id, index + 1, codes[index], labels[index], round(float(areas[index]), 6), pixels, int(eligible), block, fingerprint))
                if eligible:
                    class_support[labels[index]] += 1
                    block_support[block][labels[index]] += 1
        with (RUN / f"block_class_support_{year}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("year", "block_5km", "class", "eligible_parcels_ge2"))
            for block, support in sorted(block_support.items()):
                for cls in CLASSES:
                    writer.writerow((year, block, cls, support[cls]))

        summaries[year] = {
            "source": str(path), "source_sha256": file_sha256(path), "source_crs": expected_crs,
            "source_rows": len(gdf), "invalid_geometry_repaired": invalid_count,
            "empty_geometry": int(empty.sum()),
            "duplicate_normalized_geometry_rows": sum(n - 1 for n in geometry_fingerprints.values() if n > 1),
            "buffer_nonempty_parcels": int(keep.sum()),
            "pure_pixels": int((ids > 0).sum()), "conflicted_pixels_excluded": int((conflict & aoi_center_mask).sum()),
            "eligible_parcels_ge2": int(sum(class_support.values())),
            "eligible_by_class_ge2": dict(class_support),
            "ambiguous_season_code_parcels": sum(1 for code in codes if code in ambiguous_codes),
            "raster": str(raster_path), "raster_sha256": file_sha256(raster_path),
            "table": str(table_path), "table_sha256": file_sha256(table_path),
        }
        (RUN / f"LABEL_GRID_QC_{year}.json").write_text(json.dumps(summaries[year], ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"year": year, "source_rows": len(gdf), "pure_pixels": summaries[year]["pure_pixels"], "eligible_parcels_ge2": summaries[year]["eligible_parcels_ge2"], "conflicted_pixels_excluded": summaries[year]["conflicted_pixels_excluded"]}, ensure_ascii=False), flush=True)

    (RUN / "G0L_COMMON_GRID_SUMMARY.json").write_text(json.dumps({"grid": grid, "years": summaries, "status": "LABEL_GRID_PREFLIGHT_COMPLETE; RTC alignment and semantic crop-code review pending"}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

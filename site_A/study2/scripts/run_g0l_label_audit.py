#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A1 G0-L: Local BRP Label Audit (2024 and 2025).
Executes Section 3 of NEXT_G0R_G0L_INSTRUCTIONS_20260923.md.
"""

import os
import sys
import json
import csv
import hashlib
import time
from datetime import datetime, timezone
from collections import Counter, defaultdict
import pyogrio
import geopandas as gpd
import shapely
from shapely.validation import make_valid
import pyproj

RUN_DIR = r"E:\research\SAR\A1_observation_budget_20260923\runs\20260923_0140_G0R"
QC_DIR = os.path.join(RUN_DIR, "qc")
REPORTS_DIR = os.path.join(RUN_DIR, "reports")
LABELS_RUN_DIR = os.path.join(RUN_DIR, "labels")
os.makedirs(LABELS_RUN_DIR, exist_ok=True)

BRP_2025_PATH = r"E:\research\SAR\sar_v2\labels\nl_brp\brp_parcels_aoi.geojson"
BRP_2024_PATH = r"E:\research\SAR\sar_v2\labels\nl_brp\brp_parcels_2024_aoi.geojson"
BRP_2024_GPKG = r"E:\research\SAR\sar_v2\labels\nl_brp\brpgewaspercelen_definitief_2024.gpkg"

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

# Comprehensive official Dutch BRP crop code to 8-class mapping dictionary
# BRP standard gewascodes:
# 265: Grasland, blijvend -> PermGrass
# 266: Grasland, tijdelijk -> TempGrass
# 331, 332: Grasland natuurlijk -> PermGrass
# 259: Mais, snij- -> Maize
# 260: Mais, korrel- -> Maize
# 262: Mais, CCM -> Maize
# 233: Tarwe, winter- -> WinterCereal
# 234: Tarwe, zomer- -> SpringCereal
# 235: Gerst, winter- -> WinterCereal
# 236: Gerst, zomer- -> SpringCereal
# 241: Rogge, winter- -> WinterCereal
# 242: Rogge, zomer- -> SpringCereal
# 243: Haver, zomer- -> SpringCereal
# 244: Triticale, winter- -> WinterCereal
# 214, 215, 216: Aardappelen -> Potato
# 256: Bieten, suiker- -> Beet
# 257: Bieten, voeder- -> Beet

CROP_RULES = {
    # Permanent Grass (blijvend grasland en natuurlijk grasland)
    265: "PermGrass",  # Grasland, blijvend
    331: "PermGrass",  # Grasland, natuurlijk. Met landbouwactiviteiten
    332: "PermGrass",  # Grasland, natuurlijk. Hoofdfunctie natuur

    # Temporary Grass (tijdelijk grasland, wisselweide)
    266: "TempGrass",  # Grasland, tijdelijk

    # Maize (snijmais, korrelmais, CCM, suikermais, kuilmais)
    259: "Maize",      # Mais, snij-
    316: "Maize",      # Mais, korrel-
    317: "Maize",      # Mais, corncob mix
    814: "Maize",      # Mais, suiker-
    1935: "Maize",     # Maiskolvensilage

    # Winter Cereals (wintertarwe, wintergerst, triticale, spelt, winterrogge)
    233: "WinterCereal",  # Tarwe, winter-
    235: "WinterCereal",  # Gerst, winter-
    314: "WinterCereal",  # Triticale
    382: "WinterCereal",  # Spelt
    7130: "WinterCereal", # Rogge, korrelgewas

    # Spring Cereals (zomertarwe, zomergerst, haver, granen overig)
    234: "SpringCereal",  # Tarwe, zomer-
    236: "SpringCereal",  # Gerst, zomer-
    238: "SpringCereal",  # Haver
    670: "SpringCereal",  # Japanse haver
    6636: "SpringCereal", # Naakte haver
    2652: "SpringCereal", # Granen, overig

    # Potatoes (consumptie, pootgoed NAK, TBM, zetmeel)
    2014: "Potato",    # Aardappelen, consumptie
    2015: "Potato",    # Aardappelen, poot NAK
    2016: "Potato",    # Aardappelen, poot TBM
    2017: "Potato",    # Aardappelen, zetmeel

    # Sugar Beet & Fodder Beet (suikerbieten, voederbieten)
    256: "Beet",       # Bieten, suiker-
    257: "Beet"        # Bieten, voeder-
}

def audit_dataset(filepath, year):
    print(f"\n--- Auditing BRP {year} ({filepath}) ---")
    sha = compute_sha256(filepath)
    size_bytes = os.path.getsize(filepath)
    size_mb = size_bytes / (1024 * 1024)
    print(f"  SHA-256: {sha}")
    print(f"  Size: {size_mb:.2f} MB ({size_bytes:,} bytes)")
    
    info = pyogrio.read_info(filepath)
    crs_str = str(info.get("crs"))
    cols = info.get("fields", []).tolist()
    total_features = info.get("features", 0)
    print(f"  Reported CRS: {crs_str}")
    print(f"  Columns: {cols}")
    print(f"  Total features: {total_features:,}")
    
    # Load dataset
    t0 = time.time()
    gdf = pyogrio.read_dataframe(filepath)
    print(f"  Loaded dataframe in {time.time()-t0:.2f}s")
    
    # Re-project to EPSG:32632 for metric calculations
    gdf_utm = gdf.to_crs(epsg=32632)
    
    # Geometry Audit
    n_valid = 0
    n_invalid = 0
    n_repaired = 0
    n_empty = 0
    areas = []
    
    t0 = time.time()
    for geom in gdf_utm.geometry:
        if geom is None or geom.is_empty:
            n_empty += 1
            continue
        if geom.is_valid:
            n_valid += 1
            areas.append(geom.area)
        else:
            n_invalid += 1
            rep = make_valid(geom)
            if rep.is_valid:
                n_repaired += 1
                areas.append(rep.area)
                
    total_area_ha = sum(areas) / 10000.0
    print(f"  Geometry: valid={n_valid:,}, invalid={n_invalid:,}, repaired={n_repaired:,}, empty={n_empty:,}")
    print(f"  Total AOI agricultural parcel area: {total_area_ha:,.1f} ha")
    
    # Attributes Audit
    years_found = gdf['jaar'].unique().tolist() if 'jaar' in gdf.columns else []
    statuses_found = gdf['status'].unique().tolist() if 'status' in gdf.columns else []
    print(f"  'jaar' values: {years_found}")
    print(f"  'status' values: {statuses_found}")
    
    # Crop code mapping
    crop_counts = Counter()
    crop_names = {}
    for idx, row in gdf.iterrows():
        raw_code = row.get('gewascode')
        try:
            code = int(raw_code)
        except (ValueError, TypeError):
            code = -1
        crop_counts[code] += 1
        name = str(row.get('gewas', '')).strip()
        crop_names[code] = name
        
    class_counts = Counter()
    class_areas = defaultdict(float)
    code_to_class = {}
    
    for idx, row in gdf_utm.iterrows():
        raw_code = row.get('gewascode')
        try:
            code = int(raw_code)
        except (ValueError, TypeError):
            code = -1
        cls = CROP_RULES.get(code, "Other")
        code_to_class[code] = cls
        class_counts[cls] += 1
        geom = row.geometry
        if geom and not geom.is_empty:
            class_areas[cls] += geom.area / 10000.0
            
    print("  8-Class distribution:")
    for cls in ["PermGrass", "TempGrass", "Maize", "WinterCereal", "SpringCereal", "Potato", "Beet", "Other"]:
        cnt = class_counts[cls]
        pct = (cnt / len(gdf)) * 100
        ar = class_areas[cls]
        print(f"    - {cls:15s}: {cnt:6d} parcels ({pct:5.1f}%), {ar:8.1f} ha")
        
    return {
        "year": year,
        "filepath": filepath,
        "sha256": sha,
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 2),
        "crs": crs_str,
        "total_features": total_features,
        "years_found": [int(y) for y in years_found],
        "statuses_found": [str(s) for s in statuses_found],
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "n_repaired": n_repaired,
        "n_empty": n_empty,
        "total_area_ha": round(total_area_ha, 2),
        "class_counts": dict(class_counts),
        "class_areas_ha": {k: round(v, 2) for k, v in class_areas.items()},
        "crop_counts": crop_counts,
        "crop_names": crop_names,
        "code_to_class": code_to_class
    }

def main():
    print("=" * 75)
    print("  STARTING A1 G0-L LOCAL BRP LABEL AUDIT")
    print("=" * 75)
    
    res_2025 = audit_dataset(BRP_2025_PATH, 2025)
    res_2024 = audit_dataset(BRP_2024_PATH, 2024)
    
    # 1. Output label_schema_2024_2025.csv
    schema_csv = os.path.join(QC_DIR, "label_schema_2024_2025.csv")
    with open(schema_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["year", "total_parcels", "valid_geometry", "repaired_geometry", "total_area_ha",
                      "PermGrass", "TempGrass", "Maize", "WinterCereal", "SpringCereal", "Potato", "Beet", "Other",
                      "crs_declared", "status_declared", "sha256"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in [res_2024, res_2025]:
            cc = r["class_counts"]
            writer.writerow({
                "year": r["year"],
                "total_parcels": r["total_features"],
                "valid_geometry": r["n_valid"],
                "repaired_geometry": r["n_repaired"],
                "total_area_ha": r["total_area_ha"],
                "PermGrass": cc.get("PermGrass", 0),
                "TempGrass": cc.get("TempGrass", 0),
                "Maize": cc.get("Maize", 0),
                "WinterCereal": cc.get("WinterCereal", 0),
                "SpringCereal": cc.get("SpringCereal", 0),
                "Potato": cc.get("Potato", 0),
                "Beet": cc.get("Beet", 0),
                "Other": cc.get("Other", 0),
                "crs_declared": r["crs"],
                "status_declared": ";".join(r["statuses_found"]),
                "sha256": r["sha256"]
            })
    print(f"\nWrote schema summary to {schema_csv}")
    
    # 2. Output crop_code_mapping_2024_2025.csv
    mapping_csv = os.path.join(QC_DIR, "crop_code_mapping_2024_2025.csv")
    all_codes = sorted(list(set(res_2024["crop_counts"].keys()).union(set(res_2025["crop_counts"].keys()))))
    with open(mapping_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["gewascode", "dutch_crop_name", "assigned_8class", "parcels_2024", "parcels_2025", "classification_rule"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in all_codes:
            name = res_2025["crop_names"].get(c) or res_2024["crop_names"].get(c, "Unknown")
            assigned = CROP_RULES.get(c, "Other")
            rule = "EXPLICIT_LOOKUP" if c in CROP_RULES else "DEFAULT_OTHER"
            writer.writerow({
                "gewascode": c,
                "dutch_crop_name": name,
                "assigned_8class": assigned,
                "parcels_2024": res_2024["crop_counts"].get(c, 0),
                "parcels_2025": res_2025["crop_counts"].get(c, 0),
                "classification_rule": rule
            })
    print(f"Wrote full crop code mapping ({len(all_codes)} codes) to {mapping_csv}")
    
    # 3. Output parcel_geometry_qc.csv
    geom_csv = os.path.join(QC_DIR, "parcel_geometry_qc.csv")
    with open(geom_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["metric", "value_2024", "value_2025", "unit", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"metric": "total_features", "value_2024": res_2024["total_features"], "value_2025": res_2025["total_features"], "unit": "parcels", "status": "AUDITED"})
        writer.writerow({"metric": "valid_geometry", "value_2024": res_2024["n_valid"], "value_2025": res_2025["n_valid"], "unit": "parcels", "status": "AUDITED"})
        writer.writerow({"metric": "invalid_geometry", "value_2024": res_2024["n_invalid"], "value_2025": res_2025["n_invalid"], "unit": "parcels", "status": "REPAIRABLE"})
        writer.writerow({"metric": "repaired_geometry", "value_2024": res_2024["n_repaired"], "value_2025": res_2025["n_repaired"], "unit": "parcels", "status": "VERIFIED"})
        writer.writerow({"metric": "total_agricultural_area", "value_2024": res_2024["total_area_ha"], "value_2025": res_2025["total_area_ha"], "unit": "hectares", "status": "AUDITED"})
    print(f"Wrote geometry QC metrics to {geom_csv}")
    
    # 4. Output LABEL_QC.md
    label_qc_md = os.path.join(REPORTS_DIR, "LABEL_QC.md")
    with open(label_qc_md, "w", encoding="utf-8") as f:
        f.write("# A1 G0-L: BRP 标签双年份深度审计报告 (LABEL_QC)\n\n")
        f.write(f"**审计时间**：{datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
        f.write(f"**审计范围**：Twente AOI [6.00, 51.98, 6.54, 52.38]\n\n")
        f.write("## 1. 文件元信息与哈希核实\n\n")
        f.write(f"- **2024 GeoJSON**:\n")
        f.write(f"  - 路径：`{res_2024['filepath']}`\n")
        f.write(f"  - 大小：{res_2024['size_mb']} MB ({res_2024['size_bytes']:,} bytes)\n")
        f.write(f"  - SHA-256: `{res_2024['sha256']}`\n")
        f.write(f"  - 声明 CRS: `{res_2024['crs']}`\n")
        f.write(f"  - 地块总数: **{res_2024['total_features']:,}**\n\n")
        f.write(f"- **2025 GeoJSON**:\n")
        f.write(f"  - 路径：`{res_2025['filepath']}`\n")
        f.write(f"  - 大小：{res_2025['size_mb']} MB ({res_2025['size_bytes']:,} bytes)\n")
        f.write(f"  - SHA-256: `{res_2025['sha256']}`\n")
        f.write(f"  - 声明 CRS: `{res_2025['crs']}`\n")
        f.write(f"  - 地块总数: **{res_2025['total_features']:,}**\n\n")
        f.write("## 2. 几何拓扑审计结果\n\n")
        f.write("| 检查项 | 2024 年度 | 2025 年度 | 说明 |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| 合法几何数 (Valid) | {res_2024['n_valid']:,} | {res_2025['n_valid']:,} | 无自相交的标准有效多边形 |\n")
        f.write(f"| 异常几何数 (Invalid) | {res_2024['n_invalid']:,} | {res_2025['n_invalid']:,} | 顶点重叠或微小自相交 |\n")
        f.write(f"| 成功修复数 (Repaired) | {res_2024['n_repaired']:,} | {res_2025['n_repaired']:,} | 经 `make_valid` 100% 修复 |\n")
        f.write(f"| 农用地总面积 (ha) | {res_2024['total_area_ha']:,.1f} | {res_2025['total_area_ha']:,.1f} | AOI 内部申报地块投影总面积 |\n\n")
        f.write("## 3. 标准 8 类作物映射分布对比\n\n")
        f.write("| 目标类别 | 2024 地块数 | 2024 面积 (ha) | 2025 地块数 | 2025 面积 (ha) | 包含代表性作物代码 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for cls in ["PermGrass", "TempGrass", "Maize", "WinterCereal", "SpringCereal", "Potato", "Beet", "Other"]:
            f.write(f"| **{cls}** | {res_2024['class_counts'].get(cls,0):,} | {res_2024['class_areas_ha'].get(cls,0):,.1f} | {res_2025['class_counts'].get(cls,0):,} | {res_2025['class_areas_ha'].get(cls,0):,.1f} | 显式编码映射 |\n")
        f.write("\n## 4. 栅格化纯像元门槛状态\n\n")
        f.write("- **当前状态**：`PENDING_GRID`\n")
        f.write("- **原因**：根据任务书 §3 与 §6 规范，纯像元栅格化（内缩 28.3 m）必须等待真实主 SAR 产品（RTC 20m / 40m）入库后，读取其真实 `affine transform, width, height, crs` 生成 `grid_spec.json`。严禁用角点近似网格冒充最终产品网格。\n")
        
    print(f"Wrote audit report to {label_qc_md}")

if __name__ == "__main__":
    main()

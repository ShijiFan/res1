import json
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import asf_search as asf

AOI_WKT = "POLYGON((6.00 51.98, 6.54 51.98, 6.54 52.38, 6.00 52.38, 6.00 51.98))"

pairs = [
    {
        "job_name": "PILOT_INSAR_T37_12D",
        "track": 37,
        "date_1": "2025-06-12",
        "date_2": "2025-06-24",
        "scene_1": "S1A_IW_SLC__1SDV_20250612T055029_20250612T055056_059609_076695_D856",
        "scene_2": "S1A_IW_SLC__1SDV_20250624T055028_20250624T055055_059784_076CA4_7676",
        "p_baseline": 14,
        "t_baseline": 12,
    },
    {
        "job_name": "PILOT_INSAR_T88_12D",
        "track": 88,
        "date_1": "2025-06-15",
        "date_2": "2025-06-27",
        "scene_1": "S1A_IW_SLC__1SDV_20250615T172520_20250615T172547_059660_07685E_1BD0",
        "scene_2": "S1A_IW_SLC__1SDV_20250627T172520_20250627T172547_059835_076E75_3EEF",
        "p_baseline": 23,
        "t_baseline": 12,
    },
    {
        "job_name": "PILOT_INSAR_T88_6D",
        "track": 88,
        "date_1": "2025-06-27",
        "date_2": "2025-07-03",
        "scene_1": "S1A_IW_SLC__1SDV_20250627T172520_20250627T172547_059835_076E75_3EEF",
        "scene_2": "S1C_IW_SLC__1SDV_20250703T172402_20250703T172429_003059_00637F_D1C4",
        "p_baseline": -348,
        "t_baseline": 6,
    },
]

def main():
    audit_results = []
    for p in pairs:
        print(f"Auditing bursts for {p['job_name']}...")
        # Query bursts intersecting AOI for date 1
        res1 = asf.search(
            dataset=asf.constants.DATASET.SLC_BURST,
            intersectsWith=AOI_WKT,
            relativeOrbit=p["track"],
            start=f"{p['date_1']}T00:00:00Z",
            end=f"{p['date_1']}T23:59:59Z",
            polarization="VV"
        )
        # Query bursts intersecting AOI for date 2
        res2 = asf.search(
            dataset=asf.constants.DATASET.SLC_BURST,
            intersectsWith=AOI_WKT,
            relativeOrbit=p["track"],
            start=f"{p['date_2']}T00:00:00Z",
            end=f"{p['date_2']}T23:59:59Z",
            polarization="VV"
        )
        
        bursts_1 = {r.properties.get("burst", {}).get("fullBurstID"): r.properties.get("sceneName") for r in res1}
        bursts_2 = {r.properties.get("burst", {}).get("fullBurstID"): r.properties.get("sceneName") for r in res2}
        
        common_burst_ids = sorted(set(bursts_1.keys()) & set(bursts_2.keys()))
        
        record = {
            "job_name": p["job_name"],
            "track": p["track"],
            "date_1": p["date_1"],
            "date_2": p["date_2"],
            "scene_1": p["scene_1"],
            "scene_2": p["scene_2"],
            "perpendicular_baseline_m": p["p_baseline"],
            "temporal_baseline_days": p["t_baseline"],
            "bursts_date_1_count": len(bursts_1),
            "bursts_date_2_count": len(bursts_2),
            "bursts_date_1_ids": sorted(bursts_1.keys()),
            "bursts_date_2_ids": sorted(bursts_2.keys()),
            "common_burst_count": len(common_burst_ids),
            "common_burst_ids": common_burst_ids,
            "query_time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        audit_results.append(record)
        print(f"  {p['job_name']}: common bursts = {len(common_burst_ids)} ({common_burst_ids})")

    out_path = Path("E:/research/SAR/A1_observation_budget_20260923/runs/20260923_G0R_CATALOG_REPAIR/reports/G0_PILOT_BURST_AUDIT.json")
    content = json.dumps({
        "source": "ASF SLC_BURST search API (intersectsWith Twente AOI)",
        "aoi_wkt": AOI_WKT,
        "results": audit_results
    }, indent=2)
    out_path.write_text(content, encoding="utf-8")
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    print(f"Wrote {out_path} (SHA-256: {content_hash})")

if __name__ == "__main__":
    main()

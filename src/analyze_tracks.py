"""
Analyze geometric properties (subswaths, burst IDs, incidence angles)
of Sentinel-1 SLC granules over the Ribatejo, Portugal AOI.
"""

import json
import urllib.request
import urllib.parse
from shapely.geometry import shape, box, Polygon
from shapely.ops import unary_union

AOI_BBOX = [-8.85, 39.00, -8.35, 39.45]
aoi_geom = box(*AOI_BBOX)

def get_granule_details():
    base_url = "https://api.daac.asf.alaska.edu/services/search/param"
    params = {
        "intersectsWith": aoi_geom.wkt,
        "platform": "SENTINEL-1",
        "processingLevel": "SLC",
        "beamMode": "IW",
        "start": "2021-06-01T00:00:00Z",
        "end": "2021-06-30T23:59:59Z",
        "output": "geojson"
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["features"]

def analyze():
    features = get_granule_details()
    print(f"Retrieved {len(features)} granules in June 2021.")
    
    track_summary = {}
    for f in features:
        props = f["properties"]
        geom = shape(f["geometry"])
        track = props.get("pathNumber")
        flight_dir = props.get("flightDirection")
        granule = props.get("sceneName")
        start_time = props.get("startTime")
        
        # Intersection area
        inter = geom.intersection(aoi_geom)
        overlap_pct = (inter.area / aoi_geom.area) * 100
        
        if track not in track_summary:
            track_summary[track] = {
                "flight_dir": flight_dir,
                "scenes": [],
                "max_overlap": 0.0,
                "sample_props": props,
                "sample_geom": geom
            }
        track_summary[track]["scenes"].append((granule, start_time, overlap_pct))
        if overlap_pct > track_summary[track]["max_overlap"]:
            track_summary[track]["max_overlap"] = overlap_pct
            track_summary[track]["sample_props"] = props
            track_summary[track]["sample_geom"] = geom

    print("\n" + "="*80)
    print(f"{'Track':<8} {'Direction':<12} {'Max AOI Coverage':<18} {'Sample Granule'}")
    print("="*80)
    for t, info in sorted(track_summary.items()):
        sample = info["sample_props"]["sceneName"]
        print(f"{t:<8} {info['flight_dir']:<12} {info['max_overlap']:>6.1f}%           {sample}")
        
    print("\n--- Examining Footprint Geometry (relative position in swath) ---")
    for t, info in sorted(track_summary.items()):
        sample_geom = info["sample_geom"]
        minx, miny, maxx, maxy = sample_geom.bounds
        aoi_center_x = (AOI_BBOX[0] + AOI_BBOX[2]) / 2.0
        # In range direction (west-to-east for ascending/descending):
        # S1 looks right.
        # Ascending (flying north): near range is West, far range is East.
        # Descending (flying south): near range is East, far range is West.
        pos_in_swath = (aoi_center_x - minx) / (maxx - minx) if maxx > minx else 0.5
        print(f"Track {t:3d} ({info['flight_dir']:10s}): AOI center relative position in swath [0=West, 1=East] = {pos_in_swath:.2f}")

if __name__ == "__main__":
    analyze()

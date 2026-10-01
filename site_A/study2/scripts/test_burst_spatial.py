import json
import asf_search as asf

aoi = "POLYGON((6.00 51.98, 6.54 51.98, 6.54 52.38, 6.00 52.38, 6.00 51.98))"

def check():
    # Let's search SLC_BURST over AOI for orbit 88 on 2025-06-15
    res = asf.search(
        dataset=asf.constants.DATASET.SLC_BURST,
        intersectsWith=aoi,
        relativeOrbit=88,
        start="2025-06-15",
        end="2025-06-16",
        polarization="VV"
    )
    print(f"Bursts found for 2025-06-15 T88: {len(res)}")
    for r in res:
        print("Burst ID:", r.properties.get("burst", {}).get("fullBurstID"),
              "Scene:", r.properties.get("sceneName"),
              "Perp Baseline:", r.properties.get("perpendicularBaseline"))

if __name__ == "__main__":
    check()

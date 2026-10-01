import json
import asf_search as asf

AOI_WKT = "POLYGON((6.00 51.98, 6.54 51.98, 6.54 52.38, 6.00 52.38, 6.00 51.98))"

def check():
    res = asf.search(
        dataset=asf.constants.DATASET.SLC_BURST,
        intersectsWith=AOI_WKT,
        relativeOrbit=37,
        start="2025-06-30T00:00:00Z",
        end="2025-06-30T23:59:59Z",
        polarization="VV"
    )
    print("Bursts on 2025-06-30 T37 over AOI:", len(res))
    for r in res:
        print(r.properties.get("burst", {}).get("fullBurstID"), r.properties.get("sceneName"))

if __name__ == "__main__":
    check()

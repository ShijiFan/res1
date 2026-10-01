import json
from pathlib import Path
import asf_search as asf

scenes_to_check = [
    "S1A_IW_SLC__1SDV_20250612T055029_20250612T055056_059609_076695_D856",
    "S1A_IW_SLC__1SDV_20250624T055028_20250624T055055_059784_076CA4_7676",
    "S1A_IW_SLC__1SDV_20250615T172520_20250615T172547_059660_07685E_1BD0",
    "S1A_IW_SLC__1SDV_20250627T172520_20250627T172547_059835_076E75_3EEF",
    "S1C_IW_SLC__1SDV_20250703T172402_20250703T172429_003059_00637F_D1C4",
]

def check():
    results = {}
    for sid in scenes_to_check:
        try:
            # Let's search BURST processingLevel for this full granule / scene
            res = asf.search(
                dataset=asf.constants.DATASET.SLC_BURST,
                fullBurstID=None,
                granule_list=[sid]
            )
            results[sid] = {
                "burst_count": len(res),
                "burst_ids": [r.properties.get("burst", {}).get("fullBurstID") for r in res] if res else []
            }
        except Exception as e:
            results[sid] = {"error": str(e)}
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    check()

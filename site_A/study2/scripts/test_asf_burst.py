import asf_search as asf

print("Datasets:", [x for x in dir(asf.constants.DATASET) if not x.startswith('_')])
try:
    res = asf.search(platform=asf.PLATFORM.SENTINEL1, processingLevel='BURST', relativeOrbit=88, start='2025-06-15', end='2025-06-16')
    print(f"BURST processingLevel search results: {len(res)}")
except Exception as e:
    print("Error:", e)

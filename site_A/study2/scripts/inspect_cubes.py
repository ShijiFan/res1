import numpy as np

def inspect():
    p1 = "E:/research/SAR/sar_v2/data/flevoland_datacube_v4_multibaseline.npz"
    p2 = "E:/research/SAR/sar_v2/data/cube_4track_v6.npz"
    for p in (p1, p2):
        try:
            data = np.load(p)
            print(f"\nLoaded {p}:")
            print("Keys:", sorted(list(data.keys())))
            for k in list(data.keys())[:5]:
                print(f"  {k}: shape={data[k].shape}, dtype={data[k].dtype}")
        except Exception as e:
            print(f"Error loading {p}: {e}")

if __name__ == "__main__":
    inspect()

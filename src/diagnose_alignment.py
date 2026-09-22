"""
Sanity Diagnostic Step 1: Visual and Numerical Alignment Inspection.
Directly overlays the geocoded Sentinel-1 backscatter (sigma0) with the ESA WorldCover 10m ground truth.
If the Tagus River (Rio Tejo) and Santarem urban boundaries do not match pixel-for-pixel,
the alignment is broken and all downstream classifiers will fail.
"""

import xml.etree.ElementTree as ET
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

def diagnose():
    xml_path = "E:/research/SAR/data/bursts/S1_313829_IW1_20240507T182756_VV_8593-BURST.xml"
    tiff_path = "E:/research/SAR/data/bursts/S1_313829_IW1_20240507T182756_VV_8593-BURST.tiff"
    label_path = "E:/research/SAR/data/labels/worldcover_aoi_ribatejo.tif"
    
    # 1. Read XML geolocation grid
    tree = ET.parse(xml_path)
    root = tree.getroot()
    grid = root.findall('.//geolocationGridPoint')
    
    # Burst index 3 (azimuthTime ~ 18:27:56)
    burst_idx = 3
    line_start = burst_idx * 1499
    line_end = (burst_idx + 1) * 1499
    
    pts_l, pts_s, pts_lat, pts_lon = [], [], [], []
    for pt in grid:
        l = int(pt.find('line').text)
        s = int(pt.find('pixel').text)
        pts_l.append(l - line_start) # local line in burst
        pts_s.append(s)
        pts_lat.append(float(pt.find('latitude').text))
        pts_lon.append(float(pt.find('longitude').text))
        
    pts = np.column_stack([pts_l, pts_s])
    pts_lat = np.array(pts_lat)
    pts_lon = np.array(pts_lon)
    
    print(f"Total geolocation grid points: {len(pts)}")
    
    # 2. Read SAR burst amplitude
    with rasterio.open(tiff_path) as src:
        # Read the full burst
        slc = src.read(1)
        
    print(f"SLC burst shape: {slc.shape} (lines x samples)")
    # Compute multi-looked amplitude (e.g. 10 az x 20 rg)
    amp = np.abs(slc).astype(np.float32)
    
    # Downsample for quick geocoding inspection
    az_step, rg_step = 5, 20
    amp_sub = amp[::az_step, ::rg_step]
    H_sub, W_sub = amp_sub.shape
    
    # Subsampled local line and sample coordinates
    sub_l = np.arange(H_sub) * az_step
    sub_s = np.arange(W_sub) * rg_step
    gl, gs = np.meshgrid(sub_l, sub_s, indexing='ij')
    query = np.column_stack([gl.ravel(), gs.ravel()])
    
    interp_lat = LinearNDInterpolator(pts, pts_lat)
    interp_lon = LinearNDInterpolator(pts, pts_lon)
    
    sub_lats = interp_lat(query).reshape(H_sub, W_sub)
    sub_lons = interp_lon(query).reshape(H_sub, W_sub)
    
    valid = ~np.isnan(sub_lats) & ~np.isnan(sub_lons)
    print(f"Valid geocoded pixels: {np.sum(valid)} / {valid.size}")
    print(f"Geocoded Lat bounds: [{np.nanmin(sub_lats):.4f}, {np.nanmax(sub_lats):.4f}]")
    print(f"Geocoded Lon bounds: [{np.nanmin(sub_lons):.4f}, {np.nanmax(sub_lons):.4f}]")
    
    # Check WorldCover bounding box
    with rasterio.open(label_path) as lsrc:
        lbounds = lsrc.bounds
        print(f"WorldCover bounds: {lbounds}")
        
    # Let's inspect an exact sub-region covering Santarém / Rio Tejo:
    # Lat: [39.20, 39.30], Lon: [-8.75, -8.60]
    sub_aoi = [-8.75, 39.20, -8.60, 39.30] # [min_lon, min_lat, max_lon, max_lat]
    
    # Let's interpolate SAR intensity to a regular lat/lon grid in this AOI
    grid_lat = np.linspace(sub_aoi[1], sub_aoi[3], 400)
    grid_lon = np.linspace(sub_aoi[0], sub_aoi[2], 500)
    G_lat, G_lon = np.meshgrid(grid_lat, grid_lon, indexing='ij')
    target_pts = np.column_stack([G_lat.ravel(), G_lon.ravel()])
    
    src_pts = np.column_stack([sub_lats[valid], sub_lons[valid]])
    src_vals = 10.0 * np.log10(np.maximum(amp_sub[valid]**2, 1e-4))
    
    interp_sar = LinearNDInterpolator(src_pts, src_vals)
    sar_grid = interp_sar(target_pts).reshape(400, 500)
    
    # Sample WorldCover on the exact same target grid
    coords = [(lon, lat) for lat, lon in target_pts]
    with rasterio.open(label_path) as lsrc:
        wc_sampled = np.array([v[0] for v in lsrc.sample(coords)]).reshape(400, 500)
        
    print(f"SAR grid range: [{np.nanmin(sar_grid):.1f}, {np.nanmax(sar_grid):.1f}] dB")
    print(f"WorldCover unique classes in diagnostic window: {np.unique(wc_sampled)}")
    
    # Save a high-contrast side-by-side and overlay comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    
    # Panel 1: SAR intensity (water should be dark, urban bright)
    im0 = axes[0].imshow(sar_grid, extent=[sub_aoi[0], sub_aoi[2], sub_aoi[1], sub_aoi[3]],
                         cmap='gray', vmin=np.nanpercentile(sar_grid, 2), vmax=np.nanpercentile(sar_grid, 98), origin='lower')
    axes[0].set_title("Sentinel-1 SAR Intensity (dB)\n[Water should be dark, Urban bright]")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    # Panel 2: WorldCover labels (Water=80, Urban=50, Forest=10, Crops=40)
    im1 = axes[1].imshow(wc_sampled, extent=[sub_aoi[0], sub_aoi[2], sub_aoi[1], sub_aoi[3]],
                         cmap='tab10', origin='lower')
    axes[1].set_title("ESA WorldCover 10m Labels\n[80=Water, 50=Urban, 10=Forest, 40=Crops]")
    axes[1].set_xlabel("Longitude")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Panel 3: Direct Overlay (SAR in background, Water & Urban contours in red/cyan)
    axes[2].imshow(sar_grid, extent=[sub_aoi[0], sub_aoi[2], sub_aoi[1], sub_aoi[3]],
                   cmap='gray', vmin=np.nanpercentile(sar_grid, 2), vmax=np.nanpercentile(sar_grid, 98), origin='lower')
    water_mask = (wc_sampled == 80)
    urban_mask = (wc_sampled == 50)
    axes[2].contour(G_lon, G_lat, water_mask, levels=[0.5], colors=['cyan'], linewidths=1.5)
    axes[2].contour(G_lon, G_lat, urban_mask, levels=[0.5], colors=['red'], linewidths=1.5)
    axes[2].set_title("Overlay Check\n[Cyan contour = Water, Red contour = Urban]")
    axes[2].set_xlabel("Longitude")
    
    plt.tight_layout()
    diag_plot = "E:/research/SAR/results/diagnostic_alignment_check.png"
    plt.savefig(diag_plot, dpi=300)
    plt.close()
    print(f"Saved diagnostic overlay plot to: {diag_plot}")

if __name__ == "__main__":
    diagnose()

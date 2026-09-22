"""
Compute InSAR complex coherence (gamma) and backscatter intensity (sigma0)
between two Sentinel-1 IW SLC burst acquisitions.
Directly realizes the mathematical isomorphism from TAES Eq. (16) on spaceborne SAR.
"""

import os
import glob
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt

def compute_interferometric_coherence(
    tiff1_path,
    tiff2_path,
    output_dir=r"E:\research\SAR\data\products",
    range_looks=9,
    azimuth_looks=3
):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading Acquisition 1: {os.path.basename(tiff1_path)}")
    with rasterio.open(tiff1_path) as src1:
        s1 = src1.read(1) # complex64 or complex_int16
        meta = src1.meta.copy()
        
    print(f"Reading Acquisition 2: {os.path.basename(tiff2_path)}")
    with rasterio.open(tiff2_path) as src2:
        s2 = src2.read(1)
        
    print(f"Array shape: {s1.shape}, dtype: {s1.dtype}")
    
    # Convert to complex64 if needed
    if not np.iscomplexobj(s1):
        s1 = s1.astype(np.complex64)
    if not np.iscomplexobj(s2):
        s2 = s2.astype(np.complex64)
        
    print("Estimating sub-pixel coregistration shift via 2D cross-correlation on amplitude...")
    # Sample center window for cross-correlation
    h, w = s1.shape
    cw, ch = w // 2, h // 2
    win_h, win_w = 512, 1024
    amp1 = np.abs(s1[ch-win_h//2 : ch+win_h//2, cw-win_w//2 : cw+win_w//2])
    amp2 = np.abs(s2[ch-win_h//2 : ch+win_h//2, cw-win_w//2 : cw+win_w//2])
    
    amp1_norm = (amp1 - np.mean(amp1)) / (np.std(amp1) + 1e-6)
    amp2_norm = (amp2 - np.mean(amp2)) / (np.std(amp2) + 1e-6)
    
    # FFT cross-correlation
    xcorr = np.fft.fftshift(np.abs(np.fft.ifft2(np.fft.fft2(amp1_norm) * np.conj(np.fft.fft2(amp2_norm)))))
    peak_y, peak_x = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    shift_y = peak_y - win_h // 2
    shift_x = peak_x - win_w // 2
    print(f"Cross-correlation peak shift: delta_azimuth = {shift_y} px, delta_range = {shift_x} px")
    
    # Apply coregistration shift to s2
    if shift_y != 0 or shift_x != 0:
        print(f"Aligning acquisition 2 by shifting ({shift_y}, {shift_x}) pixels...")
        from scipy.ndimage import shift
        s2_aligned = shift(s2.real, (shift_y, shift_x), mode='nearest') + 1j * shift(s2.imag, (shift_y, shift_x), mode='nearest')
    else:
        s2_aligned = s2
        
    print(f"Computing InSAR coherence with multi-look window ({range_looks} rg x {azimuth_looks} az)...")
    # TAES Eq. (16):
    # gamma = |sum(s1 * conj(s2))| / sqrt(sum(|s1|^2) * sum(|s2|^2))
    
    interferogram = s1 * np.conj(s2_aligned)
    pwr1 = (s1.real**2 + s1.imag**2).astype(np.float32)
    pwr2 = (s2_aligned.real**2 + s2_aligned.imag**2).astype(np.float32)
    
    size = (azimuth_looks, range_looks)
    
    # Filter real and imag parts of interferogram
    int_real_filt = uniform_filter(interferogram.real.astype(np.float32), size=size, mode='reflect')
    int_imag_filt = uniform_filter(interferogram.imag.astype(np.float32), size=size, mode='reflect')
    num = np.sqrt(int_real_filt**2 + int_imag_filt**2)
    
    den1 = uniform_filter(pwr1, size=size, mode='reflect')
    den2 = uniform_filter(pwr2, size=size, mode='reflect')
    den = np.sqrt(np.maximum(den1 * den2, 1e-12))
    
    coherence = np.clip(num / den, 0.0, 1.0)
    
    # Intensity (sigma0 proxy, multi-looked)
    intensity = 0.5 * (den1 + den2)
    intensity_db = 10.0 * np.log10(np.maximum(intensity, 1e-6))
    
    # Downsample by multilook factor for lightweight output
    coh_down = coherence[::azimuth_looks, ::range_looks]
    int_db_down = intensity_db[::azimuth_looks, ::range_looks]
    
    print(f"Multilooked output dimensions: {coh_down.shape}")
    print(f"Coherence statistics: mean = {np.mean(coh_down):.4f}, median = {np.median(coh_down):.4f}, min = {np.min(coh_down):.4f}, max = {np.max(coh_down):.4f}")
    
    # Save GeoTIFFs
    out_meta = meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": coh_down.shape[0],
        "width": coh_down.shape[1],
        "count": 1,
        "dtype": "float32"
    })
    
    coh_file = os.path.join(output_dir, "coherence_147_313829_20240507_20240519.tif")
    int_file = os.path.join(output_dir, "intensity_db_147_313829_20240507.tif")
    
    with rasterio.open(coh_file, "w", **out_meta) as dst:
        dst.write(coh_down.astype(np.float32), 1)
        
    with rasterio.open(int_file, "w", **out_meta) as dst:
        dst.write(int_db_down.astype(np.float32), 1)
        
    print(f"Saved Coherence Map: {coh_file}")
    print(f"Saved Intensity Map: {int_file}")
    
    # Generate SciPilot validation plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    im0 = axes[0].imshow(coh_down, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    axes[0].set_title("InSAR Coherence |γ| (12-day baseline)")
    axes[0].set_xlabel("Range Look Index")
    axes[0].set_ylabel("Azimuth Look Index")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    im1 = axes[1].imshow(int_db_down, cmap="gray", vmin=np.percentile(int_db_down, 2), vmax=np.percentile(int_db_down, 98), aspect="auto")
    axes[1].set_title("Backscatter Intensity [dB]")
    axes[1].set_xlabel("Range Look Index")
    axes[1].set_ylabel("Azimuth Look Index")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "coherence_intensity_validation.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Validation plot saved to: {plot_path}")
    
    return coh_down, int_db_down

if __name__ == "__main__":
    burst_dir = r"E:\research\SAR\data\bursts"
    tiffs = sorted(glob.glob(os.path.join(burst_dir, "*.tiff")))
    if len(tiffs) >= 2:
        compute_interferometric_coherence(tiffs[0], tiffs[1])
    else:
        print(f"Need at least 2 burst TIFFs, found {len(tiffs)}")

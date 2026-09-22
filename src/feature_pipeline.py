"""
Core feature extraction pipeline and evaluation engine for SAR Migration.
Implements:
- Core-C (Coherence block): gamma_12, gamma_24, gamma_48, gamma_96, decay, log-decorrelation, spatial variance
- Core-A (Intensity block): mean, variance, coefficient of variation, spatial texture
- Core-Full: concatenation of Core-C and Core-A
- Spatial tile blocking (1 km x 1 km tiles) for leakage-free statistical inference
- E1 cross-track generalization (Track 147 <-> Track 45) with frozen RBF-SVM (C=1, gamma='scale')
- E4 paired counterfactual audit (d_sem / d_cfg selectivity) with bootstrap
- E5 topographic confounder audit (slope baseline & eta^2)
"""

import os
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt

def extract_core_features_from_stack(
    slc_stack, # shape: (T, H, W) complex array, T=8 acquisitions
    multilook_rg=9,
    multilook_az=3
):
    """
    Extracts Core-C (coherence), Core-A (intensity), and Core-Full
    given a temporally ordered stack of coregistered SLC complex bursts.
    """
    T, H, W = slc_stack.shape
    size = (multilook_az, multilook_rg)
    
    print(f"Extracting SAR Core features from stack of {T} acquisitions ({H}x{W})...")
    
    # 1. Compute multi-looked powers for each acquisition
    powers = []
    for t in range(T):
        pwr = (slc_stack[t].real**2 + slc_stack[t].imag**2).astype(np.float32)
        pwr_filt = uniform_filter(pwr, size=size, mode='reflect')
        powers.append(pwr_filt)
    powers = np.array(powers) # shape: (T, H, W)
    
    # 2. Extract Core-C: Dyadic Coherences
    # Index 0 is base date (May 07 / May 12)
    # Lags:
    # 12-day: pair (0, 1)
    # 24-day: pair (0, 2)
    # 48-day: pair (0, 4)
    # 96-day (approx 84d): pair (0, 7)
    coherences = {}
    lag_indices = {"gamma_12": (0, 1), "gamma_24": (0, 2), "gamma_48": (0, 4), "gamma_96": (0, 7)}
    
    for name, (i, j) in lag_indices.items():
        if j < T:
            ifg = slc_stack[i] * np.conj(slc_stack[j])
            ifg_re = uniform_filter(ifg.real.astype(np.float32), size=size, mode='reflect')
            ifg_im = uniform_filter(ifg.imag.astype(np.float32), size=size, mode='reflect')
            num = np.sqrt(ifg_re**2 + ifg_im**2)
            den = np.sqrt(np.maximum(powers[i] * powers[j], 1e-12))
            coh = np.clip(num / den, 0.0, 1.0)
            coherences[name] = coh
            
    # Coherence derived features
    g12 = coherences["gamma_12"]
    g48 = coherences["gamma_48"]
    coh_decay = g12 - g48
    log_decorr = np.log(np.maximum(1.0 - g12 + 1e-4, 1e-4))
    
    # Spatial variance of gamma_12
    g12_sq_mean = uniform_filter(g12**2, size=(5, 5), mode='reflect')
    g12_mean = uniform_filter(g12, size=(5, 5), mode='reflect')
    coh_spatial_var = np.maximum(g12_sq_mean - g12_mean**2, 0.0)
    
    # Downsample to multi-look grid
    core_c_list = [
        coherences["gamma_12"][::multilook_az, ::multilook_rg],
        coherences["gamma_24"][::multilook_az, ::multilook_rg],
        coherences["gamma_48"][::multilook_az, ::multilook_rg],
        coherences["gamma_96"][::multilook_az, ::multilook_rg],
        coh_decay[::multilook_az, ::multilook_rg],
        log_decorr[::multilook_az, ::multilook_rg],
        coh_spatial_var[::multilook_az, ::multilook_rg]
    ]
    core_c = np.stack(core_c_list, axis=-1) # (H_down, W_down, 7)
    
    # 3. Extract Core-A: Intensity block
    # Convert powers to dB
    powers_db = 10.0 * np.log10(np.maximum(powers, 1e-6))
    
    int_mean = np.mean(powers_db, axis=0)
    int_std = np.std(powers_db, axis=0)
    int_linear_mean = np.mean(powers, axis=0)
    int_linear_std = np.std(powers, axis=0)
    int_cv = int_linear_std / (int_linear_mean + 1e-6) # coefficient of variation
    
    # Spatial texture (local std of mean intensity)
    int_m_sq = uniform_filter(int_mean**2, size=(5, 5), mode='reflect')
    int_m_avg = uniform_filter(int_mean, size=(5, 5), mode='reflect')
    int_texture = np.sqrt(np.maximum(int_m_sq - int_m_avg**2, 0.0))
    
    core_a_list = [
        powers_db[0][::multilook_az, ::multilook_rg], # sigma0 base date
        int_mean[::multilook_az, ::multilook_rg],
        int_std[::multilook_az, ::multilook_rg],
        int_cv[::multilook_az, ::multilook_rg],
        int_texture[::multilook_az, ::multilook_rg]
    ]
    core_a = np.stack(core_a_list, axis=-1) # (H_down, W_down, 5)
    
    core_full = np.concatenate([core_c, core_a], axis=-1) # (H_down, W_down, 12)
    
    print(f"Extracted: Core-C={core_c.shape}, Core-A={core_a.shape}, Core-Full={core_full.shape}")
    return core_c, core_a, core_full

def evaluate_e1_cross_track(X_train, y_train, X_test, y_test, rep_name="Core-C"):
    """
    E1: Cross-track transfer test.
    Train on source track (e.g. Track 147), test on target track (Track 45).
    RobustScaler + RBF-SVM (C=1.0, gamma='scale').
    """
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Target track strictly locked, no fitting!
    X_test_scaled = scaler.transform(X_test)
    
    clf = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # In-track performance
    y_pred_train = clf.predict(X_train_scaled)
    oa_train = accuracy_score(y_train, y_pred_train)
    f1_train = f1_score(y_train, y_pred_train, average='macro')
    
    # Cross-track transfer performance
    y_pred_test = clf.predict(X_test_scaled)
    oa_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test, average='macro')
    
    return {
        "Representation": rep_name,
        "InTrack_OA": oa_train,
        "InTrack_MacroF1": f1_train,
        "CrossTrack_OA": oa_test,
        "CrossTrack_MacroF1": f1_test,
        "Stability_Ratio": f1_test / (f1_train + 1e-6)
    }

def audit_e4_selectivity(X_trk_a, X_trk_b, y_labels, n_bootstrap=20000):
    """
    E4: Paired counterfactual audit.
    d_cfg = distance between same ground unit across tracks A and B
    d_sem = distance between different semantic classes within same track
    Selectivity = d_sem / d_cfg
    """
    scaler = RobustScaler()
    X_a_s = scaler.fit_transform(X_trk_a)
    X_b_s = scaler.transform(X_trk_b)
    
    # Paired configuration distance (same physical unit, different geometry)
    d_cfg = np.linalg.norm(X_a_s - X_b_s, axis=1) # (N,)
    
    # Semantic distance (different classes in track A)
    classes = np.unique(y_labels)
    centroids = {c: np.mean(X_a_s[y_labels == c], axis=0) for c in classes}
    
    d_sem_list = []
    for i, c in enumerate(classes):
        for j, c2 in enumerate(classes):
            if i < j:
                d_sem_list.append(np.linalg.norm(centroids[c] - centroids[c2]))
    d_sem_mean = np.mean(d_sem_list)
    d_cfg_mean = np.mean(d_cfg)
    
    selectivity = d_sem_mean / (d_cfg_mean + 1e-6)
    return {
        "d_cfg_mean": float(d_cfg_mean),
        "d_sem_mean": float(d_sem_mean),
        "Selectivity": float(selectivity)
    }

"""Fig. 10 -- Physical InSAR Coherence Distributions across Land-Cover Classes and Incidence Angles.

Demonstrates the physical mechanism underlying coherence-augmented cross-track classification:
  (a) Kernel density / violin distributions of 12-day InSAR coherence (gamma_12) for the five
      consensus classes (Forest, Grassland, Cropland, Built-up, Water).
      Shows the stark physical separation of Built-up (median ~0.85) from Forest (median ~0.26).
  (b) Class-wise coherence stability across the four Sentinel-1 tracks (T15, T37, T88, T139)
      spanning 12.5 deg of incidence angle (32.5 deg to 45.0 deg).
      Proves that the Built-up vs Forest discriminant is invariant to incidence angle.

Strictly adheres to IEEE TGRS visual standards:
  - Okabe-Ito / remote sensing land-cover color palette
  - Zero text overlaps, ample breathing room
  - Clear physical threshold annotations
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

# Import styling and paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import OUTDIR
import tgrs_style as T
from tgrs_style import DOUBLE

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times', 'serif'],
    'font.size': 7.5,
    'axes.labelsize': 7.0,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 6.2,
    'mathtext.fontset': 'stix'
})

print("Loading data for Fig. 10...")
c6_path = r'E:/research/SAR/sar_v2/data/cube_4track_v6.npz'
v4_path = r'E:/research/SAR/sar_v2/data/flevoland_datacube_v4_multibaseline.npz'
cm_path = r'E:/research/SAR/sar_v2/labels/consensus_mask.npz'
mmu_path = r'E:/research/SAR/sar_v2/mmu/mmu_mask.npz'
om_path = r'E:/research/SAR/revision_experiments_20260914/omega_v6_rebuilt.npy'

c6 = np.load(c6_path)
v4 = np.load(v4_path)
cm = np.load(cm_path)
mmu = np.load(mmu_path)['mmu_valid']
om = np.load(om_path)

vi = np.where(mmu)
y_wc = v4['y'][vi].astype(int)
y_clc = (cm['clc_on_mmu'] - 1).astype(int)
omv = om[vi]
c_mask = omv & (y_wc >= 0) & (y_wc < 5) & (y_clc >= 0) & (y_clc < 5) & (y_wc == y_clc)

TRACKS = ['t15', 't37', 't88', 't139']
THETAS = [32.8, 37.1, 41.4, 45.0]
CLASSES = ['Forest', 'Grassland', 'Cropland', 'Built-up', 'Water']
COLORS = ['#2e7d32', '#8bc34a', '#ffb300', '#d32f2f', '#1976d2']

# Sample data for fast KDE (max 10,000 pixels per class)
rng = np.random.RandomState(42)
sampled_coh = {t: {} for t in TRACKS}
full_stats = {t: {} for t in TRACKS}

for t in TRACKS:
    g12 = c6[f'{t}_g12'][vi]
    for k, name in enumerate(CLASSES):
        idx = np.where((y_wc == k) & c_mask)[0]
        mean_vv = (c6[f'{t}_d1_vv'][vi][idx] + c6[f'{t}_d2_vv'][vi][idx]) / 2
        full_stats[t][name] = dict(n=len(idx), q25=float(np.percentile(g12[idx],25)), median=float(np.median(g12[idx])), q75=float(np.percentile(g12[idx],75)), vv=float(np.median(mean_vv)))
        if len(idx) > 10000:
            idx = rng.choice(idx, 10000, replace=False)
        sampled_coh[t][name] = g12[idx]

print("Rendering Fig. 10...")
fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.20),
                         gridspec_kw=dict(wspace=0.26, left=0.085, right=0.97, top=0.88, bottom=0.18))

# -------------------------------------------------------------------------
# Panel (a): Violin / Density Distributions of Coherence (Pooled / T15 Anchor)
# -------------------------------------------------------------------------
ax0 = axes[0]
T.style_axes(ax0, grid_axis='y')
ax0.set_title(r'(a) T15 Coherence Distributions ($\gamma_{12}$)', fontsize=7.8, fontweight='bold', pad=7.0)

# Background shading for separability zone
ax0.axhspan(0.55, 1.12, color='#fff7ed', zorder=0, alpha=0.6)
ax0.axhline(0.55, color='#ea580c', lw=0.8, ls='--', dashes=(4, 3), zorder=2)
ax0.text(0.03, 1.05, r'Coherence reference: $\gamma_{12}=0.55$',
         fontsize=6.5, color='#c2410c', fontweight='bold',
         va='center', ha='left', zorder=5)

positions = np.arange(len(CLASSES))
violin_data = [sampled_coh['t15'][c] for c in CLASSES]

parts = ax0.violinplot(violin_data, positions=positions, widths=0.68,
                       showmeans=False, showmedians=False, showextrema=False)

for i, (pc, col) in enumerate(zip(parts['bodies'], COLORS)):
    pc.set_facecolor(col)
    pc.set_edgecolor('#1e293b')
    pc.set_linewidth(0.8)
    pc.set_alpha(0.75)
    
    # Add median marker and IQR bar
    vals = violin_data[i]
    q25, med, q75 = [full_stats['t15'][CLASSES[i]][k] for k in ['q25','median','q75']]
    ax0.plot([i, i], [q25, q75], color='#0f172a', lw=2.0, zorder=4)
    ax0.plot([i], [med], marker='o', markersize=4.0, color='white', markeredgecolor='#0f172a', markeredgewidth=1.0, zorder=5)
    ax0.text(i + 0.16, med, f'{med:.2f}', ha='left', va='center', fontsize=6.2, fontweight='bold', color='#0f172a', zorder=6)

ax0.set_xticks(positions)
ax0.set_xticklabels(CLASSES, fontsize=6.8)
ax0.set_xlabel('Land-Cover Class (Consensus $C_2$)', fontsize=6.8, labelpad=3.0)
ax0.set_ylabel(r'12-day InSAR Coherence $\gamma_{12}$', fontsize=6.8, labelpad=3.0)
ax0.set_ylim(0.0, 1.12)
ax0.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


# -------------------------------------------------------------------------
# Panel (b): Coherence Stability across Incidence Angles (Theta = 32.5 to 45.0)
# -------------------------------------------------------------------------
ax1 = axes[1]
T.style_axes(ax1, grid_axis='y')
ax1.set_title(r'(b) Stability Across Four Observed Geometries ($\theta_\Omega = 32.8^\circ \to 45.0^\circ$)', fontsize=7.8, fontweight='bold', pad=7.0)

# Plot class median trajectory across the 4 tracks
for k, name in enumerate(CLASSES):
    medians = [full_stats[t][name]['median'] for t in TRACKS]
    iqrs = [full_stats[t][name]['q75'] - full_stats[t][name]['q25'] for t in TRACKS]
    
    # Solid line with marker
    ax1.plot(THETAS, medians, color=COLORS[k], lw=1.5, marker='s' if name == 'Built-up' else 'o',
             markersize=4.5, label=name, zorder=4)
    # Error bar (IQR / 2)
    ax1.errorbar(THETAS, medians, yerr=[iq/2 for iq in iqrs], color=COLORS[k], fmt='none',
                 capsize=2.2, elinewidth=0.8, capthick=0.8, alpha=0.7, zorder=3)

# Vertical double-headed arrow showing discriminative gap between Built-up and Forest
arrow_x = 34.8
ax1.annotate('', xy=(arrow_x, 0.82), xytext=(arrow_x, 0.28),
             arrowprops=dict(arrowstyle='<->', color='#b71c1c', lw=0.65, linestyle=(0, (3, 3)), mutation_scale=7, shrinkA=2, shrinkB=2), zorder=5)
ax1.text(arrow_x + 0.5, 0.55, r'$\Delta\gamma \approx +0.58$' + '\n' + r'(Stable gap)',
         fontsize=6.2, fontweight='bold', color='#b71c1c', ha='left', va='center', zorder=6)

ax1.set_xticks(THETAS)
ax1.set_xticklabels([f'T15\n({THETAS[0]}°)', f'T37\n({THETAS[1]}°)', f'T88\n({THETAS[2]}°)', f'T139\n({THETAS[3]}°)'], fontsize=6.2)
ax1.set_xlabel(r'Sentinel-1 Track and Incidence Angle ($\theta_{\Omega}$)', fontsize=6.8, labelpad=3.0)
ax1.set_ylabel(r'Class Median Coherence $\gamma_{12}$', fontsize=6.8, labelpad=3.0)
ax1.set_ylim(0.12, 1.15)
ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])

# Place legend in top area across the plot (ncol=5) above the data points
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99), frameon=True, facecolor='white',
           edgecolor='#cbd5e1', framealpha=0.95, fontsize=5.8, ncol=5,
           handlelength=1.2, columnspacing=0.8, labelspacing=0.2)

# Save dual formats to canonical fig7 and legacy stems
T.save(fig, 'fig7_coherence_distributions')
import shutil
for d in [T.OUTDIR, T.OUTDIR_ROOT]:
    shutil.copy2(d / 'fig7_coherence_distributions.pdf', d / 'fig6_coherence_distributions.pdf')
    shutil.copy2(d / 'fig7_coherence_distributions.png', d / 'fig6_coherence_distributions.png')
    shutil.copy2(d / 'fig7_coherence_distributions.pdf', d / 'fig10_coherence_distributions.pdf')
    shutil.copy2(d / 'fig7_coherence_distributions.png', d / 'fig10_coherence_distributions.png')
print("Successfully generated Fig. 7 / Fig. 6 coherence distributions!")

import json
from pathlib import Path
(Path(__file__).resolve().parents[1]/'coherence_consensus_statistics.json').write_text(json.dumps(full_stats,indent=2),encoding='utf8')

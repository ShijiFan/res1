"""Fig. 2 -- Cartographic Study Area and Spatial Partitions for IEEE TGRS.

Generates publication-grade cartographic figure showing:
  (a) Main Evaluation Split (28 train / 12 eval) with Netherlands regional locator inset,
      mutual four-track sensor validity mask outline (Omega), 10 km scale bar, North arrow,
      and 400 m k-d-tree exclusion buffer guard callout.
  (b) 3-Fold Continuous Spatial Cross-Validation (Fold 1 North, Fold 2 Central, Fold 3 South).
  (c) North 5.0 km Directional Holdout Robustness Control (6 eval, 5 buffer, 29 train).

Strictly adheres to IEEE TGRS visual standards:
  - Okabe-Ito colorblind-safe palette
  - True geographic coordinates (UTM Zone 32N & WGS84)
  - Clear typography hierarchy (DejaVu Sans / Arial)
  - Zero text overlap or clipping
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch, Patch
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

# Import tgrs styling and paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import OUTDIR, S_DIR
import tgrs_style as T
from tgrs_style import DOUBLE, COL

# Set font styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times', 'serif'],
    'font.size': 7.5,
    'axes.labelsize': 7.0,
    'xtick.labelsize': 6.2,
    'ytick.labelsize': 6.2,
    'legend.fontsize': 6.0,
    'mathtext.fontset': 'stix'
})

# Load split manifest
split_path = os.path.join(S_DIR, 'split_manifest.json')
with open(split_path, 'r') as f:
    d = json.load(f)

H = d['historical_development_split']
CV = d['source_spatial_cv_3fold']
N = d['continuous_buffered_holdouts']['north_holdout_5km_buffer']

ALL_BLOCKS = sorted(H['train_blocks'] + H['eval_blocks'])
EV_BLOCKS = set(H['eval_blocks'])
TR_BLOCKS = set(H['train_blocks'])

F1_BLOCKS = set(CV['fold_1_blocks'])
F2_BLOCKS = set(CV['fold_2_blocks'])
F3_BLOCKS = set(CV['fold_3_blocks'])

NE_BLOCKS = set(N['eval_blocks'])
NB_BLOCKS = set(N['buffer_blocks_excluded'])
NT_BLOCKS = set(N['train_blocks'])

# Load four-track mutual sensor validity mask
c6_path = r'E:/research/SAR/sar_v2/data/cube_4track_v6.npz'
c6 = np.load(c6_path)
sensor_mask = c6['four_track_valid'].astype(float) # (1077, 965)

# Downsample mask and extract boundary contour in block coordinates
mask_sub = sensor_mask[::4, ::4]
mask_smooth = gaussian_filter(mask_sub, sigma=1.0)
fig_dummy, ax_dummy = plt.subplots()
cs = ax_dummy.contour(mask_smooth, levels=[0.5])
cont_pts = cs.allsegs[0][0]
plt.close(fig_dummy)

# Convert contour to block coordinates (where block c spans [c, c+1] and row r spans [-r-1, -r])
x_cont = cont_pts[:, 0] * 4.0 / 125.0
y_cont = - cont_pts[:, 1] * 4.0 / 125.0

# Load Netherlands detailed GeoJSON for locator map
geo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'regional_detailed.geojson')
countries_geojson = None
if os.path.exists(geo_path):
    with open(geo_path, 'r') as f:
        countries_geojson = json.load(f)

# Color definitions (Okabe-Ito & publication palette)
C_ = {
    'train': '#e2e8f0',     # Light neutral slate
    'eval': '#D55E00',      # Okabe-Ito vermilion
    'buf': '#E69F00',       # Okabe-Ito orange / amber
    'f1': '#0072B2',        # Okabe-Ito blue
    'f2': '#009E73',        # Okabe-Ito bluish green
    'f3': '#CC79A7',        # Okabe-Ito reddish purple
    'mask_bg': '#f8fafc',   # Mutual validity subtle background
    'mask_line': '#64748b'  # Charcoal dashed contour
}

# Create figure with 3 panels
fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 3.55),
                         gridspec_kw=dict(wspace=0.14, left=0.075, right=0.985, top=0.88, bottom=0.25))

# Coordinate mappings for block labels
lon_labels = {2: '6.16°E', 3: '6.23°E', 4: '6.31°E', 5: '6.38°E', 6: '6.46°E', 7: '6.53°E'}
lat_labels = {
    1: '52.30°N', 2: '52.25°N', 3: '52.21°N', 4: '52.16°N',
    5: '52.12°N', 6: '52.07°N', 7: '52.03°N', 8: '51.98°N'
}

def setup_panel_axes(ax, is_leftmost=False):
    ax.set_xlim(0.3, 8.2)
    ax.set_ylim(-10.2, -0.4)
    ax.set_aspect('equal')
    
    # Tick marks for columns (2 to 7)
    col_ticks = [2, 4, 6]
    ax.set_xticks([c + 0.5 for c in col_ticks])
    ax.set_xticklabels([f'{c}\n({lon_labels[c]})' for c in col_ticks], fontsize=5.5)
    ax.set_xlabel('Block column (5 km width / Longitude)', fontsize=6.6, labelpad=3.5)
    
    # Tick marks for rows (1 to 8)
    row_ticks = list(range(1, 9))
    ax.set_yticks([-r - 0.5 for r in row_ticks])
    if is_leftmost:
        ax.set_yticklabels([f'R{r} ({lat_labels[r]})' for r in row_ticks], fontsize=5.6)
        ax.set_ylabel('Block row (5 km height / Latitude)', fontsize=6.5, labelpad=2.0)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)
    
    ax.tick_params(axis='x', length=2.5, pad=1.5)
    if is_leftmost:
        ax.tick_params(axis='y', length=2.5, pad=1.5)
        
    for spine in ax.spines.values():
        spine.set_color('#94a3b8')
        spine.set_linewidth(0.8)

# -------------------------------------------------------------
# Panel (a): Main Split (Headline Results)
# -------------------------------------------------------------
ax0 = axes[0]
setup_panel_axes(ax0, is_leftmost=True)
ax0.set_title('(A) Main evaluation split', fontsize=7.6, fontweight='bold', loc='left', pad=6.0)

# Background sensor mask
ax0.fill(x_cont, y_cont, color=C_['mask_bg'], zorder=0)
ax0.plot(x_cont, y_cont, color=C_['mask_line'], lw=1.0, ls='--', dashes=(3, 2), zorder=1)

# Render blocks
for b in ALL_BLOCKS:
    r, c = b // 100, b % 100
    is_ev = b in EV_BLOCKS
    fc = C_['eval'] if is_ev else C_['train']
    ax0.add_patch(Rectangle((c, -r-1), 1, 1, facecolor=fc, edgecolor='white', lw=1.2, zorder=2))
    
    txt_color = 'white' if is_ev else '#334155'
    ax0.text(c + 0.5, -r - 0.5, str(b), ha='center', va='center', fontsize=5.8,
             fontweight='bold', color=txt_color, zorder=3)

# Add Netherlands locator map inset in open top-left area
# Positioned safely at x in [0.4, 2.6], y in [-4.6, -0.8]
ax_nl = ax0.inset_axes([0.02, 0.52, 0.28, 0.42])
ax_nl.set_facecolor('#f0f7ff') # light water background
if countries_geojson:
    for feat in countries_geojson['features']:
        name = feat['properties'].get('ADMIN') or feat['properties'].get('name')
        geom = feat['geometry']
        poly_list = geom['coordinates'] if geom['type'] == 'MultiPolygon' else [geom['coordinates']]
        for poly in poly_list:
            ring = np.array(poly[0])
            fc = '#cbd5e1' if name == 'Netherlands' else '#f1f5f9'
            ec = '#475569' if name == 'Netherlands' else '#cbd5e1'
            ax_nl.fill(ring[:, 0], ring[:, 1], fc=fc, ec=ec, lw=0.5, zorder=1)

# Twente AOI red bounding box in locator map: [6.00, 51.98, 6.54, 52.38]
ax_nl.add_patch(Rectangle((5.95, 51.95), 0.65, 0.48, facecolor='#ef4444', edgecolor='#b91c1c', lw=1.0, zorder=5))
ax_nl.text(5.6, 52.2, 'Site-1\nTwente', fontsize=4.5, fontweight='bold', color='#b91c1c', ha='right', va='center', zorder=6)
ax_nl.annotate('', xy=(5.9, 52.2), xytext=(5.65, 52.2),
               arrowprops=dict(arrowstyle='->', color='#b91c1c', lw=0.8), zorder=6)
ax_nl.text(3.6, 53.2, 'North\nSea', fontsize=4.2, color='#64748b', style='italic', zorder=4)
ax_nl.text(6.6, 51.0, 'DE', fontsize=4.8, color='#94a3b8', fontweight='bold', zorder=4)
ax_nl.text(4.2, 50.8, 'BE', fontsize=4.8, color='#94a3b8', fontweight='bold', zorder=4)

ax_nl.set_xlim(3.1, 7.3)
ax_nl.set_ylim(50.6, 53.7)
ax_nl.set_aspect('equal')
ax_nl.set_xticks([])
ax_nl.set_yticks([])
for spine in ax_nl.spines.values():
    spine.set_color('#64748b')
    spine.set_linewidth(0.6)
ax_nl.text(0.5, 0.93, 'Study Area (NL)', transform=ax_nl.transAxes, fontsize=5.0, fontweight='bold',
           ha='center', va='top', color='#0f172a', zorder=7,
           bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='#64748b', lw=0.5, alpha=0.9))

# Add 400m Buffer schematic / callout badge in top-right open area (Cols 4.3-8.0, Rows 1-2)
ax0.text(6.05, -1.25, 'Spatial leakage guard\n($k$-d tree; $d_{min} \geq 400$ m)\nTraining exclusion: 400 m\nApplied before sampling',
         fontsize=5.1, ha='center', va='center', linespacing=1.25,
         color='#0f172a', zorder=5,
         bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#cbd5e1', lw=0.7))

# Add Scale Bar (10 km = 2 block widths) and North Arrow at bottom-left
sb_x0, sb_y0 = 0.65, -9.8
ax0.add_patch(Rectangle((sb_x0, sb_y0), 1.0, 0.15, facecolor='#0f172a', edgecolor='white', lw=0.5, zorder=6))
ax0.add_patch(Rectangle((sb_x0 + 1.0, sb_y0), 1.0, 0.15, facecolor='white', edgecolor='#0f172a', lw=0.5, zorder=6))
ax0.text(sb_x0, sb_y0 + 0.22, '0', fontsize=5.0, ha='center', va='bottom', color='#0f172a', fontweight='bold', zorder=6)
ax0.text(sb_x0 + 1.0, sb_y0 + 0.22, '5', fontsize=5.0, ha='center', va='bottom', color='#0f172a', fontweight='bold', zorder=6)
ax0.text(sb_x0 + 2.0, sb_y0 + 0.22, '10 km', fontsize=5.0, ha='center', va='bottom', color='#0f172a', fontweight='bold', zorder=6)

# North Arrow
ax0.annotate('N', xy=(1.0, -6.5), xytext=(1.0, -7.6),
             arrowprops=dict(facecolor='#0f172a', edgecolor='none', width=1.2, headwidth=4.5, headlength=4.5),
             ha='center', va='bottom', fontsize=5.8, fontweight='bold', color='#0f172a', zorder=6)

# Legend for Panel (a)
legend_a = [
    Patch(facecolor=C_['train'], edgecolor='#cbd5e1', lw=0.8, label='Training blocks (28)'),
    Patch(facecolor=C_['eval'], edgecolor='#b91c1c', lw=0.8, label='Held-out evaluation blocks (12)'),
    Line2D([0], [0], color=C_['mask_line'], lw=1.0, ls='--', dashes=(3, 2), label=r'Sensor valid boundary $M_2$ (633k px)')
]
ax0.legend(handles=legend_a, loc='upper center', bbox_to_anchor=(0.5, -0.165),
           frameon=True, facecolor='white', edgecolor='#e2e8f0', framealpha=0.9,
           fontsize=5.8, ncol=1, handlelength=1.1, handleheight=0.7, handletextpad=0.4, labelspacing=0.25)


# -------------------------------------------------------------
# Panel (b): 3-Fold Spatial Cross-Validation
# -------------------------------------------------------------
ax1 = axes[1]
setup_panel_axes(ax1, is_leftmost=False)
ax1.set_title('(B) Spatial model selection', fontsize=7.6, fontweight='bold', loc='left', pad=6.0)

# Background sensor mask
ax1.fill(x_cont, y_cont, color=C_['mask_bg'], zorder=0)
ax1.plot(x_cont, y_cont, color=C_['mask_line'], lw=1.0, ls='--', dashes=(3, 2), zorder=1)

# Render blocks
for b in ALL_BLOCKS:
    r, c = b // 100, b % 100
    if b in F1_BLOCKS:
        fc = C_['f1']
        is_dark = True
    elif b in F2_BLOCKS:
        fc = C_['f2']
        is_dark = True
    elif b in F3_BLOCKS:
        fc = C_['f3']
        is_dark = True
    else: # Held out from CV (the 12 eval blocks)
        fc = '#cbd5e1'
        is_dark = False
        
    ax1.add_patch(Rectangle((c, -r-1), 1, 1, facecolor=fc, edgecolor='white', lw=1.2, zorder=2))
    
    txt_color = 'white' if is_dark else '#334155'
    ax1.text(c + 0.5, -r - 0.5, str(b), ha='center', va='center', fontsize=5.8,
             fontweight='bold', color=txt_color, zorder=3)

# Legend for Panel (b)
legend_b = [
    Patch(facecolor=C_['f1'], label='Fold 1, North (7 blocks)'),
    Patch(facecolor=C_['f2'], label='Fold 2, Central (9 blocks)'),
    Patch(facecolor=C_['f3'], label='Fold 3, South (12 blocks)'),
    Patch(facecolor='#cbd5e1', edgecolor='#94a3b8', lw=0.8, label='Held out from CV (12 blocks)')
]
ax1.legend(handles=legend_b, loc='upper center', bbox_to_anchor=(0.5, -0.165),
           frameon=True, facecolor='white', edgecolor='#e2e8f0', framealpha=0.9,
           fontsize=5.8, ncol=1, handlelength=1.1, handleheight=0.7, handletextpad=0.4, labelspacing=0.25)


# -------------------------------------------------------------
# Panel (c): North 5.0 km Directional Holdout
# -------------------------------------------------------------
ax2 = axes[2]
setup_panel_axes(ax2, is_leftmost=False)
ax2.set_title(r'(C) North 5 km holdout', fontsize=7.6, fontweight='bold', loc='left', pad=6.0)

# Background sensor mask
ax2.fill(x_cont, y_cont, color=C_['mask_bg'], zorder=0)
ax2.plot(x_cont, y_cont, color=C_['mask_line'], lw=1.0, ls='--', dashes=(3, 2), zorder=1)

# Render blocks
for b in ALL_BLOCKS:
    r, c = b // 100, b % 100
    if b in NE_BLOCKS:
        fc = C_['eval']
        is_dark = True
    elif b in NB_BLOCKS:
        fc = C_['buf']
        is_dark = True
    else:
        fc = C_['train']
        is_dark = False
        
    ax2.add_patch(Rectangle((c, -r-1), 1, 1, facecolor=fc, edgecolor='white', lw=1.2, zorder=2))
    
    txt_color = 'white' if is_dark else '#334155'
    ax2.text(c + 0.5, -r - 0.5, str(b), ha='center', va='center', fontsize=5.8,
             fontweight='bold', color=txt_color, zorder=3)

# Legend for Panel (c)
legend_c = [
    Patch(facecolor=C_['eval'], edgecolor='#b91c1c', lw=0.8, label='Evaluation blocks (6 blocks)'),
    Patch(facecolor=C_['buf'], edgecolor='#b45309', lw=0.8, label='5.0 km exclusion buffer (5 blocks)'),
    Patch(facecolor=C_['train'], edgecolor='#cbd5e1', lw=0.8, label='Training blocks (29 blocks)')
]
ax2.legend(handles=legend_c, loc='upper center', bbox_to_anchor=(0.5, -0.165),
           frameon=True, facecolor='white', edgecolor='#e2e8f0', framealpha=0.9,
           fontsize=5.8, ncol=1, handlelength=1.1, handleheight=0.7, handletextpad=0.4, labelspacing=0.25)

# Save dual formats (PDF and PNG) to both manuscript/figures and figures/
T.save(fig, 'fig2_study_area')

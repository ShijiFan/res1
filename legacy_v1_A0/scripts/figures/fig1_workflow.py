"""Fig. 1 -- Observation-budget-controlled cross-track evaluation protocol.
Redesigned to SciPilot & IEEE TGRS publication standard with clean spacing.
"""
from common import *
from tgrs_style import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

fig = plt.figure(figsize=(DOUBLE, 3.85))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Palette
C_BORDER = '#d0d7de'
C_INTEN = COL['inten']   # Vermillion
C_COH = COL['coh']       # Blue
C_FUSED = COL['full']     # Green
C_GUARD = '#7b1fa2'     # Purple
C_EVAL = '#2c3e50'      # Dark Slate

def draw_card(x, y, w, h, title, panel_tag, header_col, bg_col='#ffffff'):
    """Draw a clean scientific card with top header banner."""
    # Base card
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                         fc=bg_col, ec=C_BORDER, lw=0.8, zorder=1)
    ax.add_patch(box)
    
    # Header bar
    hh = 6.8
    header_box = FancyBboxPatch((x, y + h - hh), w, hh,
                                boxstyle="round,pad=0,rounding_size=1.2",
                                fc=header_col, ec='none', zorder=2)
    ax.add_patch(header_box)
    ax.add_patch(Rectangle((x, y + h - hh), w, 2.0, fc=header_col, ec='none', zorder=2))
    
    # Title text
    ax.text(x + 1.8, y + h - hh / 2.0, f"({panel_tag})  {title}",
            fontsize=8.0, fontweight='bold', color='white', va='center', ha='left', zorder=3)
    return y

# -------------------------------------------------------------
# Card 1: Observation Budget Parity (Left)
# -------------------------------------------------------------
x1, y1, w1, h1 = 1.0, 48.0, 30.0, 50.5
draw_card(x1, y1, w1, h1, "Observation Budget Parity", "a", C_COH)

# Content for Card 1
ax.text(x1 + 2.2, y1 + 40.5, r"Strict $N=2$ Satellite Budget:", fontsize=7.2, fontweight='bold', color='#111111', va='top')
ax.text(x1 + 2.2, y1 + 37.0, 
        r"• Exactly 2 SLC acquisitions per track" + "\n" +
        r"• 12-day repeat ($\Delta t = 12$ d, C-band 5.4 GHz)" + "\n" +
        r"• Dual-pol: VV + VH complex channels",
        fontsize=6.5, color='#333333', linespacing=1.4, va='top')

# Orbit Geometry Graphic
ax.add_patch(FancyBboxPatch((x1 + 2.0, y1 + 5.5), w1 - 4.0, 19.5,
                            boxstyle="round,pad=0,rounding_size=0.8", fc='#f0f7fc', ec='#cfe2ff', lw=0.6, zorder=2))

ax.text(x1 + w1 / 2.0, y1 + 23.5, "4 Satellite Tracks (Cross-Track Range)", 
        fontsize=6.6, fontweight='bold', color=C_COH, ha='center', va='top')

tracks = [
    ("T15", "Asc.", r"32.8$^\circ$"),
    ("T37", "Desc.", r"37.1$^\circ$"),
    ("T88", "Asc.", r"41.4$^\circ$"),
    ("T139", "Desc.", r"45.0$^\circ$")
]
bx0 = x1 + 2.8
for tid, direct, inc in tracks:
    ax.add_patch(FancyBboxPatch((bx0, y1 + 7.5), 5.8, 12.5, boxstyle="round,pad=0,rounding_size=0.5",
                                fc='white', ec=C_COH, lw=0.6, zorder=3))
    ax.text(bx0 + 2.9, y1 + 16.8, tid, fontsize=6.8, fontweight='bold', color=C_COH, ha='center', va='center', zorder=4)
    ax.text(bx0 + 2.9, y1 + 13.0, direct, fontsize=5.8, color='#555555', ha='center', va='center', zorder=4)
    ax.text(bx0 + 2.9, y1 + 10.0, inc, fontsize=6.2, fontweight='bold', color='#222222', ha='center', va='center', zorder=4)
    bx0 += 6.2

ax.text(x1 + w1 / 2.0, y1 + 2.8, r"Angle disparity: $\Delta\theta_{\mathrm{max}} = 12.2^\circ$ on common support $\Omega$",
        fontsize=5.9, color='#555555', style='italic', ha='center')


# -------------------------------------------------------------
# Card 2: Representation Ladder (Middle)
# -------------------------------------------------------------
x2, y2, w2, h2 = 33.0, 48.0, 34.0, 50.5
draw_card(x2, y2, w2, h2, "Five-Level Representation Ladder", "b", C_INTEN)

rungs = [
    (r"$I_8$", "8-D", "2-date intensity baseline (VV, VH, ratios)", '#8c8c8c'),
    (r"$I_{28}$", "28-D", r"$I_8$ + multi-scale textures & gradients", C_INTEN),
    (r"$I_8{+}g_{12}$", "9-D", r"$I_8$ + raw 12-day scalar coherence $\gamma_{12}$", C_COH),
    (r"$F_{12}$", "12-D", r"$I_8{+}g_{12}$ + 3 derived transforms (texture, grad, log)", '#56B4E9'),
    (r"$I_{28}{+}g_{12}$", "29-D", "Full fusion: 28-D intensity + scalar coherence", C_FUSED)
]

ry = y2 + 37.5
for sym, dim, desc, col in rungs:
    ax.add_patch(FancyBboxPatch((x2 + 2.0, ry - 0.5), 10.5, 5.8, boxstyle="round,pad=0,rounding_size=0.6",
                                fc=col, ec='none', zorder=3))
    ax.text(x2 + 4.8, ry + 2.4, sym, fontsize=7.2, fontweight='bold', color='white', ha='center', va='center', zorder=4)
    ax.text(x2 + 9.6, ry + 2.4, dim, fontsize=6.0, color='white', ha='center', va='center', zorder=4)
    ax.text(x2 + 13.5, ry + 2.4, desc, fontsize=5.8, color='#222222', va='center', ha='left', zorder=3)
    ry -= 6.8

# Invariant badge
ax.add_patch(FancyBboxPatch((x2 + 2.0, y2 + 2.2), w2 - 4.0, 4.6,
                            boxstyle="round,pad=0,rounding_size=0.6", fc='#fef3f2', ec='#fecaca', lw=0.6, zorder=2))
ax.text(x2 + w2 / 2.0, y2 + 4.5, r"Physical scalar $\gamma_{12} \in [0, 1]$ introduces phase stability",
        fontsize=6.2, fontweight='bold', color='#b71c1c', ha='center', va='center', zorder=3)


# -------------------------------------------------------------
# Card 3: Spatial Guards & Protocols (Right)
# -------------------------------------------------------------
x3, y3, w3, h3 = 69.0, 48.0, 30.0, 50.5
draw_card(x3, y3, w3, h3, "Spatial Validation Guard", "c", C_GUARD)

ax.text(x3 + 2.2, y3 + 40.5, "Strict Autocorrelation Control:", fontsize=7.2, fontweight='bold', color='#111111', va='top')
ax.text(x3 + 2.2, y3 + 37.0, 
        r"• 40 Spatial Blocks ($5 \times 5$ km grid)" + "\n" +
        r"• 28 Training Blocks (20,000 pixels)" + "\n" +
        r"• 12 Held-out Evaluation Blocks (9,600 px)",
        fontsize=6.5, color='#333333', linespacing=1.4, va='top')

ax.add_patch(FancyBboxPatch((x3 + 2.0, y3 + 6.2), w3 - 4.0, 21.5,
                            boxstyle="round,pad=0,rounding_size=0.8", fc='#faf5ff', ec='#e9d5ff', lw=0.6, zorder=2))
ax.text(x3 + w3 / 2.0, y3 + 25.8, "Spatial Isolation Guarantees", fontsize=6.6, fontweight='bold', color=C_GUARD, ha='center', va='top')

guards = [
    ("400 m Buffer Guard", "k-d tree distance floor between train & val", '#6b21a8'),
    ("3-Fold Spatial CV", "North, Central, South source partitions", '#4c1d95'),
    ("5.0 km Holdout", "Directional North buffer robustness check", '#581c87')
]
gy = y3 + 19.8
for g_title, g_desc, g_col in guards:
    ax.plot([x3 + 4.2], [gy + 0.4], marker='o', markersize=3.8, color=g_col, zorder=3)
    ax.text(x3 + 5.8, gy + 1.4, g_title, fontsize=6.2, fontweight='bold', color=g_col, va='center', ha='left', zorder=3)
    ax.text(x3 + 5.8, gy - 1.2, g_desc, fontsize=5.5, color='#555555', va='center', ha='left', zorder=3)
    gy -= 5.5

ax.text(x3 + w3 / 2.0, y3 + 3.1, "No feature-window overlap; residual correlation bounded",
        fontsize=5.6, color='#555555', style='italic', ha='center', va='center')


# -------------------------------------------------------------
# Card 4: Evaluation Protocol & Inference (Bottom Full-Width)
# -------------------------------------------------------------
x4, y4, w4, h4 = 1.0, 1.5, 98.0, 43.0
draw_card(x4, y4, w4, h4, "Cross-Track Transfer Evaluation & Statistical Inference Framework", "d", C_EVAL)

# Column 1: Transfer Matrix Setup
cx1 = x4 + 2.5
ax.text(cx1, y4 + 32.5, "12 Cross-Track Pairs ($s \\neq t$)", fontsize=7.4, fontweight='bold', color=C_COH, va='top')
ax.text(cx1, y4 + 28.5,
        r"• Train on source track $s \in \{15, 37, 88, 139\}$" + "\n" +
        r"• Evaluate on target track $t \neq s$" + "\n" +
        r"• $4 \times 3 = 12$ directional transfer tasks" + "\n" +
        r"• Evaluates cross-incidence & cross-heading" + "\n" +
        r"• Strict held-out spatial test blocks",
        fontsize=6.4, color='#333333', linespacing=1.35, va='top')

# Column 2: Estimands & Formulas
cx2 = x4 + 34.0
ax.text(cx2 + 15.0, y4 + 32.5, "Primary Estimands & Gap Formulation", fontsize=7.4, fontweight='bold', color=C_INTEN, ha='center', va='top')

# Formula Box 1: Balanced Accuracy
ax.add_patch(FancyBboxPatch((cx2, y4 + 17.5), 30.0, 11.0, boxstyle="round,pad=0,rounding_size=0.6",
                            fc='#fff7ed', ec='#fed7aa', lw=0.6, zorder=2))
ax.text(cx2 + 15.0, y4 + 24.5, r"$\mathrm{BA}_{\mathrm{pooled}} = \frac{1}{12} \sum_{s \neq t} \frac{1}{5} \sum_{k=1}^5 \mathrm{Recall}_{s \rightarrow t}(k)$",
        fontsize=7.2, fontweight='bold', color='#9a3412', ha='center', va='center', zorder=3)
ax.text(cx2 + 15.0, y4 + 20.0, "Primary Metric: Unweighted macro-average across 5 classes",
        fontsize=5.8, color='#7c2d12', ha='center', va='center', zorder=3)

# Formula Box 2: Transfer Gap
ax.add_patch(FancyBboxPatch((cx2, y4 + 3.5), 30.0, 11.5, boxstyle="round,pad=0,rounding_size=0.6",
                            fc='#f0fdf4', ec='#bbf7d0', lw=0.6, zorder=2))
ax.text(cx2 + 15.0, y4 + 10.8, r"Transfer Gap: $G = \mathrm{Acc}_{\mathrm{in}} - \mathrm{Acc}_{\mathrm{cross}}$",
        fontsize=7.2, fontweight='bold', color='#166534', ha='center', va='center', zorder=3)
ax.text(cx2 + 15.0, y4 + 6.2, r"Gap Compression: $\Delta G = G(I_{28}) - G(I_{28}{+}g_{12}) = \Delta_{\mathrm{cross}} - \Delta_{\mathrm{in}}$",
        fontsize=6.1, color='#15803d', ha='center', va='center', zorder=3)

# Column 3: Rigorous Bootstrap Inference
cx3 = x4 + 67.5
ax.text(cx3, y4 + 32.5, "Joint Block Bootstrap Inference", fontsize=7.4, fontweight='bold', color='#1e293b', va='top')
ax.text(cx3, y4 + 28.5,
        r"• $B = 20,000$ joint spatial block bootstrap resamples" + "\n" +
        r"• Shared block resample indices across all arms" + "\n" +
        r"• Complete-class replicates: $19,306$ valid draws" + "\n" +
        r"• Nonparametric percentile 95% confidence intervals" + "\n" +
        r"• Paired 12-block exact sign test ($p=0.0005$)",
        fontsize=6.4, color='#333333', linespacing=1.35, va='top')

# Badges at bottom right
ax.add_patch(FancyBboxPatch((cx3, y4 + 3.0), 28.0, 6.8, boxstyle="round,pad=0,rounding_size=0.6",
                            fc='#f1f5f9', ec='#cbd5e1', lw=0.6, zorder=2))
ax.text(cx3 + 14.0, y4 + 6.4, "Saved Models, Predictions & Verification Logs",
        fontsize=6.1, fontweight='bold', color='#334155', ha='center', va='center', zorder=3)


# Inter-card Connecting Arrows
# -------------------------------------------------------------
def draw_flow_arrow(x0, y0, x1, y1, shrinkA=0, shrinkB=0):
    arrow = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=9,
                            lw=1.1, color='#475569', shrinkA=shrinkA, shrinkB=shrinkB, zorder=10)
    ax.add_patch(arrow)

draw_flow_arrow(x1 + w1, y1 + h1 / 2.0, x2, y2 + h2 / 2.0)
draw_flow_arrow(x2 + w2, y2 + h2 / 2.0, x3, y3 + h3 / 2.0)

# Downward arrows from top cards to bottom evaluation card with air gaps
draw_flow_arrow(x1 + w1 / 2.0, y1, x1 + w1 / 2.0, y4 + h4, shrinkA=1.5, shrinkB=1.5)
draw_flow_arrow(x2 + w2 / 2.0, y2, x2 + w2 / 2.0, y4 + h4, shrinkA=1.5, shrinkB=1.5)
draw_flow_arrow(x3 + w3 / 2.0, y3, x3 + w3 / 2.0, y4 + h4, shrinkA=1.5, shrinkB=1.5)

save(fig, 'fig1_workflow')

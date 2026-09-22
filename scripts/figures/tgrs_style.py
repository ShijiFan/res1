"""Shared IEEE TGRS figure style and SciPilot-compliant layout routines."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

mpl.use('Agg')

# Professional SciPilot + IEEE TGRS typography (Times New Roman serif)
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times', 'serif'],
    'font.size': 8.0,
    'axes.labelsize': 8.2,
    'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'xtick.major.size': 3.0,
    'ytick.major.size': 3.0,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'mathtext.fontset': 'stix',
})

# Universal Colorblind-Safe Palette (Okabe-Ito)
COL = dict(
    coh='#0072B2',       # Blue (Coherence)
    inten='#D55E00',     # Vermillion (Intensity baseline)
    full='#009E73',      # Bluish Green (Fused / Full)
    dl='#CC79A7',        # Reddish Purple
    amber='#E69F00',     # Orange
    sky='#56B4E9',       # Sky Blue
    neutral='#555555',   # Dark Slate
    chance='#999999',    # Gray
    bg_alt='#f8f9fa'     # Very light tint for alternating bands
)

SINGLE, DOUBLE = 3.5, 7.16
OUTDIR = Path(__file__).resolve().parents[1] / 'checked_manuscript/figures'
OUTDIR_ROOT = Path(__file__).resolve().parents[1] / 'figure_previews'

def add_panel_label(ax, letter, x=-0.12, y=1.06):
    """Place bold panel label (a), (b) at top left following IEEE TGRS standard."""
    ax.text(x, y, f'({letter})', transform=ax.transAxes,
            fontsize=9.0, fontweight='bold', va='bottom', ha='left', color='#111111')

def style_axes(ax, grid_axis='x'):
    """Apply clean minimalist IEEE spines and soft gridlines."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    if grid_axis:
        ax.grid(True, axis=grid_axis, color='#e2e2e2', linestyle='--', linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)

def save(fig, stem):
    """Save both PDF (vector for TeX) and PNG (raster for preview) to both target directories."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    OUTDIR_ROOT.mkdir(parents=True, exist_ok=True)
    
    pdf_path = OUTDIR / f'{stem}.pdf'
    png_path = OUTDIR / f'{stem}.png'
    
    fig.savefig(pdf_path, dpi=600)
    fig.savefig(png_path, dpi=300)
    
    # Also copy to root figures
    shutil.copy2(pdf_path, OUTDIR_ROOT / f'{stem}.pdf')
    shutil.copy2(png_path, OUTDIR_ROOT / f'{stem}.png')
    print(f'Wrote {stem}.pdf and {stem}.png to manuscript/figures and figures')
    plt.close(fig)

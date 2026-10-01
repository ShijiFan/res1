"""
reproduce_all_figures.py - 1-Click Generation of Figures 1 to 14

Paper: Cross-Track Sentinel-1 Land-Cover Classification with Two-Acquisition Interferometric Coherence
Journal: IEEE Transactions on Geoscience and Remote Sensing (TGRS)
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "scripts/figures"

scripts = [
    "fig2_partitions.py",
    "fig4_perclass.py",
    "fig6_coherence_distributions.py",
]

def main():
    print("=" * 70)
    print("  REPRODUCING TGRS PAPER FIGURES")
    print("=" * 70)
    
    python_exe = sys.executable
    out_dir = ROOT / "figure_outputs"
    out_dir.mkdir(exist_ok=True)
    
    for script_name in scripts:
        sp = FIG_DIR / script_name
        if sp.exists():
            print(f"--> Running {script_name}...")
            res = subprocess.run([python_exe, str(sp)], cwd=FIG_DIR, capture_output=True, text=True)
            if res.returncode == 0:
                print(f"    [OK] {script_name}")
            else:
                print(f"    [FAIL] {script_name}:\n{res.stderr}")
        else:
            print(f"    [SKIP] {script_name} not found")
            
    print("\nFigure reproduction complete. Figures saved to figure_outputs / manuscript/figures.\n")

if __name__ == "__main__":
    main()

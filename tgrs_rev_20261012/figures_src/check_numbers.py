"""Compare every number in the revised manuscript (+ supplement) with the base manuscript.

Prints numbers that are new (absent from the base) and numbers of the base that disappeared from both the
revised main text and the supplement. New numbers must be traceable to this revision's manifests.
"""
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW = HERE.parent / "manuscript"
OLD = HERE.parents[1] / "merged_tgrs_20260930" / "manuscript"
NUM = re.compile(r"(?<![A-Za-z_\\])[-+]?\d[\d,]*\.?\d*")


def text(root, files):
    s = ""
    for f in files:
        t = (root / f).read_text(encoding="utf-8")
        t = re.sub(r"(?m)%.*$", "", t)                      # comments
        t = re.sub(r"\\(label|ref|cite|includegraphics|input)\{[^}]*\}", "", t)
        t = re.sub(r"\\(begin|end)\{[^}]*\}(\{[^}]*\})?", "", t)
        t = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}|\\(textwidth|columnwidth)", "", t)
        s += t + "\n"
    return s


def nums(s):
    return Counter(x.strip("+-").rstrip(".,") for x in NUM.findall(s) if x.strip("+-.,"))


old_files = ["main.tex"] + [f"chapters/{p.name}" for p in sorted((OLD / "chapters").glob("*.tex"))]
new_main = ["main.tex"] + [f"chapters/{p.name}" for p in sorted((NEW / "chapters").glob("0[1-6]*.tex"))]
o = nums(text(OLD, old_files))
n_main = nums(text(NEW, new_main))
n_supp = nums(text(NEW, ["supplement.tex"]))
new_only = sorted(set(n_main) | set(n_supp) - set(o))
gone = sorted(set(o) - set(n_main) - set(n_supp))
print("numbers in revision but not in base:", [x for x in sorted((set(n_main) | set(n_supp)) - set(o))])
print("numbers in base missing from revision:", gone)

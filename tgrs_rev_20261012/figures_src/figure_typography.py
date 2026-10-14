"""Normalize text at the final manuscript width; preserve all plotted data."""
import re
import matplotlib as mpl
from matplotlib.text import Text
mpl.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none'})
def normalize_figure(fig):
    mpl.rcParams.update({'mathtext.fontset':'stix'})
    titles={id(a.title) for a in fig.axes}|{id(a._left_title) for a in fig.axes}|{id(a._right_title) for a in fig.axes}
    for obj in fig.findobj(Text):
        obj.set_fontfamily('Times New Roman')
        s=obj.get_text();obj.set_text(re.sub(r'\(([A-H])\)',lambda m:'('+m[1].lower()+')',s))
        old=obj.get_fontsize()
        if id(obj) in titles or re.match(r'^\([a-h]\)',obj.get_text()):obj.set_fontsize(10.5);obj.set_fontweight('normal')
        elif old>=9:obj.set_fontsize(9.5)
        else:obj.set_fontsize(min(9.5,max(8.0,old)))
    # Tight-layout result charts have sufficient room after width expansion.
    if len(fig.axes)==2 and fig.get_figwidth()<6.2:
        fig.tight_layout(w_pad=2.0)

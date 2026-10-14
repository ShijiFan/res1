# Copy of merged_tgrs_20260930/plan/literature_repair_20261002/plot_price.py; only change: panel (b) x-range zoomed to the data.
from pathlib import Path
import sys,json,hashlib,datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
H=Path(__file__).resolve().parent;R=H.parents[1];S=H;SUBP=R/'merged_tgrs_20260930/plan'
sys.path.insert(0,str(S));import make_figures as old
p=SUBP/'substantive_20261002/statistics/price_thresholds.csv'
pt=pd.read_csv(p);pt=pt[pt.M.isin([2,4,6])]
fig,axs=plt.subplots(1,2,figsize=(7.16,3.15),gridspec_kw={'width_ratios':[1.25,1]})
markers={(2024,37):'o',(2024,88):'s',(2025,37):'^',(2025,88):'v'}
for site,color,off in [(1,'#0072B2',-.12),(2,'#D55E00',.12)]:
    for idx,((year,track),marker) in enumerate(markers.items()):
        z=pt[(pt.site==site)&(pt.year==year)&(pt.track==track)]
        yy=z.M.map({2:0,4:1,6:2}).values+off+(idx-1.5)*.045
        axs[0].scatter(z.interpolated_r,yy,c=color,s=27,marker=marker,zorder=3)
axs[0].set_yticks([0,1,2],['2','4','6']);axs[0].set_ylabel('Coherence budget M (pairs)')
axs[0].set_xlabel('Break-even interferogram / RTC price ratio')
axs[0].set_xlim(0,.32);axs[0].set_ylim(-.45,3.55)
axs[0].set_xticks([0,.1,.2,.3]);axs[0].set_title('(a) Interpolated price thresholds',loc='left')
handles=[Line2D([],[],marker='o',color=c,ls='',label='Site '+s) for s,c in [('A','#0072B2'),('B','#D55E00')]]
leg=axs[0].legend(handles=handles,loc='upper left',frameon=False,ncol=2,fontsize=8)
axs[0].add_artist(leg)
handles=[Line2D([],[],marker=m,color='.3',ls='',label=f'{y} T{t}') for (y,t),m in markers.items()]
axs[0].legend(handles=handles,loc='upper left',bbox_to_anchor=(0,.90),frameon=False,ncol=2,fontsize=8,columnspacing=.8,handletextpad=.3)
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()}
for site,path,color in [('A',R/'A1_observation_budget_20260923/runs/20260924_A1_MAIN','#0072B2'),('B',R/'site2_20260930/runs/MAIN','#D55E00')]:
    src=path/'stats/T_E1_deltaC.csv';hashes[str(src)]=hashlib.sha256(src.read_bytes()).hexdigest()
    x=pd.read_csv(src);axs[1].scatter(x.dC*100,x.dPol*100,s=16,c=color,label=f'Site {site}',alpha=.8)
axs[1].plot([-1,19],[-1,19],color='.5',lw=.7,ls='--');axs[1].set_xlim(-0.6,4.6);axs[1].set_ylim(-1,19);axs[1].text(3.1,3.7,'equal gain',color='.4',fontsize=7,rotation=12)
axs[1].set_xlabel('Coherence gain (pp BA)');axs[1].set_ylabel('Polarization gain (pp BA)')
axs[1].set_title('(b) Matched uniform dates',loc='left');axs[1].legend(frameon=False)
for ax in axs:old.grid(ax,'both')
fig.tight_layout(w_pad=2);old.save(fig,'fig_cost_pol')
(H/'fig_cost_pol_provenance.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script':str(Path(__file__).resolve()),'input_sha256':hashes},indent=2))

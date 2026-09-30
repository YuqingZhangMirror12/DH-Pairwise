"""Measured-distribution figures, no illustrative or generated model data."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
root=Path(a.root);raw=json.loads((root/'raw_summary.json').read_text())
out=root/'figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
colors={'sim_select':'#3979ad','dunhuang_cv':'#bf5d3a','turufan':'#39795c'}
labels={'sim_select':'SIM SELECT','dunhuang_cv':'Dunhuang','turufan':'Turufan (no layout GT)'}

fig,ax=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
for s in ['sim_select','dunhuang_cv']:
    h=np.asarray(raw[s]['all']['correct_pair_distance_pooled_hist']);x=(np.arange(len(h))+.5)*.1
    ax[0].plot(x,h.cumsum()/h.sum(),label=labels[s],color=colors[s],lw=2)
    v=raw[s]['all']['radii']['10']['nearest_wrong_edge'];pr=[1,5,10,50,90]
    ax[1].plot(pr,[v['p'+str(i)] for i in pr],marker='o',label=f"{labels[s]} ({v['n']} modes)",color=colors[s])
ax[0].axhline(.95,color='#999',ls='--',lw=1);ax[0].set(xlabel='Pairwise displacement distance (px)',ylabel='Cumulative edge-pair fraction',xlim=(0,40),ylim=(0,1),title='Spread of GT20-labelled correct edges')
ax[1].set(xlabel='Percentile',ylabel='Nearest wrong-edge distance (px)',xticks=[1,5,10,50,90],title='Correct mode to nearest wrong edge\n10px analysis bandwidth')
for x in ax:x.grid(alpha=.15);x.legend(fontsize=9)
fig.suptitle('Spread and nearby wrong support overlap; no universal clean separation',fontsize=13)
fig.savefig(out/'raw_spread_and_separation.png');plt.close(fig)

fig,ax=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
bins=['<=3','3-4','4-5','5-6']
for s in ['sim_select','dunhuang_cv']:
    pts=[(i,raw[s][b]) for i,b in enumerate(bins) if raw[s][b]['correct_pair_distance_pooled']['p95'] is not None]
    ax[0].plot([i for i,v in pts],[v['correct_pair_distance_pooled']['p95'] for i,v in pts],marker='o',color=colors[s],label=labels[s])
    ax[1].plot([i for i,v in pts],[v['per_pair_p95_over_spacing'].get('p50',np.nan) for i,v in pts],marker='o',color=colors[s],label=labels[s])
    for i,v in pts:ax[0].annotate(f"n={v['gt_pairs']}",(i,v['correct_pair_distance_pooled']['p95']),xytext=(0,8 if s=='dunhuang_cv' else -17),textcoords='offset points',ha='center',fontsize=8)
ax[0].set(ylabel='Pooled edge-pair P95 (px)',title='Pixel spread by spacing bin',ylim=(10,29))
ax[1].set(ylabel='Median per-pair (P95 / spacing)',title='Dividing by spacing does not align the domains')
for x in ax:x.set_xticks(range(4),bins);x.set_xlabel('Contour spacing (px)');x.grid(alpha=.15);x.legend(fontsize=9)
fig.savefig(out/'spacing_effect.png');plt.close(fig)

fig,ax=plt.subplots(1,3,figsize=(14,4.7),constrained_layout=True)
for s in ['sim_select','dunhuang_cv']:
    z=raw[s]['all']['radii'];r=[8,10,12,16]
    ax[0].plot(r,[z[str(i)]['wrong_mode_mass']['p50'] for i in r],marker='o',color=colors[s],label=labels[s])
    ax[0].plot(r,[z[str(i)]['wrong_mode_mass']['p95'] for i in r],ls='--',color=colors[s])
    ax[1].plot(r,[z[str(i)]['wrong_mode_distance']['p50'] for i in r],marker='o',color=colors[s],label=labels[s])
z=raw['turufan']['4-5']['radii'];r=[8,10,12,16]
ax[2].plot(r,[z[str(i)]['mode_count']['p50'] for i in r],marker='o',color=colors['turufan'],label='Median modes / pair')
ax[2].plot(r,[z[str(i)]['mode_count']['p95'] for i in r],ls='--',color=colors['turufan'],label='P95 modes / pair')
ax[0].set(title='Wrong-edge modes: absolute mass\nsolid median; dashed P95',ylabel='Q x arc-cell mass (px)',yscale='log')
ax[1].set(title='Wrong mode to nearest correct mode',ylabel='Median displacement distance (px)')
ax[2].set(title='Turufan spacing 4-5px\n250 pairs; no correctness labels',ylabel='Modes before any candidate budget')
for x in ax:x.set_xlabel('Analysis bandwidth (px)');x.set_xticks(r);x.grid(alpha=.15);x.legend(fontsize=9)
fig.savefig(out/'wrong_and_turufan_modes.png');plt.close(fig)

comparison=root/'simple_results/summary.json'
if comparison.exists():
    d=json.loads(comparison.read_text());names=['baseline','s8','s10','s12','s16']
    fig,ax=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    for s in ['sim_select','dunhuang_cv']:
        ax[0].plot(range(5),[100*d[n][s]['complete_raw_fraction'] for n in names],marker='o',color=colors[s],label=labels[s])
    ax[0].set(ylabel='Pairs with all GT20 raw edges\nin a correct retained cluster (%)',title='Same raw-edge definition, old and new')
    ax[1].bar(range(5),[d[n]['dunhuang_cv']['top_correct'] for n in names],color=['#888']+['#bf5d3a']*4)
    ax[1].axhline(216,ls='--',color='#666');ax[1].set(ylim=(0,292),ylabel='Correct / 292',title='Dunhuang highest-support layout')
    ax[2].bar(range(5),[d[n]['sim_select']['raw_edge_mixed20_40'] for n in names],color=['#888']+['#3979ad']*4)
    ax[2].set(ylabel='Clusters (pre-budget)',title='SIM severe raw-edge mixing\nContains GT20 and beyond40px edges')
    for x in ax:x.set_xticks(range(5),['Old','8px','10px','12px','16px']);x.grid(axis='y',alpha=.15)
    ax[0].legend(fontsize=9)
    fig.savefig(out/'before_after.png');plt.close(fig)
    native=root/'simple_results/native_mode_audit/summary.json'
    if native.exists():
        h=json.loads(native.read_text())
        fig,ax=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
        for s in ['sim_select','dunhuang_cv']:
            ax[0,0].plot(range(5),[100*h[s][n]['fraction'] for n in names],marker='o',color=colors[s],label=labels[s])
        ax[0,0].set(title='Complete native hypotheses in one correct cluster',ylabel='Positive pairs (%)',ylim=(0,105))
        ax[0,0].legend(fontsize=9)
        counts=[d[n]['dunhuang_cv']['top_correct'] for n in names]
        ax[0,1].bar(range(5),counts,color=['#888']+['#bf5d3a']*4)
        ax[0,1].axhline(216,ls='--',color='#666')
        for i,v in enumerate(counts):ax[0,1].text(i,v+4,str(v),ha='center')
        ax[0,1].set(title='Dunhuang top-support layout',ylabel='Correct / 292',ylim=(0,292))
        for key,label,c in [('old_reference_union_coverage','Union membership','#3979ad'),('old_reference_directional_coverage','After original directional weighting','#bf5d3a')]:
            ax[1,0].plot(range(5),[100*d[n]['dunhuang_cv'][key]['mean'] for n in names],marker='o',label=label,color=c)
        ax[1,0].set(title='Dunhuang: fixed reference edges',ylabel='Mean per-pair coverage (%)',ylim=(0,105))
        ax[1,0].legend(fontsize=9)
        ax[1,1].bar(range(5),[d[n]['sim_select']['raw_edge_mixed20_40'] for n in names],color=['#888']+['#3979ad']*4)
        ax[1,1].set(title='SIM raw-edge severe mixing is NOT zero',ylabel='Pre-budget clusters containing GT20 and >40px edges')
        for x in ax.flat:
            x.set_xticks(range(5),['Old','8px','10px','12px','16px'])
            x.grid(axis='y',alpha=.15)
        fig.suptitle('16px selected on SIM native-hypothesis metrics; not full acceptance\nHypothesis completeness is NOT raw-edge completeness or Attention retention',fontsize=13)
        fig.savefig(out/'aggregation_comparison.png');plt.close(fig)
print(json.dumps({'figures':[str(p) for p in sorted(out.glob('*.png'))]}))

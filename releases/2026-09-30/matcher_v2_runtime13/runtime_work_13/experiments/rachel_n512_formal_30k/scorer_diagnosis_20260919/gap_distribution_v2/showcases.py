"""Render actual masks/GT and a literal pixel-scale seam zoom; no model output."""
import argparse
import base64
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
from ..s7_balanced_v2.materialize import initialize,clean_positive,negative_source,STATE
from .measure import read,save,raw_outline,side_distances

COLORS=['#168a95','#e7a54a']

def display(ax,sample,title,zoom=None):
    masks=[np.asarray(getattr(sample,'mask_'+s)[0],bool) for s in 'ab']
    shifts=[np.zeros(2),-sample.translation_a_to_b_rc] if sample.label else [np.zeros(2),np.array([0.,850.])]
    bounds=[]
    from matplotlib.colors import to_rgba
    for m,o,col in zip(masks,shifts,COLORS):
        c=np.argwhere(m);lo=c.min(0);hi=c.max(0)+1
        rgba=np.zeros((*m.shape,4));rgba[m]=to_rgba(col)
        ax.imshow(rgba,origin='upper',extent=(o[1]-.5,o[1]+799.5,o[0]+799.5,o[0]-.5),interpolation='nearest')
        bounds.extend([lo+o,hi+o])
    points=np.array(bounds);lo=points.min(0);hi=points.max(0)
    if zoom is None:
        center=(lo+hi)/2;span=max(hi-lo)*1.08
    else:center=np.asarray(zoom);span=160.
    ax.set_xlim(center[1]-span/2,center[1]+span/2);ax.set_ylim(center[0]+span/2,center[0]-span/2)
    ax.set_aspect('equal');ax.set_title(title,fontsize=11);ax.axis('off')
    bar=20 if zoom is not None else 100
    x=center[1]-.43*span;y=center[0]+.43*span
    ax.plot([x,x+bar],[y,y],color='#263e4b',lw=2)
    ax.text(x,y-.02*span,f'{bar}px',fontsize=8,color='#263e4b')
    if zoom is not None:
        for size,col in ((64,'#30254f'),(32,'#e65257'),(16,'#266848'),(7,'#161616')):
            ax.add_patch(Rectangle((center[1]-size/2,center[0]-size/2),size,size,fill=False,lw=1.2,ec=col))
        ax.text(.02,.99,'Boxes: 7 / 16 / 32 / 64 px',va='top',transform=ax.transAxes,fontsize=8)


def seam_center(sample):
    a,b=[raw_outline(getattr(sample,'mask_'+s)[0]) for s in 'ab']
    shift=-sample.translation_a_to_b_rc
    d,use=side_distances(a,b,shift);ids=np.flatnonzero(use['seam40'])
    if not len(ids):return (a['points'].mean(0)+b['points'].mean(0)+shift)/2
    # Representative upper-quartile local gap, not the most dramatic maximum.
    i=ids[np.argmin(abs(d[ids]-np.quantile(d[ids],.75)))]
    j=cKDTree(b['points']+shift).query(a['points'][i])[1]
    return (a['points'][i]+b['points'][j]+shift)/2


def raw_gap_mean(sample):
    a,b=[raw_outline(getattr(sample,'mask_'+s)[0]) for s in 'ab']
    values=[]
    for x,y,t in ((a,b,-sample.translation_a_to_b_rc),(b,a,sample.translation_a_to_b_rc)):
        d,use=side_distances(x,y,t);values.append(d[use['seam40']])
    return float(np.mean(np.concatenate(values)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--allow-partial',action='store_true',help='Qualitative figures only, never use completed-subset statistics as full distributions')
    a=p.parse_args();root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    protocol=read(root/'protocol.json');initialize(protocol['options'])
    if (root/'train_s7b_24k.json').exists():
        manifest=read(root/'train_s7b_24k.json');entries=manifest['entries']
        metrics={r['pair_id']:r for r in (json.loads(l) for l in (root/'pair_metrics.jsonl').read_text().splitlines())}
        selection_scope='completed dataset'
    elif a.allow_partial:
        groups=[read(f) for f in sorted((root/'groups').glob('*.json'))]
        entries=[e for g in groups for e in g['entries']]
        metrics={r['pair_id']:r for g in groups for r in g['statistics']}
        selection_scope=f'illustrations from {len(groups)} completed groups; not a distribution estimate'
    else:raise ValueError('Complete manifest required unless --allow-partial is explicitly selected')
    specs=[('clean','无腐蚀 · 完整接缝',lambda e:e['label'] and e['corrosion_recipe']=='clean' and not e['partial_applied']),
        ('partial','无腐蚀＋曲线Partial',lambda e:e['label'] and e['corrosion_recipe']=='clean' and e['partial_applied']),
        ('wave','单一波状腐蚀',lambda e:e['label'] and e['corrosion_recipe']=='wave' and not e['partial_applied']),
        ('partial-wave','Partial＋波状腐蚀',lambda e:e['label'] and e['corrosion_recipe']=='wave' and e['partial_applied']),
        ('wave-gaps','波状＋1–5处缺口',lambda e:e['label'] and e['corrosion_recipe']=='wave_gaps' and not e['partial_applied']),
        ('partial-gaps','Partial＋波状＋缺口',lambda e:e['label'] and e['corrosion_recipe']=='wave_gaps' and e['partial_applied']),
        ('local','波状＋局部深腐蚀',lambda e:e['label'] and e['corrosion_recipe']=='wave_local'),
        ('mirror','共同镜像（GT同步）',lambda e:e['label'] and bool(e['offline_paired_mirror'])),
        ('ratio','大小悬殊碎片',lambda e:e['label'] and metrics[e['pair_id']]['area_ratio']<.25),
        ('gen5','Gen5结构增强',lambda e:e['label'] and e['source_stratum'].startswith('gen5')),
        ('within-neg','同parent明确非邻接负例',lambda e:not e['label'] and e['negative_kind']=='same_parent_nonadjacent'),
        ('cross-neg','跨Gen负例',lambda e:not e['label'] and e['negative_kind']=='cross_gen'),
        ('anchor-neg','同锚多负伙伴之一',lambda e:not e['label'] and bool(e['anchor_group_id']))]
    selected=[];used=set()
    for key,title,test in specs:
        pool=[e for e in entries if test(e) and e['pair_id'] not in used]
        if not pool:continue
        if key in ('partial','partial-wave','partial-gaps'):
            visible=[e for e in pool if metrics[e['pair_id']]['augmentation']['partial']['material_retention'][0]<=.85]
            if visible:pool=visible
        med=np.median([metrics[e['pair_id']]['mean_fragment_area_px'] for e in pool])
        e=min(pool,key=lambda x:abs(metrics[x['pair_id']]['mean_fragment_area_px']-med));used.add(e['pair_id'])
        new,report=load_sample(root/e['artifact_path'])
        if e['label']:original=clean_positive(e['source_pair_id'])
        else:
            n=next(v for v in STATE['plan']['negative'] if v['pair_id']==e['source_pair_id'])
            original,_=negative_source(n)
        from ..seam_context_v3.augmentation import paired_mirror
        if e['offline_paired_mirror']:original=paired_mirror(original,e['offline_paired_mirror'])
        fig,axes=plt.subplots(1,3,figsize=(14,4.8));fig.patch.set_facecolor('white')
        display(axes[0],original,'Original TRAIN pair / GT' if new.label else 'Original negatives / unplaced')
        display(axes[1],new,'Augmented pair / GT' if new.label else 'Augmented negatives / unplaced')
        if new.label:display(axes[2],new,'Seam zoom: native pixel widths',seam_center(new))
        else:
            axes[2].axis('off');axes[2].text(.04,.8,'NEGATIVE PAIR\nNo GT seam or layout\nRandom own-contour damage\nNo fabricated correspondence',fontsize=14,va='top')
        fig.suptitle(f'S7-D | {key} | label={int(new.label)}',fontsize=14)
        fig.tight_layout();path=out/(key+'.png');fig.savefig(path,dpi=150,bbox_inches='tight');plt.close(fig)
        m=metrics[e['pair_id']];scale=report['pair_shared_scale']['common_scale']
        caption=f"实际配方 {e['corrosion_recipe']}；Partial={e['partial_applied']}；共同缩放 {scale:.3f}；面积比 1:{1/m['area_ratio']:.2f}。"
        partial=m['augmentation'].get('partial')
        if partial:
            ordinal=0 if new.label else 1
            caption+=f"裁切片保留材料{partial['material_retention'][ordinal]:.1%}。"
            if new.label:caption+=f"继承接缝保留{partial['source_seam_retention']:.1%}（不是整圈轮廓比例）。"
        if new.label:caption+=f"平滑轮廓的近接长度≤20px为 {m['d20_length_px']:.1f}px；原始轮廓双向近接点均隙≤40px为 {raw_gap_mean(new):.2f}px。左侧保留同镜像方向，缩放前后用各自100px标尺；右图160px范围。"
        else:caption+='前两栏只是分开展示，不代表可以拼接；负例无GT接缝。'
        selected.append(dict(id=key,title=title,pair_id=e['pair_id'],caption=caption,
            image='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode(),
            source_artifact=str(root/e['artifact_path']),recipe=e['corrosion_recipe'],label=bool(e['label']),selection_scope=selection_scope))
    save(out/'showcases.json',selected)
    print(json.dumps(dict(rendered=len(selected),out=str(out))))


if __name__=='__main__':main()

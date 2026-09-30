"""Render exact masks/GT-projected gap evidence; no synthetic illustrations."""
import argparse
import base64
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample


def plot_pair(ax,a,b,translation,title,limits=None):
    # Both images retain their exact pixel coordinates. No scale-normalizing
    # either fragment independently, no invented arrangement for negatives.
    t=np.asarray(translation);color=ListedColormap(['#168b93','#e9ac47'])
    for mask,offset,index in ((a,t,0),(b,np.zeros(2),1)):
        m=np.ma.masked_where(~mask,np.full(mask.shape,index))
        ax.imshow(m,cmap=color,vmin=0,vmax=1,interpolation='nearest',
            extent=[offset[1]-.5,offset[1]+799.5,offset[0]+799.5,offset[0]-.5])
    if limits is None:
        p=np.vstack([np.argwhere(a)+t,np.argwhere(b)])
        lo=p.min(0)-10;hi=p.max(0)+10
        limits=(lo,hi)
    lo,hi=limits;ax.set_xlim(lo[1],hi[1]);ax.set_ylim(hi[0],lo[0]);ax.set_aspect('equal')
    ax.set_title(title,fontsize=10);ax.axis('off')
    y=hi[0]-10;x=lo[1]+12;ax.plot([x,x+50],[y,y],c='#334155',lw=2);ax.text(x,y-4,'50px',fontsize=7)
    return limits


def render(record,bundle,out,projection=None,projection_receipt=None):
    name=Path(record['sample_path']).name;new,_=load_sample(bundle/'samples'/name);old,_=load_sample(bundle/'baseline'/name)
    with np.load(bundle/'proof'/name) as z:proof={k:z[k] for k in z.files}
    fig,axes=plt.subplots(2,2,figsize=(11,8.2),gridspec_kw={'height_ratios':[1.45,1]},layout='constrained')
    ispos=bool(record['label']);gt=new.translation_a_to_b_rc if ispos else np.array([0.,-900.])
    p=np.vstack([np.argwhere(old.mask_a[0])+gt,np.argwhere(old.mask_b[0]),
                 np.argwhere(new.mask_a[0])+gt,np.argwhere(new.mask_b[0])])
    limits=(p.min(0)-15,p.max(0)+15)
    plot_pair(axes[0,0],old.mask_a[0]>0,old.mask_b[0]>0,gt,'Existing v14 / same GT' if ispos else 'Existing v14 / negative: unplaced',limits)
    plot_pair(axes[0,1],new.mask_a[0]>0,new.mask_b[0]>0,gt,'New paired pilot / same GT' if ispos else 'New paired pilot / negative: unplaced',limits)
    # Depth revision: show the weathered side even when the approved cut was on the other fragment.
    side=next((s for s,v in record['detail']['primary_damage'].items() if v.get('applied')),record['detail']['trim']['side'])
    base=np.unpackbits(proof['packed_fragment_'+side],axis=1).astype(bool)
    trim=np.unpackbits(proof['packed_trim_'+side],axis=1).astype(bool)
    primary=np.unpackbits(proof['packed_primary_'+side],axis=1).astype(bool)
    final=getattr(new,'mask_'+side)[0]>0
    image=np.ones((*base.shape,3));image[base]=[.14,.34,.4]
    image[base&~trim]=[.86,.84,.81];image[trim&~primary]=[.89,.25,.15];image[primary&~final]=[.53,.3,.75]
    changed=base&~final;pix=np.argwhere(changed)
    if len(pix):lo=np.maximum(0,pix.min(0)-12);hi=np.minimum(799,pix.max(0)+12)
    else:lo=np.maximum(0,np.argwhere(base).min(0)-10);hi=np.minimum(799,np.argwhere(base).max(0)+10)
    axes[1,0].imshow(image,interpolation='nearest');axes[1,0].set_xlim(lo[1],hi[1]);axes[1,0].set_ylim(hi[0],lo[0])
    axes[1,0].set_aspect('equal');axes[1,0].set_title(f'Side {side.upper()}: gray trim / red major / purple light',fontsize=9)
    # Show the actual new trim boundary, not an invented illustrative curve.
    from scipy import ndimage
    edge=trim&~ndimage.binary_erosion(trim)&(ndimage.distance_transform_edt(base)>1.5)
    yy,xx=np.where(edge);axes[1,0].scatter(xx,yy,s=.5,c='#e59600',label='Actual TRAIN-donor cut edge')
    axes[1,0].legend(fontsize=7,loc='lower left')
    if ispos:
        projected=proof if projection is None else {'gap_'+k:v for k,v in projection.items()}
        q=projected['gap_a_gap'];valid=projected['gap_a_valid'];w=projected['gap_a_source_weight']
        affected=projected['gap_a_primary_affected'];arc=np.cumsum(w)-w/2
        curve=q.copy();curve[~valid]=np.nan
        discontinuity=np.r_[False,np.linalg.norm(np.diff(projected['gap_a_source_points'],axis=0),axis=1)>5]
        curve[discontinuity]=np.nan
        if projection is not None:
            old_curve=projected['gap_a_old_gap'].copy();old_curve[~projected['gap_a_old_valid']|discontinuity]=np.nan
            axes[1,1].plot(arc,old_curve,c='#748395',ls='--',lw=1,label='v14 / same original partners')
        axes[1,1].plot(arc,curve,c='#168b93',lw=1,label='New final two-sided gap')
        active=valid&affected;axes[1,1].scatter(arc[active],q[active],s=6,c='#de583a',label='Primary footprint')
        axes[1,1].axhline(25,c='#e6a03a',ls='--',lw=.8,label='25px audit ceiling');axes[1,1].set_xlabel('Side-A retained source arc (px)',fontsize=8)
        axes[1,1].set_ylabel('Gap at unchanged GT (px)',fontsize=8);axes[1,1].set_ylim(bottom=0)
        axes[1,1].grid(alpha=.15);axes[1,1].legend(fontsize=7)
        g=record['detail']['gap'];peak=g['primary_gap_peak_px'];unresolved=max(0.,1-g['ray_resolved_fraction'])
        if projection_receipt is not None:peak=projection_receipt['new_primary_gap_peak_px'];unresolved=projection_receipt['unresolved_fraction']
        value='N/A' if peak is None else f'{peak:.2f}px'
        axes[1,1].set_title(f'Bilateral primary peak {value}; unresolved {100*unresolved:.2f}%',fontsize=9)
    else:
        axes[1,1].axis('off');axes[1,1].text(.05,.8,'Negative pair: NO GT seam / gap.',fontsize=12)
        axes[1,1].text(.05,.6,'Matched material-retention cut;\nrandom position, no positive-pair label.',fontsize=10)
    plan=record['detail']['trim'];fraction=plan['retained_fraction']
    shortening='no GT seam' if fraction is None else f'original support shortened {100*(1-fraction):.1f}%'
    area=100*plan['material_removed_fraction'];donor=plan['donor_index']
    damage=record['detail']['primary_damage'].get(side,{})
    depths=', '.join(f'{x:.1f}' for x in damage.get('requested_peak_depths_px',[])) or 'none'
    weak=damage.get('weak_peak_px',0);k=record.get('requested_gap_count',0)
    axes[1,0].set_title(f'Side {side.upper()}: major peaks [{depths}]px; weak {weak:.1f}px; K={k}\nGray=cut / red=primary / purple=1–3px light',fontsize=8)
    fig.suptitle(f'{record["id"]} | {record["recipe"]} | {"positive" if ispos else "negative"}\n'
        f'{shortening} | {plan["size_class"]} side {plan["side"].upper()} | area loss {area:.2f}% <=20%\n'
        f'TRAIN donor {donor}: {plan["donor"]["lineage"]} | {"one" if plan["mode"]=="one" else "two"} curved boundaries',fontsize=10)
    out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=130);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--out',required=True);p.add_argument('--projection')
    a=p.parse_args();bundle=Path(a.bundle);out=Path(a.out)
    manifest=json.loads((bundle/'manifest.json').read_text());audit=json.loads((bundle/'pixel_audit.json').read_text())
    if audit['status']!='passed':raise ValueError('pixel audit required')
    from .run import GROUP_NAMES
    records=manifest['entries'];groups={};selected=set()
    names=dict(GROUP_NAMES,light70='未腐蚀轮廓70%连续轻微退化（1–3px）',
        crop_smaller='新增自然曲线：裁较小片（70%）',crop_larger='新增自然曲线：裁较大片（30%）')
    for key,label in names.items():
        pool=[r for r in records if key in r['groups'] or (key=='light70' and r['recipe']!='clean')]
        if key in ('native','gen5','union_tiny','unequal','mirror_h','mirror_v') or key.startswith('negative_'):
            chosen=pool[:10]
        else:chosen=[]
        if not chosen:
            chosen=[r for r in pool if r['label']][:5]+[r for r in pool if not r['label']][:5]
        if len(chosen)!=10:raise ValueError('10 actual examples required: '+key)
        groups[key]=dict(label=label,ids=[r['id'] for r in chosen]);selected.update(groups[key]['ids'])
    rows=[];by_audit={r['id']:r for r in audit['receipts']};by_projection={}
    if a.projection:
        correction=json.loads((Path(a.projection)/'projection_audit.json').read_text())
        if correction['status']!='passed':raise ValueError('fixed-pair projection audit must pass')
        by_projection={r['id']:r for r in correction['receipts']}
    for r in records:
        if r['id'] not in selected:continue
        projected=None;receipt=by_projection.get(r['id'])
        if receipt:
            with np.load(Path(a.projection)/(r['id']+'.npz')) as z:projected={k:z[k] for k in z.files}
        image=out/'images'/(r['id']+'.png');render(r,bundle,image,projected,receipt)
        gap=dict(r['detail']['gap']) if r['detail']['gap'] else None
        if receipt:gap.update(primary_gap_peak_px=receipt['new_primary_gap_peak_px'],gap_max_px=receipt['all_resolved_gap_max_px'])
        rows.append(dict(id=r['id'],label='正例' if r['label'] else '负例',recipe=r['recipe'],groups=r['groups'],
            source_stratum=r['source_stratum'],negative_kind=r['negative_kind'],mirror=r['mirror'],
            image='data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode(),
            baseline_pair_id=r['baseline_pair_id'],sample_sha256=r['sample_sha256'],proof_sha256=r['proof_sha256'],
            trim=r['detail']['trim'],gap=gap,paired_gap=receipt or by_audit[r['id']]['paired_gap'],
            background=r['detail']['background'],primary_damage=r['detail']['primary_damage'],
            requested_gap_count=r.get('requested_gap_count',0),previous_review_id=r.get('previous_review_id'),
            unchanged_non_depth_recipe=r.get('unchanged_non_depth_recipe',False),
            old_L40=r['baseline_metrics'].get('d40_length_px'),new_L40=r['new_metrics'].get('d40_length_px'),
            inherited_correspondences=r['inherited_correspondences']))
    out.mkdir(parents=True,exist_ok=True)
    (out/'rendered.json').write_text(json.dumps(dict(groups=groups,rows=rows),ensure_ascii=False)+'\n')
    print(json.dumps(dict(rendered_unique=len(rows),groups=len(groups),per_group=10)))

if __name__=='__main__':main()

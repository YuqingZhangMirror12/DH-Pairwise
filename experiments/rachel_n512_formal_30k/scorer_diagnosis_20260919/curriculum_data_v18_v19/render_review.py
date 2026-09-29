"""Render measured mask evidence; no generated or illustrative replacement masks."""
import argparse,base64,hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.aggressive_data_v17.render_review import plot_pair
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_balanced_v2.latent_seam import ray_project
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.curriculum_data_v18_v19.plan import group_names
from .spec import SPECS

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unpack(z,key):return np.unpackbits(z[key],axis=1).astype(bool)

def render(record,path):
    final,_=load_sample(record['sample_path']);old,_=load_sample(record['baseline_sample_path'])
    with np.load(record['proof_path']) as archive:z={k:archive[k] for k in archive.files}
    base=[unpack(z,'packed_fragment_'+s) for s in 'ab'];new=[getattr(final,'mask_'+s)[0]>0 for s in 'ab']
    positive=bool(record['label']);shift=final.translation_a_to_b_rc if positive else np.array([0.,-900.])
    points=np.vstack([np.argwhere(base[0])+shift,np.argwhere(base[1])]);limits=(points.min(0)-14,points.max(0)+14)
    fig,axes=plt.subplots(2,4,figsize=(20,10.5),layout='constrained')
    plot_pair(axes[0,0],*base,shift,'Pre-trim source / same reference',limits)
    plot_pair(axes[0,1],old.mask_a[0]>0,old.mask_b[0]>0,shift,'Archived v14 / same base',limits)
    crop_stage='primary' if record['recipe']=='partial' else 'trim'
    cut=[unpack(z,'packed_'+crop_stage+'_'+s) for s in 'ab']
    floor=record['detail'].get('crop_only_seam_floor')
    crop_title=('Crop only: common / small perimeter = '+str(round(100*floor['after']['common_over_smaller_perimeter'],2))+'%' if positive else 'Crop only / negative')
    plot_pair(axes[0,2],*cut,shift,crop_title,limits)
    plot_pair(axes[0,3],*new,shift,record['version']+' / final: erosion + light' if positive else record['version']+' / negative, unplaced',limits)
    if positive:
        # Green belongs to the CROP-only panel, not to post-erosion contact.
        p0=z['pristine_a_points'];q0=z['pristine_a_partner_points']
        ia=np.rint(p0).astype(int);ib=np.rint(q0).astype(int)
        alive=cut[0][tuple(ia.T)]&cut[1][tuple(ib.T)]
        p=p0+shift;keep=z['pristine_a_source_edges']&alive&np.roll(alive,-1)
        for i in np.flatnonzero(keep):
            segment=p[[i,(i+1)%len(p)]]
            axes[0,2].plot(segment[:,1],segment[:,0],color='#00a64f',lw=3)
        i=np.flatnonzero(final.target_a>=0);j=final.target_a[i]
        p=final.points_rc_a[i]+shift;q=final.points_rc_b[j]
        for a,b in zip(p,q):axes[0,3].plot([a[1],b[1]],[a[0],b[0]],c='#6d28d9',alpha=.25,lw=.45)
    side=next((s for s,v in record['detail']['primary_damage'].items() if v.get('applied')),record['detail']['trim']['side'])
    before=unpack(z,'packed_fragment_'+side);trim=unpack(z,'packed_trim_'+side)
    primary=unpack(z,'packed_primary_'+side);last=getattr(final,'mask_'+side)[0]>0
    image=np.ones((*before.shape,3));image[before]=[.13,.38,.41]
    image[before&~trim]=[.73,.73,.72];image[trim&~primary]=[.87,.25,.12];image[primary&~last]=[.52,.28,.75]
    ax=axes[1,0];ax.imshow(image,interpolation='nearest')
    p=np.argwhere(before);lo=p.min(0)-8;hi=p.max(0)+8
    ax.set_xlim(lo[1],hi[1]);ax.set_ylim(hi[0],lo[0]);ax.set_aspect('equal');ax.axis('off')
    ax.set_title('Side '+side.upper()+': gray trim / red primary / purple light',fontsize=10)
    other='b' if side=='a' else 'a'
    b0=unpack(z,'packed_fragment_'+other);bt=unpack(z,'packed_trim_'+other)
    bp=unpack(z,'packed_primary_'+other);bf=getattr(final,'mask_'+other)[0]>0
    im=np.ones((*b0.shape,3));im[b0]=[.13,.38,.41];im[b0&~bt]=[.73,.73,.72]
    im[bt&~bp]=[.87,.25,.12];im[bp&~bf]=[.52,.28,.75]
    ax=axes[1,1];ax.imshow(im,interpolation='nearest');xy=np.argwhere(b0);lo=xy.min(0)-8;hi=xy.max(0)+8
    ax.set_xlim(lo[1],hi[1]);ax.set_ylim(hi[0],lo[0]);ax.set_aspect('equal');ax.axis('off')
    ax.set_title('Side '+other.upper()+': gray trim / red primary / purple light',fontsize=10)
    if positive:
        ax=axes[1,2];p=z['gap_a_source_points'];q=z['gap_a_partner_source_points']
        arc=np.cumsum(z['gap_a_source_weight']);valid=z['gap_a_valid'];g=z['gap_a_gap'].copy();g[~valid]=np.nan
        breaks=np.r_[False,np.linalg.norm(np.diff(p,axis=0),axis=1)>3];g[breaks]=np.nan
        ax.plot(arc,g,c='#168b93',lw=1,label=record['version']+' final')
        pa,_,va=ray_project(base[0],old.mask_a[0],p);pb,_,vb=ray_project(base[1],old.mask_b[0],q)
        previous=np.linalg.norm(pa+final.translation_a_to_b_rc-pb,axis=1);previous[~(va&vb)|breaks]=np.nan
        ax.plot(arc,previous,c='#94a3b8',lw=.8,ls='--',label='v14 same original partners')
        active=valid&z['gap_a_primary_affected'];ax.scatter(arc[active],z['gap_a_gap'][active],s=4,c='#e46431',label='main footprint')
        cap=record['detail']['spec']['gap_cap'];ax.axhline(cap,c='#bd8a26',ls=':',label=str(int(cap))+'px ceiling')
        ax.set_ylim(bottom=0);ax.set_xlabel('Remaining original source arc (px)');ax.set_ylabel('Two-sided gap (px)')
        ax.legend(fontsize=7);ax.grid(alpha=.15)
        gap=record['detail']['gap'];ax.set_title('Primary peak '+str(round(gap['primary_gap_peak_px'],2))+'px' if gap['primary_gap_peak_px'] is not None else 'Clean / Partial: no primary-gap quota',fontsize=10)
    else:
        axes[1,2].axis('off');axes[1,2].text(.05,.7,'Negative: NO common-seam GT or gap.\nIndependent crop and corrosion placement.',fontsize=10)
    ax=axes[1,3];ax.axis('off');d=record['detail'];plan=d['trim'];damage=d['primary_damage'].get(side,{})
    lines=[record['id'],record['recipe'], 'Cut side: '+plan['size_class']+' '+plan['side'].upper(),
        'New cut area loss: '+str(round(plan['material_removed_fraction']*100,2))+'% (cap20%)',
        'Target / actual shortening: '+(str(round(plan['target_removed_fraction']*100,2))+'% / '+str(round((1-plan['retained_fraction'])*100,2))+'%' if positive else 'N/A'),
        'Primary requested peaks: '+', '.join(str(round(x,1)) for x in damage.get('requested_peak_depths_px',[])),
        'Weak requested peak: '+str(round(damage.get('weak_peak_px',0),2))+'px',
        'Crop applied: '+str(plan.get('applied',True))+'; '+str(plan.get('skip_reason') or ''),
        'Light: '+str(d['spec']['light'])+'px; 70% WHOLE current contour',
        'Inherited GT matches: '+str(record['inherited_correspondences']),
        'TRAIN donor: '+str(plan['donor_index'])]
    if positive:
        lines+=['CROP gate (before erosion/light):',
            'Before / after common-perimeter ratio: '+str(round(100*floor['before']['common_over_smaller_perimeter'],2))+'% / '+str(round(100*floor['after']['common_over_smaller_perimeter'],2))+'%',
            'After common / smaller perimeter: '+str(round(floor['after']['common_retained_length_px'],1))+' / '+str(round(floor['after']['full_perimeter_px'][floor['after']['smaller_fragment']],1))+'px',
            'Original <20%: '+str(floor['originally_short'])+'; never force new crop']
        pristine=d['pristine_seam']
        lines+=['EXACT paired pristine: '+str(round(100*pristine['conservative_fraction'],2))+'% (descriptive only; no quota)',
            'Original / pristine common arc: '+str(round(pristine['original_common_length_px'],1))+' / '+str(round(pristine['pristine_common_length_px'],1))+'px']
        gap=d['gap'];lines+=['Field coverage A/B: '+str([round(100*x,1) for x in gap['primary_field_coverage_by_side']])+'%',
            'Raster coverage A/B: '+str([round(100*x,1) for x in gap['primary_raster_coverage_by_side']])+'%',
            'Evidence islands A/B: '+str([d['evidence_islands']['sides'][s]['count'] for s in 'ab']),
            'Unresolved source arc: '+str(round((1-gap['ray_resolved_fraction'])*100,2))+'%']
    ax.text(.01,.98,'\n'.join(lines),va='top',fontsize=8,linespacing=1.35)
    fig.suptitle('Curriculum data | original masks + inherited GT | independent pixel audit passed',fontsize=12)
    path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=115);plt.close(fig)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--partial',action='store_true');a=p.parse_args()
 root=Path(a.root);out=Path(a.out)
 if (root/'pipeline_failure.json').exists():raise ValueError('failure needs diagnosis; not render as complete')
 if not a.partial and not (root/'generation_complete.json').exists():raise ValueError('complete generation receipt required')
 rows=[];groups=[];population=[];receipts=[]
 for v in SPECS:
  if not (root/v/'manifest.json').exists():continue
  entries=read(root/v/'manifest.json')['entries'];audit=read(root/v/'pixel_audit.json')
  if audit['status']!='passed' or len(entries)!=audit['pairs']:raise ValueError('all rows require independent audit')
  by_audit={x['id']:x for x in audit['receipts']};selected=set()
  for key,label in group_names(v).items():
   pool=[r for r in entries if key in r['groups']]
   pos=[r for r in pool if r['label']];neg=[r for r in pool if not r['label']]
   chosen=pos[:10]
   if not a.partial and len(chosen)!=10:raise ValueError('10 actual examples required '+v+key)
   if not chosen:continue
   groups.append(dict(id=v+'-'+key,version=v,type=key,label=label,ids=[r['id'] for r in chosen],count=len(chosen)))
   selected.update(r['id'] for r in chosen)
  for r in entries:
   measure=by_audit[r['id']];receipts.append(measure)
   population.append(dict(id=r['id'],version=v,label=r['label'],recipe=r['recipe'],
       gap_peak=measure['paired_gap']['new_primary_gap_peak_px'] if r['label'] else None,
       cut_area_loss=r['detail']['trim']['material_removed_fraction'],
       trim_fraction=1-r['detail']['trim']['retained_fraction'] if r['label'] else None,
       pristine_fraction=measure['pristine_seam']['minimum'] if r['label'] else None,
       crop_applied=r['detail']['trim'].get('applied',True),
       crop_floor=measure.get('crop_only_seam_floor'),
       inherited=r['inherited_correspondences'],source_base_key=r['source_base_key'],
       source_families=r['source_families'],donor_families=r['donor_families']))
   if r['id'] not in selected:continue
   if sha(r['sample_path'])!=r['sample_sha256'] or sha(r['proof_path'])!=r['proof_sha256']:raise ValueError('post-audit artifact changed')
   image=out/'images'/(r['id']+'.png');render(r,image)
   row={key:r[key] for key in ('id','version','recipe','groups','baseline_pair_id','source_stratum','negative_kind','mirror','inherited_correspondences','sample_sha256','proof_sha256','source_base_key')}
   row.update(label='正例' if r['label'] else '负例',detail=r['detail'],audit=measure,
              image='data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode())
   rows.append(row)
 result=dict(complete=not a.partial,rows=rows,groups=groups,population=population,protocol=read(root/'protocol.json'),
             pixel_audit_count=len(receipts),source_root=str(root),
             independent_unique_display=len(rows),display_slots=sum(g['count'] for g in groups))
 out.mkdir(parents=True,exist_ok=True)
 (out/'rendered.json').write_text(json.dumps(result,ensure_ascii=False)+'\n')
 print(json.dumps({k:result[k] for k in ('complete','pixel_audit_count','independent_unique_display','display_slots')}))
if __name__=='__main__':main()

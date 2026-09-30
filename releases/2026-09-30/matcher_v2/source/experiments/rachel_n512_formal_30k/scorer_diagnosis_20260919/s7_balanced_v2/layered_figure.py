"""Literal three-stage v13 masks; primary and light effects shown separately."""
import base64
from dataclasses import replace
import numpy as np
import matplotlib.pyplot as plt
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
from ..gap_distribution_v2.showcases import display
from .conservative_weather import NAMES


def zoom(ax,before,after,title,points=None):
    before=before.astype(bool);after=after.astype(bool)
    rgb=np.ones((*before.shape,3));rgb[before]=[.68,.72,.73];rgb[after]=[.12,.30,.35];rgb[before&~after]=[.90,.28,.19]
    ax.imshow(rgb,interpolation='nearest');changed=np.argwhere(before&~after)
    if points is None:
        lo=changed.min(0);hi=changed.max(0);center=(lo+hi)/2;radius=max(20.,float(np.max(hi-lo)/2)+8)
    else:center=points;radius=65.
    ax.set_xlim(center[1]-radius,center[1]+radius);ax.set_ylim(center[0]+radius,center[0]-radius)
    ax.set_aspect('equal');ax.set_title(title,fontsize=10)


def figure(root,entry,row,out,key):
    sample,report=load_sample(root/entry['artifact_path'])
    with np.load(root/entry['weather_artifact'],allow_pickle=False) as data:primary={k:data[k] for k in data.files}
    light={}
    if entry.get('background_artifact'):
        with np.load(root/entry['background_artifact'],allow_pickle=False) as data:light={k:data[k] for k in data.files}
    masks=[np.unpackbits(primary['packed_preweather_'+s],axis=1).astype(np.float32)[None] for s in 'ab']
    before=replace(sample,mask_a=masks[0],mask_b=masks[1],translation_a_to_b_rc=primary['translation_a_to_b_rc'])
    middle={s:np.unpackbits(light['packed_before_'+s],axis=1).astype(bool) if light else getattr(sample,'mask_'+s)[0].astype(bool) for s in 'ab'}
    fig,axs=plt.subplots(3,2,figsize=(13,11),gridspec_kw={'height_ratios':[1.25,1,1]})
    left,right,mainzoom,maincurve,microzoom,microcurve=axs.ravel()
    suffix='same GT pose' if sample.label else 'negative / NO GT layout'
    display(left,before,'After fragment transforms, before ALL damage / '+suffix)
    display(right,sample,'Final: primary damage + light layer / '+suffix)
    right.set_xlim(left.get_xlim());right.set_ylim(left.get_ylim())
    side=next((s for s in 'ab' if s+'_total' in primary),None)
    partial=row['recipe']=='partial'
    if side or partial:
        if partial:side=report['compound']['partial']['sides'][0 if sample.label else 1]
        zoom(mainzoom,getattr(before,'mask_'+side)[0],middle[side],'Primary stage ONLY: red = removed material')
        if partial:
            maincurve.axis('off');info=report['compound']['partial']
            description=('Original seam retained: '+f'{info["source_seam_retention"]:.1%}' if sample.label else 'Matched material-retention crop; no GT seam')
            if info.get('mode'):
                description='Mode: '+info['mode']+'\n'+description
                if sample.label:
                    check=info['support_constraint'];smaller=check['smaller_fragment']
                    description+='\nCommon retained curve / smaller perimeter:\n'+f'{check["common_retained_length_px"]:.1f} / {check["full_perimeter_px"][smaller]:.1f}px = {check["common_over_smaller_perimeter"]:.1%} (>=15%)'
                    if info['mode']=='middle':
                        description+='\nRetained flanks: '+ ' / '.join(f'{v:.1f}' for v in info['retained_flanks_px'])+'px'
                        description+=f'\nRemoved original middle arc: {info["removed_middle_arc_px"]:.1f}px'
                description+='\nMeasured after crop, BEFORE the light layer.'
            maincurve.text(.04,.93,'Partial crop; NO primary weather overlay\n'+description+'\nOnly the final1-3px light exception is allowed.',va='top',fontsize=10)
        else:
            arc=primary[side+'_arc'];total=primary[side+'_total'];edge=primary[side+'_edge'];perimeter=float(edge.sum())
            active=np.sort(arc[total>0]);gaps=np.diff(np.r_[active,active[0]+perimeter]);start=active[(np.argmax(gaps)+1)%len(active)]
            u=(arc-start+10)%perimeter;ids=np.argsort(u);ids=ids[u[ids]<=perimeter-gaps.max()+20]
            for field,color,label in (('major','#216778','Major'),('weak','#ad762c','Primary weak1-4px'),('total','#cb4836','Primary combined')):
                maincurve.plot(u[ids],primary[side+'_'+field][ids],color=color,label=label,lw=1.3)
            maincurve.set_ylim(-.2,10);maincurve.legend(fontsize=8);maincurve.grid(alpha=.2)
            maincurve.set_xlabel('Source contour arc (px)');maincurve.set_ylabel('Recession (px)')
            maincurve.set_title('Primary stage field; max9px, affected arc<=50%',fontsize=10)
    else:
        for ax in (mainzoom,maincurve):ax.axis('off')
        mainzoom.text(.05,.8,'Clean15%: no primary damage\nNo Partial crop and no light layer',va='top',fontsize=12)
    if light:
        side='a';points=light[side+'_points'];total=light[side+'_total'];center=points[int(np.argmax(total))]
        zoom(microzoom,middle[side],sample.mask_a[0],'Extra light ONLY: red =1-3px recession (fragment A)',center)
        arc=light[side+'_arc'];eligible=light[side+'_eligible']
        microcurve.fill_between(arc,0,3,where=~eligible,color='#ddd',alpha=.55,label='Already damaged / new cut: excluded')
        microcurve.plot(arc,total,color='#216778',lw=1.2,label='Continuous progressive field')
        microcurve.set_ylim(-.1,3.2);microcurve.set_xlabel('Remaining contour arc (px)');microcurve.set_ylabel('Extra recession (px)')
        microcurve.legend(fontsize=7);microcurve.grid(alpha=.2)
        fractions=[report['background_degradation'][s]['actual_affected_fraction'] for s in 'ab']
        microcurve.set_title(f'Actual untouched-arc coverage: A{fractions[0]:.1%} / B{fractions[1]:.1%}',fontsize=10)
    else:
        for ax in (microzoom,microcurve):ax.axis('off')
        microzoom.text(.05,.8,'Clean sample unchanged:0px /0%\nSource scale and mirror may still apply',va='top',fontsize=12)
    version='v14' if (report['compound']['partial'] or {}).get('mode') or 'partial_original_points_rc_a' in primary else 'v13'
    # Non-Partial recipes are unchanged and are still part of the same revision.
    if 'v14' in str(root):version='v14'
    fig.suptitle(f'Layered {version} | {row["recipe"]} | {"POSITIVE" if sample.label else "NEGATIVE"} | source fragment transforms FIRST',fontsize=13)
    fig.tight_layout();file=out/(key+'.png');fig.savefig(file,dpi=115);plt.close(fig)
    caption=f"{NAMES[row['recipe']]}；{'正例，保持原GT摆放' if sample.label else '明确负例，分开展示，没有GT摆放'}。上方为同尺度全部损伤前后；中行为主损伤，底行为新增轻退化，两层红色删除区域分开显示。"
    if partial:
        caption+='Partial不叠加主腐蚀或1–4px旧弱层，仅允许新增1–3px轻退化。'
        info=report['compound']['partial']
        if info.get('mode'):
            caption+='本例为'+('中段切除、两侧保留' if info['mode']=='middle' else '端部裁切')+'。'
            if sample.label:
                check=info['support_constraint'];small=check['smaller_fragment']
                caption+=f'裁切后公共原接缝{check["common_retained_length_px"]:.1f}px／面积较小碎片的完整周长{check["full_perimeter_px"][small]:.1f}px＝{check["common_over_smaller_perimeter"]:.2%}；在轻退化前计量，删去中段和新切边不计入接缝。'
    if light:
        caption+='新增层实际覆盖未腐蚀原轮廓：'+ '；'.join(f'{s.upper()}侧{report["background_degradation"][s]["actual_affected_fraction"]:.2%}' for s in 'ab')+'。新切边和已损伤轮廓不计入该分母。'
    else:caption+='该15%无腐蚀类别未施加任何退化。'
    return dict(id=key,pair_id=entry['pair_id'],recipe=row['recipe'],name=NAMES[row['recipe']],
        label=bool(sample.label),partial=bool(row['partial']),partial_mode=(report['compound']['partial'] or {}).get('mode'),caption=caption,
        source_stratum=entry['source_stratum'],artifact=str(root/entry['artifact_path']),
        weatherSource=str(root/entry['weather_artifact']),backgroundSource=str(root/entry['background_artifact']) if light else None,
        image='data:image/png;base64,'+base64.b64encode(file.read_bytes()).decode())

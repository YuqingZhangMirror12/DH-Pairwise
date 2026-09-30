"""Mask-backed v4.2 reproduction audit and exactly thirty review figures."""
import argparse
from collections import Counter
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,obj):
    with Path(path).open('x') as stream:
        json.dump(obj,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def compose(ma,mb,t):
    ra,ca=np.nonzero(ma);rb,cb=np.nonzero(mb)
    rb=rb-int(t[0]);cb=cb-int(t[1])
    r0=min(ra.min(),rb.min())-15;c0=min(ca.min(),cb.min())-15
    h=max(ra.max(),rb.max())+16-r0;w=max(ca.max(),cb.max())+16-c0
    img=np.ones((h,w,3));occupied=np.zeros((h,w),bool)
    img[ra-r0,ca-c0]=(.62,.78,.95);occupied[ra-r0,ca-c0]=True
    img[rb-r0,cb-c0]=np.where(occupied[rb-r0,cb-c0,None],(.45,.15,.55),(.98,.76,.50))
    return img,np.array([r0,c0])


def render(entry,sample,out):
    ma,mb=sample.mask_a[0]>0,sample.mask_b[0]>0
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),dpi=140,gridspec_kw={'width_ratios':[1.65,1]})
    if entry['label']:
        t=sample.translation_a_to_b_rc
        image,origin=compose(ma,mb,t)
        for ax in axes:ax.imshow(image,interpolation='nearest');ax.set_aspect('equal');ax.axis('off')
        pts=sample.points_rc_a[sample.target_a>=0]
        mid=np.median(pts,axis=0)-origin
        axes[1].set_xlim(mid[1]-45,mid[1]+45);axes[1].set_ylim(mid[0]+45,mid[0]-45)
        axes[1].plot([mid[1]-38,mid[1]-18],[mid[0]+37]*2,color='#222',lw=2)
        axes[1].text(mid[1]-38,mid[0]+33,'20 px',fontsize=9)
        axes[0].set_title('Actual masks at exact GT (not model output)')
        axes[1].set_title('90 x 90 px seam crop / nearest pixels')
    else:
        for ax,mask,color,side in zip(axes,[ma,mb],[(.62,.78,.95),(.98,.76,.5)],'AB'):
            r,c=np.nonzero(mask);image=np.ones((*mask.shape,3));image[mask]=color
            ax.imshow(image,interpolation='nearest');ax.set_xlim(c.min()-15,c.max()+15);ax.set_ylim(r.max()+15,r.min()-15)
            ax.set_aspect('equal');ax.axis('off');ax.set_title('Independent fragment '+side+' / no GT assembly')
    fig.suptitle(entry['id']+' | '+entry['meta']['base']+' | v4.2 reference geometry, new seed',fontsize=11)
    fig.tight_layout();fig.savefig(out,facecolor='white');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path);p.add_argument('--metrics-code',required=True,type=Path)
    p.add_argument('--official-code',required=True,type=Path);p.add_argument('--output-new',required=True,type=Path)
    p.add_argument('--render',action='store_true');a=p.parse_args()
    sys.path.insert(0,str(a.official_code))
    from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
    spec=importlib.util.spec_from_file_location('frozen_metrics',a.metrics_code);metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
    root=a.root.resolve();manifest=json.loads((root/'manifest.json').read_text());complete=json.loads((root/'generation_complete.json').read_text())
    assert complete['status']=='generated' and complete['manifest_sha256']==sha(root/'manifest.json') and not manifest['failed']
    entries=manifest['entries'];assert len(entries)==complete['expected']
    assert len({x['pair_id'] for x in entries})==len(entries)
    out=a.output_new.resolve();out.mkdir(parents=True,exist_ok=False)
    errors=[];records=[];loaded={};input_hashes=[]
    for e in entries:
        path=Path(e['sample_path']);assert sha(path)==e['sample_sha256']
        s,report=load_sample(path)
        if a.render:
            loaded[e['pair_id']]=s
        assert s.pair_id==e['pair_id'] and bool(s.label)==e['label']
        for side in 'ab':
            mask=getattr(s,'mask_'+side);assert mask.shape==(1,800,800) and set(np.unique(mask))<={0.,1.}
        h=hashlib.sha256()
        for key in ['mask_a','mask_b','points_rc_a','points_rc_b','contour_valid_a','contour_valid_b']:
            value=np.ascontiguousarray(getattr(s,key));h.update(key.encode());h.update(value.tobytes())
        input_hashes.append(h.hexdigest())
        if e['label']:
            assert sha(e['proof_path'])==e['proof_sha256']
            with np.load(e['proof_path']) as proof:
                shape=tuple(proof['shape'])
                for side in 'ab':
                    parent=np.unpackbits(proof['final_parent_'+side],axis=-1)[:,:shape[1]].astype(bool)
                    labs,n=ndimage.label(parent);sizes=np.bincount(labs.ravel());sizes[0]=0;parent=labs==sizes.argmax()
                    r,c=np.nonzero(parent);shift=proof['shift_'+side].astype(int)
                    predicted=np.zeros((800,800),bool);predicted[r+shift[0],c+shift[1]]=True
                    if not np.array_equal(predicted,getattr(s,'mask_'+side)[0]>0):errors.append(e['id']+':placed pixels')
                if not np.array_equal(proof['shift_b']-proof['shift_a'],s.translation_a_to_b_rc):errors.append(e['id']+':GT translation')
            assert np.sum(s.target_a>=0)>=8
        else:
            assert not s.translation_valid and np.all(s.target_a==-1) and np.all(s.target_b==-1)
        records.append(dict(pair_id=e['pair_id'],id=e['id'],kind=e['recipe'][-1],positive=e['label'],
            base=e['meta']['base'],metrics=metrics.measure(s),matches=int((s.target_a>=0).sum()),
            ignored_a=int((s.target_a==-2).sum()),ignored_b=int((s.target_b==-2).sum())))
    if len(set(input_hashes))!=len(input_hashes):errors.append('duplicate model inputs')
    census={}
    for kind in 'MJR':
        rows=[r for r in records if r['kind']==kind and r['positive']]
        q={}
        for field in ['extent_px','bend_range','rough_std','gap_frac_over3','slide_m10','slide_m20','slide_m40']:
            values=[r['metrics']['seam'][field] for r in rows if r['metrics']['seam'] and r['metrics']['seam'][field] is not None]
            q[field]={str(p):float(np.percentile(values,p)) for p in [10,25,50,75,90]} if values else None
        census[kind]=dict(positives=len(rows),quantiles=q,bases=dict(Counter(r['base'] for r in rows)))
    audit=dict(status='passed' if not errors else 'failed',scope='source-bound file, official-loader, reciprocal targets, parent-pixel centering, exact GT and input uniqueness',
        rows=len(entries),errors=errors,manifest_sha256=sha(root/'manifest.json'),records=records,census=census,
        training_admitted=False,model_inference=False,
        supervision_caveat='Reference targets use mutual distance; explicit exclusion/ignore of all damaged seam intervals is not proven. Visual reproduction only.',
        sampling_caveat='Exact reference retry behavior redraws difficulty parameters. Distribution checks, not configured intervals alone, are required.',
        metrics_code_sha256=sha(a.metrics_code),finished_unix=time.time())
    save(out/'audit.json',audit)
    if errors:raise RuntimeError(str(errors))
    if a.render:
        assert manifest['mode']=='review30' and len(entries)==30
        blocks=[]
        for kind in 'MJR':
            subset=sorted([e for e in entries if e['recipe']=='straight_'+kind],key=lambda e:(not e['label'],e['id']))
            assert len(subset)==10
            for i,e in enumerate(subset):
                filename=e['id']+'.png';render(e,loaded[e['pair_id']],out/filename)
                marker='正样本 · GT拼合' if e['label'] else '负样本 · 独立展示，无GT拼合'
                row=next(r for r in records if r['pair_id']==e['pair_id'])
                m=row['metrics']['seam'];stats='' if m is None else f'接缝长 {m["extent_px"]:.1f}px；局部粗糙度 {m["rough_std"]:.2f}px；20px滑动失配 {m["slide_m20"]:.2f}px'
                blocks.append(f'<section id="{kind}-{i+1}"><h2>{kind} · {i+1}/10 · {marker}</h2><p>{html.escape(e["meta"]["base"])} · {stats}</p><img loading="lazy" src="{filename}" alt="{kind} 第{i+1}组实际掩膜及接缝放大"><details><summary>来源、参数与独立测量</summary><pre>'+html.escape(json.dumps(dict(id=e['id'],seed=manifest['seed'],meta=e['meta'],measurements=row),ensure_ascii=False,indent=2))+'</pre></details></section>')
        page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v4.2 方法复用 · 新种子直缝审核30组</title><style>body{font:16px/1.65 system-ui,sans-serif;color:#213640;background:#f3f6f8;max-width:1250px;margin:28px auto;padding:0 20px}header,section{background:white;border:1px solid #dce3e8;border-radius:12px;padding:22px;margin:18px 0}h1{font-size:27px}h2{font-size:20px}img{display:block;width:100%;height:auto}pre{overflow:auto;font-size:12px}.notice{background:#fff4dc;border-left:4px solid #c98c1c;padding:12px}nav{position:sticky;top:0;background:#eff6ff;padding:12px;z-index:2}a{color:#07599c}details{margin-top:10px}</style>
<header><h1>v4.2 方法复用 · 新种子直缝审核</h1><p>M / J / R 每类10组，共30组；每类8正、2负。蓝色为A片，橙色为B片，紫色为重叠，白色为间隙。</p><p>直接复用你提供的 v4.2 形状函数：相关噪声不再除以3，缺口按整段渐变收窄；不改原参考文件。M/J仿真碎片来自TRAIN来源白名单；R及J少量困难尾部为程序合成。已排除疑似同写卷来源u4761_frag3。</p><p class="notice">这30组是新种子的视觉小试，不是模型预测，也不代表全量分布或训练准入。旧参考仅按距离恢复对应点，全部损伤段的显式忽略仍需单独审计。既有GPU训练未改。</p><p>负例仅并列展示，不能当作GT拼接。接缝局部图固定为90×90像素、最近邻显示，无视觉平滑。样本按预先确定的ID与J子类覆盖选择，不按模型分数挑图。</p>'''
        page+=f'<p>新种子：{manifest["seed"]}；来源/像素/GT检查：30/30；HTML只展示本页30组。</p></header><nav><a href="#M-1">M · 10组</a>　<a href="#J-1">J · 10组</a>　<a href="#R-1">R · 10组</a></nav>'+''.join(blocks)+'</html>'
        (out/'index.html').write_text(page)
        save(out/'render_complete.json',dict(status='complete',rows=30,per_type=10,positives_per_type=8,
            html_sha256=sha(out/'index.html'),audit_sha256=sha(out/'audit.json'),manifest_sha256=sha(root/'manifest.json'),
            training_admitted=False,model_predictions=False))
    print(json.dumps(dict(status='complete',output=str(out),rows=len(entries),census=census)))


if __name__=='__main__':main()

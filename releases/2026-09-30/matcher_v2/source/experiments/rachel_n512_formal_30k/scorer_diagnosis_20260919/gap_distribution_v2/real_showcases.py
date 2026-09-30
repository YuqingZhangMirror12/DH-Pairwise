"""Illustrative GT cases selected by observed gap, not model success/failure."""
import argparse
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .measure import read,save,raw_outline,side_distances
from .showcases import display,seam_center


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--masks',required=True)
    a=p.parse_args();root=Path(a.root);out=root/'real_showcases';out.mkdir(exist_ok=True)
    rows=[json.loads(l) for l in (root/'Dunhuang292/pairs.jsonl').read_text().splitlines()]
    fragments=read(a.masks)['fragments'];records=[];used=set()
    for target in (2.,5.,12.):
        row=min((r for r in rows if r['pair_id'] not in used and r['seam40_gap']['n']),
            key=lambda r:abs(r['seam40_gap']['mean']-target));used.add(row['pair_id'])
        masks=[]
        for side in 'ab':
            f=fragments[row['fragment_'+side+'_id']];x,y,w,h=f['bbox_xywh']
            crop=np.array(Image.open(io.BytesIO(base64.b64decode(f['png'].split(',')[1]))))[:,:,3]>0
            assert crop.shape==(h,w)
            mask=np.zeros((800,800),np.float32);mask[y:y+h,x:x+w]=crop
            assert int(mask.sum())==row['area_px_'+side]
            masks.append(mask[None])
        sample=SimpleNamespace(mask_a=masks[0],mask_b=masks[1],translation_a_to_b_rc=np.array(row['translation_gt_rc']),label=True)
        fig,axes=plt.subplots(1,3,figsize=(14,4.8));display(axes[0],sample,'Dunhuang / original GT')
        display(axes[1],sample,'GT seam zoom / actual Patch widths',seam_center(sample))
        aa,bb=[raw_outline(m[0]) for m in masks]
        for x,y,shift,origin in ((aa,bb,-sample.translation_a_to_b_rc,np.zeros(2)),(bb,aa,sample.translation_a_to_b_rc,-sample.translation_a_to_b_rc)):
            d,_=side_distances(x,y,shift);points=x['points']+origin
            scatter=axes[2].scatter(points[:,1],points[:,0],c=d,cmap='viridis',vmin=0,vmax=64,s=2)
        axes[2].invert_yaxis();axes[2].set_aspect('equal');axes[2].set_title('All-contour GT distance / saturated at64px',fontsize=10);axes[2].axis('off')
        fig.colorbar(scatter,ax=axes[2],shrink=.7,label='Nearest other contour (px)')
        fig.suptitle('Dunhuang GT | '+row['pair_id'][:26],fontsize=12);fig.tight_layout()
        path=out/(f'gap-{target:g}.png');fig.savefig(path,dpi=150,bbox_inches='tight');plt.close(fig)
        records.append(dict(id='real-'+str(target),title=f'敦煌GT · 间隙约{target:g}px案例',pair_id=row['pair_id'],
            caption=f"实际GT；以每对近接带平均距离接近{target:g}px选取，用于说明形态，不代表随机样本或模型结果。此例双向近接点距离均值{row['seam40_gap']['mean']:.2f}px，面积{row['area_px_a']:,}/{row['area_px_b']:,}px²；右侧整圈距离颜色在64px饱和，统计中未截断。",
            image='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode(),
            source_artifact=a.masks,label=True,recipe='real GT'))
    save(out/'showcases.json',records);print(json.dumps(dict(rendered=len(records))))


if __name__=='__main__':main()

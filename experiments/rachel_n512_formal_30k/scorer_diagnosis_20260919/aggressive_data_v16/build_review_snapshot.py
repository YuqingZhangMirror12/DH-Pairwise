"""Replace only the reviewed dataset in the existing stable8782 Data report."""
import argparse,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(metric,values,unit):
    x=np.asarray(values,float);q=np.quantile(x,[0,.1,.5,.9,1])
    return dict(metric=metric,n=len(x),unit=unit,mean=float(x.mean()),
        **dict(zip(('min','p10','p50','p90','max'),map(float,q))))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--app',required=True)
    lifecycle=p.add_mutually_exclusive_group()
    lifecycle.add_argument('--complete',action='store_true')
    lifecycle.add_argument('--paused',action='store_true',help='preview authoring awaits the background CPU review batch')
    p.add_argument('--probe',action='store_true')
    a=p.parse_args();root=Path(a.root);app=Path(a.app)
    bundle=root/('probe_audit_01' if a.probe else 'review_bundle_01');render=root/('probe_rendered_01' if a.probe else 'rendered_01')
    manifest=read(bundle/'manifest.json');audit=read(bundle/'pixel_audit.json');rendered=read(render/'rendered.json')
    assert audit['status']=='passed' and len(manifest['entries'])==audit['pairs']
    if not a.probe:assert len(rendered['groups'])==26 and all(len(g['ids'])==10 for g in rendered['groups'].values())
    positives=[r for r in manifest['entries'] if r['label']];n=len(positives)
    assert positives and all(r['detail']['trim'].get('endpoint_audit') for r in positives)
    size_counts=audit['positive_size_counts']
    if not a.probe:assert size_counts['smaller']*10==n*7
    records=manifest['entries'];by_audit={r['id']:r for r in audit['receipts']}
    values=[r['detail']['trim'] for r in records]
    rows=[stats('原可继承接缝裁短',[100*(1-r['detail']['trim']['retained_fraction']) for r in positives],'%'),
        stats('裁较小片：新增裁切面积损失',[100*r['material_removed_fraction'] for r in values if r['size_class']=='smaller'],'%'),
        stats('裁较大片：新增裁切面积损失',[100*r['material_removed_fraction'] for r in values if r['size_class']=='larger'],'%'),
        stats('最终L40近接曲线缩短',[100*(1-r['new_metrics']['d40_length_px']/r['baseline_metrics']['d40_length_px']) for r in positives],'%')]
    gap=[by_audit[r['id']]['paired_gap'] for r in positives]
    strong=[g for g in gap if g['new_primary_gap_peak_px'] is not None]
    if strong:rows+=[stats('强腐蚀区：新双侧gap峰值',[g['new_primary_gap_peak_px'] for g in strong],'px')]
    rows+=[stats('未解析源弧占比',[g['unresolved_fraction']*100 for g in gap],'%')]
    snapshot=read(app/'src/data.json');snapshot.update(status='curved-v16-awaiting-human-review',
        buildStatus='complete' if a.complete else ('paused' if a.paused else 'updating'),generatedAt=datetime.now(timezone.utc).isoformat(),
        reviewStatus=(f'{len(records)}对自然曲线预检已通过逐像素审计；仅预览，完整每类10例审核小试仍在制作' if a.probe else
            f'{len(records)}对自然曲线小试已通过逐像素审计；等待人工审核，不扩量、不启动新训练'),
        curveRevision=dict(version=16,pairs=len(records),positive_pairs=n,side_counts=size_counts,
            display_groups=len(rendered['groups']),unique_display_pairs=len(rendered['rows']),
            bank_profiles=768,bank_source_families=128,partial_preview=a.probe,endpoint_only_verified=True),
        reviewProvenance=dict(manifest_sha256=sha(bundle/'manifest.json'),pixel_audit_sha256=sha(bundle/'pixel_audit.json'),
            supersedes='v15 straight-cut and v16 source01–03 interior-cut pilots retained, not accepted for expansion'))
    snapshot['queries']['pilot_cases']=dict(rows=rendered['rows'],source=dict(label='v16自然轮廓裁短：实际像素和独立审计',
        tables=[str(bundle/'manifest.json'),str(bundle/'pixel_audit.json'),str(render/'rendered.json')],
        notes=['70%小片30%大片；两类新增裁切面积均≤20%；公共继承曲线目标20%、栅格实测19–21%。',
            '只用TRAIN写卷轮廓库；保存每例曲线、来源及SHA，不使用真实评价集轮廓。',
            '按两侧原始公共轮廓顺序复核：只能去掉一端或两端，新增裁切不允许删去中段。',
            '负例不报告虚构接缝或GT间隙；旧v15不复用。',
            '原始对应在裁切前固定，不将未解析投影填零。']))
    snapshot['queries']['pilot_groups']=dict(rows=[dict(id=k,**v) for k,v in rendered['groups'].items()],
        source=dict(label='增强类型固定选例规则',tables=['aggressive_data_v16/render_review.py'],
            notes=[(f'目前仅{len(records)}对预检预览，非完整覆盖；完整审核目标每类10例。' if a.probe else
                '每组10例，组间可能共用样本；不是260对独立样本，不按图像美观选例。')]))
    snapshot['queries']['pilot_summary']=dict(rows=rows,source=dict(label='全体已审计小试，不仅统计展示案例',
        tables=['pilot_01/manifest.json','review_bundle_01/pixel_audit.json'],
        notes=['按样本等权分位数；L40与继承源曲线长度分开。面积损失只统计本次新增裁切，不含已有腐蚀。']))
    (app/'src/data.json').write_text(json.dumps(snapshot,ensure_ascii=False)+'\n')
    (root/'review_receipt.json').write_text(json.dumps(dict(curveRevision=snapshot['curveRevision'],
        summary=rows,provenance=snapshot['reviewProvenance']),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(snapshot['curveRevision'],ensure_ascii=False))

if __name__=='__main__':main()

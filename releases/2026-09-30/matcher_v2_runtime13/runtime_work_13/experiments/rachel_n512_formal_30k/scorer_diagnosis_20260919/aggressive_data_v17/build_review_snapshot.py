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
    if not a.probe:assert len(rendered['groups'])==27 and all(len(g['ids'])==10 for g in rendered['groups'].values())
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
    if strong:rows+=[stats('主损伤区（含弱腐蚀）：新双侧gap峰值',[g['new_primary_gap_peak_px'] for g in strong],'px')]
    rows+=[stats('未解析源弧占比',[g['unresolved_fraction']*100 for g in gap],'%')]
    damage=[v for r in records for v in r['detail']['primary_damage'].values() if v.get('applied')]
    weak=[v['weak_peak_px'] for v in damage if v['weak_applied']]
    major=[p for v in damage for p in v['requested_peak_depths_px']]
    rows+=[stats('单侧弱腐蚀设定峰值（按施加侧）',weak,'px'),stats('单侧主腐蚀设定峰值（按损伤区域）',major,'px')]
    snapshot=read(app/'src/data.json');snapshot.update(status='depth-v17-awaiting-human-review',
        buildStatus='complete' if a.complete else ('paused' if a.paused else 'updating'),generatedAt=datetime.now(timezone.utc).isoformat(),
        reviewStatus=(f'{len(records)}对新深度预检通过逐像素审计；完整分组仍在制作' if a.probe else
            f'{len(records)}对v17深度修订通过逐像素审计；保留已审核裁切，等待本轮深度人工审核，不扩量'),
        curveRevision=dict(version=17,pairs=len(records),positive_pairs=n,side_counts=size_counts,
            display_groups=len(rendered['groups']),unique_display_pairs=len(rendered['rows']),
            bank_profiles=768,bank_source_families=128,partial_preview=a.probe,endpoint_only_verified=True,
            approved_cut_pixels_unchanged=True,primary_gap_peak_range_px=[5,25],notch_count_range=[1,4]),
        reviewProvenance=dict(manifest_sha256=sha(bundle/'manifest.json'),pixel_audit_sha256=sha(bundle/'pixel_audit.json'),
            supersedes='v16 endpoint_revision_01 cuts preserved exactly; primary depth and notch count revised, old outputs retained'))
    snapshot['queries']['pilot_cases']=dict(rows=rendered['rows'],source=dict(label='v17深度修订：原裁切逐像素不变、实际腐蚀与双侧间隙审计',
        tables=[str(bundle/'manifest.json'),str(bundle/'pixel_audit.json'),str(render/'rendered.json')],
        notes=['70%小片30%大片；两类新增裁切面积均≤20%；公共继承曲线目标20%、栅格实测19–21%。',
            '只用TRAIN写卷轮廓库；保存每例曲线、来源及SHA，不使用真实评价集轮廓。',
            '按两侧原始公共轮廓顺序复核：只能去掉一端或两端，新增裁切不允许删去中段。',
            '单侧弱腐蚀3–8px，主腐蚀5–15px，缺口1–4处；同侧主＋弱合成封顶15px。',
            '主损伤区最终双侧间隙峰值5–25px，不是整缝最小间隙；负例不虚构GT。',
            '清洁与Partial对照数值归档原样保留；1–3px轻退化规则不变。',
            '原始对应在裁切前固定，不将未解析投影填零。']))
    snapshot['queries']['pilot_groups']=dict(rows=[dict(id=k,**v) for k,v in rendered['groups'].items()],
        source=dict(label='增强类型固定选例规则',tables=['aggressive_data_v17/render_review.py'],
            notes=[(f'目前仅{len(records)}对预检预览，非完整覆盖；完整审核目标每类10例。' if a.probe else
                '27组各10例，组间可能共用样本；270展示槽不是270对独立样本。实际K4另列10例，不冒称训练配额。')]))
    snapshot['queries']['pilot_summary']=dict(rows=rows,source=dict(label='全体已审计小试，不仅统计展示案例',
        tables=['pilot_01/manifest.json','review_bundle_01/pixel_audit.json'],
        notes=['按样本等权分位数；L40与继承源曲线长度分开。面积损失只统计本次新增裁切，不含已有腐蚀。']))
    snapshot['queries']['depth_protocol']=dict(rows=[
        dict(kind='连续弱腐蚀',old='峰值1–4px',new='峰值3–8px',shape='原连续渐进／平滑内收与肩部过渡不变'),
        dict(kind='起伏退蚀',old='峰值5–9px',new='5≤峰值<15px',shape='连续起伏、受影响弧长上限不变'),
        dict(kind='局部突变',old='峰值5–9px',new='峰值5–15px',shape='保留局部突然内缩'),
        dict(kind='局部渐进',old='峰值5–9px',new='峰值5–15px',shape='约1px起步，连续变深与起伏不变'),
        dict(kind='缺口',old='1–3处，每处峰值5–9px',new='1–4处，每处峰值5–15px',shape='每处必须独立移除实际像素；合计弧长≤50%')],
        source=dict(label='用户2026-09-27深度修订与实际生成协议',
            tables=[str(bundle/'protocol.json'),str(bundle/'pixel_audit.json'),'aggressive_data_v17/PROTOCOL.md'],
            notes=['连续均匀采样上界不含端点；接受样本经过像素/拓扑筛选，不能把接受后的深度分布称均匀。',
                '同侧主＋弱合成封顶15px；补充1–3px轻退化及clean/Partial规则不变。',
                '每个正例主损伤区双侧最终gap峰值5–25px；不是单侧场，也不是整缝最小5px。']))
    (app/'src/data.json').write_text(json.dumps(snapshot,ensure_ascii=False)+'\n')
    (root/'review_receipt.json').write_text(json.dumps(dict(curveRevision=snapshot['curveRevision'],
        summary=rows,provenance=snapshot['reviewProvenance']),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(snapshot['curveRevision'],ensure_ascii=False))

if __name__=='__main__':main()

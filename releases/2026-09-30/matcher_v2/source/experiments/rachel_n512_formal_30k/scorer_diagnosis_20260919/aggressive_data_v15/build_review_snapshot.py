"""Bind the reviewed pilot and test receipts into the existing Data app."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row(metric,values,unit):
    x=np.asarray(values,float);q=np.quantile(x,[0,.1,.5,.9,1])
    return dict(metric=metric,n=len(x),unit=unit,mean=float(x.mean()),
        **dict(zip(('min','p10','p50','p90','max'),map(float,q))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--app',required=True);p.add_argument('--complete',action='store_true');a=p.parse_args()
    root=Path(a.root);app=Path(a.app);bundle=root/'review_bundle_01';projection=root/'projection_audit_01'
    manifest=read(bundle/'manifest.json');audit=read(bundle/'pixel_audit.json');fixed=read(projection/'projection_audit.json')
    rendered=read(root/'rendered_02'/'rendered.json');records=manifest['entries'];positives=[r for r in records if r['label']]
    assert audit['status']==fixed['status']=='passed' and len(records)==226 and len(positives)==113
    assert len(rendered['groups'])==24 and all(len(g['ids'])==10 for g in rendered['groups'].values())
    assert len(rendered['rows'])==len(set(r['id'] for r in rendered['rows']))==136
    model_root=app.parent
    for environment in ('local','remote'):
        receipt=read(model_root/('cpu_tests_source02_'+environment+'.json'))
        # Keep the full upstream receipt; its explicit test run was already
        # inspected. Do not infer formal GPU success from this CPU evidence.
        assert receipt
    source_short=[100*(1-r['detail']['trim']['retained_fraction']) for r in positives]
    final_short=[100*(1-r['new_metrics']['d40_length_px']/r['baseline_metrics']['d40_length_px']) for r in positives]
    strong=[r for r in fixed['receipts'] if r['new_primary_gap_peak_px'] is not None]
    assert len(strong)==56
    rows=[row('原可继承接缝裁短',source_short,'%'),row('最终L40近接曲线缩短',final_short,'%'),
        row('新：强腐蚀段双侧gap峰值',[r['new_primary_gap_peak_px'] for r in strong],'px'),
        row('旧v14：同位置双侧gap峰值',[r['old_primary_gap_peak_px'] for r in strong],'px'),
        row('未解析源弧比例',[100*r['unresolved_fraction'] for r in fixed['receipts']],'%')]
    increased=sum(r['new_primary_gap_peak_px']>r['old_primary_gap_peak_px'] for r in strong)
    provenance=dict(bundle_manifest_sha256=sha(bundle/'manifest.json'),pixel_audit_sha256=sha(bundle/'pixel_audit.json'),
        projection_audit_sha256=sha(projection/'projection_audit.json'),actual_pairs=len(records),positive_pairs=len(positives),
        selected_unique_pairs=len(rendered['rows']),display_groups=len(rendered['groups']),display_slots=240,
        stronger_peak_pairs=increased,strong_peak_pairs=len(strong),
        mask_or_label_changes_for_projection_correction=False)
    snapshot=read(app/'src'/'data.json');snapshot.update(generatedAt=datetime.now(timezone.utc).isoformat(),
        status='awaiting-human-review',buildStatus='complete' if a.complete else 'creating',
        reviewStatus='226对小试已完成，逐像素审计及固定原始对应的gap复核均通过；等待人工审核',
        reviewProvenance=provenance)
    snapshot['queries']['pilot_cases']=dict(rows=rendered['rows'],source=dict(label='226对分层小试中的136对展示样本：实际掩膜及配对v14',
        tables=['pilot_03/manifest.json','review_bundle_01/pixel_audit.json','projection_audit_01/projection_audit.json','rendered_02/rendered.json'],
        notes=['24组×10个展示位置，共136对不同样本；同一对可用于多个增强分组，不能当240个独立样本。',
            '峰值限定主腐蚀影响区；平滑肩部和未腐蚀段可以小于5px。负例无GT接缝，gap为null。',
            '原始伙伴在裁短前固定；源映射不确定与射线失败保留在分母、单列未解析，不补成零。',
            '原像素审计的旧gap rematch值保留供审计，当前展示用独立修正后的原始固定伙伴投影。']))
    snapshot['queries']['pilot_groups']=dict(rows=[dict(id=k,**v) for k,v in rendered['groups'].items()],
        source=dict(label='预先登记的增强类型与固定选例顺序',tables=['aggressive_data_v15/run.py:GROUP_NAMES','rendered_02/rendered.json'],
            notes=['腐蚀和裁短类各前5正5负；碎片组前10例；负例组前10负例。不是按图像美观挑选。']))
    snapshot['queries']['pilot_summary']=dict(rows=rows,source=dict(label='全体113正例／其中56强腐蚀正例，不仅统计展示案例',
        tables=['pilot_03/manifest.json','projection_audit_01/projection_audit.json'],
        notes=['按样本等权分位数，不是弧长加权分位数；单例gap已分别记录原弧分布。',
            'L40缩短和继承原始接缝裁短是两个不同定义，不互换。',
            '有33/56个强腐蚀正例的峰值高于配对v14；并非每个样本都加深。']))
    snapshot['queries']['model_protocol']['source']['tables']=['cpu_tests_source02_local.json','cpu_tests_source02_remote.json','binary_scorer_v1/PROTOCOL.md']
    (app/'src'/'data.json').write_text(json.dumps(snapshot,ensure_ascii=False)+'\n')
    (root/'review_receipt.json').write_text(json.dumps(dict(provenance=provenance,summary=rows),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(provenance,ensure_ascii=False))


if __name__=='__main__':main()

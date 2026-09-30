"""Mechanical append of verified stage queries; preserve all older report rows."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def canonical(x): return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--app',type=Path,required=True)
    p.add_argument('--artifact',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--build-status',choices=['updating','complete'],required=True);a=p.parse_args()
    analysis=a.artifact/'three_light_models_analysis_01.json'
    projection=a.artifact/'three_light_models_report_queries_01.json'
    verification=a.artifact/'three_light_models_recount_01.json'
    bundle, check=read(projection),read(verification)
    assert bundle['status']=='model_complete' and check['status']=='passed'
    assert bundle['source_analysis_sha256']==check['analysis_sha256']==sha(analysis)
    assert set(check['models'])==set(bundle['completed_models']) and len(check['models'])==3
    target=a.app/'src/data.json';original_sha=sha(target);annotation_sha=sha(a.annotations)
    data=read(target);assert data['id']=='report:4b51ee5a-d8bb-4663-8305-69763a99c639'
    historic={k:canonical(v) for k,v in data['queries'].items() if not k.startswith('decoder_factor_')}
    before=a.artifact/'report_data_before_factor_controls.json'
    if not before.exists(): shutil.copyfile(target,before)
    data['queries'].update(bundle['queries'])
    labels={r['model']:r['model_label'] for r in bundle['queries']['decoder_factor_models']['rows']}
    data['queries']['decoder_factor_admission']=dict(rows=[dict(model=model,model_label=labels[model],
        **{k:v for k,v in result.items() if k!='all_modes_vs_baseline'},**result['all_modes_vs_baseline'])
        for model,result in check['models'].items()],source=dict(type='file',
            files=[verification.name,analysis.name],evidenceFlow=[dict(title='原始记录独立计数与逐例相等性',
            detail=f'复算SHA256：{sha(verification)}；分析SHA256：{sha(analysis)}。独立核对12,600项指标、8,100组配对ID和180个CAL阈值。')],
            caveats=['只展示3个完整模型；复杂头不在本节。','完整文件、模型不变及真实Linux退出码0已核验；父控制器依次wait，因此另三份父退出回执尚待补写。','2619=敦煌639＋Turufan480＋本模型仿真SELECT1500；不同模型包含同样样本，不作为独立样本相加。']))
    data['buildStatus']=a.build_status;data['generatedAt']=datetime.now(timezone.utc).isoformat()
    assert all(canonical(data['queries'][k])==v for k,v in historic.items())
    assert sha(target)==original_sha and sha(a.annotations)==annotation_sha
    pending=target.with_name('data.factor-stage.pending.json')
    with pending.open('x') as f: json.dump(data,f,ensure_ascii=False,separators=(',',':'),allow_nan=False);f.write('\n')
    assert sha(target)==original_sha and sha(a.annotations)==annotation_sha
    os.replace(pending,target)
    assert sha(a.annotations)==annotation_sha
    receipt=dict(status='bound',build_status=a.build_status,app_id=data['id'],
        report_data_before_sha256=original_sha,report_data_after_sha256=sha(target),
        historical_query_sha256=historic,annotation_sha256=annotation_sha,
        projection_sha256=sha(projection),verification_sha256=sha(verification),
        training_changed=False,remote_writes=False)
    out=a.artifact/('report_binding_'+a.build_status+'.json')
    with out.open('x') as f: json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='historical_query_sha256'}))


if __name__=='__main__': main()

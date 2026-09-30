"""Build the reviewed report snapshot from frozen inference and diagnostic rows."""
import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent;DIAG=HERE.parent
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def source(label,files,caveats=()):return dict(label=label,files=[str(x) for x in files],caveats=list(caveats))
def heat(m):
    m=np.asarray(m,dtype=float);z=np.clip((np.log10(np.maximum(m,1e-6))+6)/6,0,1)
    # Fixed sequential palette; identical transfer function for every panel.
    anchors=np.array([[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]],float)
    rgba=np.stack([np.interp(z,np.linspace(0,1,5),anchors[:,k]) for k in range(3)],-1).astype(np.uint8)
    s=BytesIO();Image.fromarray(rgba).save(s,format='PNG')
    return dict(png='data:image/png;base64,'+base64.b64encode(s.getvalue()).decode(),shape=list(m.shape),
                min=float(m.min()),max=float(m.max()),color='fixed purple-teal-yellow palette, shared log10 scale [-6,0]; values<=1e-6 clipped for display')


def run(a):
    root=Path(a.root);c=read(root/'comparison.json');metrics=[]
    labels={'v3_B22':'v3 · B22','S7_M12_matched_C16':'旧 S7 · 端点 C16'}
    for split,d in c['splits'].items():
        for key,m in d['models'].items():
            for policy in ('SIM冻结','真实五折'):
                v=m['sim_frozen'] if policy=='SIM冻结' else m['cv']['bounded_max_f1']['pooled_out_of_fold']
                ts=[m['sim_frozen']['threshold']] if policy=='SIM冻结' else [f['threshold'] for f in m['cv']['bounded_max_f1']['folds']]
                metrics.append(dict(dataset='敦煌' if split=='dunhuang_cv' else 'Turufan',model=labels[key],policy=policy,
                    threshold=float(np.median(ts)),thresholds=ts,accuracy=v['accuracy'],precision=v['precision'],recall=v['recall'],
                    f1=v['f1'],auroc=v['auroc'],tp=v['tp'],fp=v['fp'],fn=v['fn'],tn=v['tn'],
                    layout=v.get('layout_correct_total'),joint=v.get('layout_correct_accepted'),
                    n=v['n'],positive=v['positive'],negative=v['negative'],primary=True))
    old=read(DIAG/'bounded_real_calibration_v2/results/results.json')
    for split,d in old['results'].items():
        for key,label in [('joint_D_h4','旧 D · 双任务监督'),('gcn_pairing_h4','旧 GCN · Pairing式'),('gcn_shredding_h4','旧 GCN · Shredding式')]:
            m=d['models'][key]
            for policy in ('SIM冻结','真实五折'):
                v=m['old_sim'] if policy=='SIM冻结' else m['cv']['bounded_max_f1']['pooled_out_of_fold']
                ts=[m['old_sim_threshold']] if policy=='SIM冻结' else [f['threshold'] for f in m['cv']['bounded_max_f1']['folds']]
                metrics.append(dict(dataset='敦煌' if split=='real' else 'Turufan',model=label,policy=policy,
                    threshold=float(np.median(ts)),thresholds=ts,accuracy=v['accuracy'],precision=v['precision'],recall=v['recall'],
                    f1=v['f1'],auroc=v['auroc'],tp=v['tp'],fp=v['fp'],fn=v['fn'],tn=v['tn'],
                    layout=v.get('layout_correct_total'),joint=v.get('layout_correct_accepted'),n=v['n'],positive=v['positive'],negative=v['negative'],primary=False))
    evidence=read(root/'mechanism_analysis.json');cohorts=[]
    for split,groups in evidence['cohorts'].items():
        for name,r in groups.items():
            cohorts.append(dict(dataset=split,group=name,n=r['n'],single_edge=r['single_edge'],up_to8=r['up_to8'],
                median_edges=r['edges']['q10_q25_median_q75_q90'][2] if r['edges']['n'] else None))
    q={
      'metrics':dict(rows=metrics,source=source('冻结 B22 与历史模型，同一来源分组五折',
          [root/'comparison.json',DIAG/'bounded_real_calibration_v2/results/results.json'],
          ['阈值中位数仅用于概览；实际每个测试Pair使用其折外阈值。','敦煌295人工保留正例+508构造负例；Turufan301正例+301跨来源构造负例。',
           '历史模型/人工剔除曾依据真实数据作研究选择，不能称为全新盲测。','GCN指我们评分器的聚合变体，不是原论文的完整模型。'])),
      'support':dict(rows=cohorts,source=source('赢家候选的对应数量',[root/'mechanism_analysis.json'],['对应数量不是GT接缝长度；分组使用冻结仿真阈值0.47。'])),
      'gradients':dict(rows=read(root/'gradient_probe/records.json'),source=source('60个训练样本的只读梯度探针',[root/'gradient_probe/protocol.json'],['分层诊断样本；负余弦不是权重共享有害的因果证明。'])),
    }
    if (root/'report_inputs/sim_strict_source_disjoint.json').exists():
        sim=read(root/'report_inputs/sim_strict_source_disjoint.json')
        q['simulation']=dict(rows=[sim['metrics']],source=source('SIM TEST去除训练/CAL/SELECT同源Pair',[root/'report_inputs/sim_strict_source_disjoint.json'],['剩2400对，1299正/1101负；不是原始3000对。']))
    if a.full:
        inputs=read(root/'report_inputs/masks_and_pairs.json');oof=read(root/'oof_predictions.json')
        names=read(root/'report_inputs/local_path_receipt.json')['cases']
        def fragment_name(token):
            cid=token.split('/')[0]
            if cid in names:
                occurrence=names[cid]['occurrences'][0]
                folder=Path(occurrence['group_directory'])
                return f"{occurrence['collection']}/{folder.parent.name}/{folder.name} · {token.rsplit('/',1)[-1]}"
            return token.rsplit('/',1)[-1]
        bpath=root/'probes_baseline_strict';base={r['pair_id']:r for r in read(bpath/'predictions.json')}
        assert len(base)==1405
        cases=[]
        for split,meta in inputs['manifests'].items():
            predictions={r['pair_id']:r for r in rows(root/split/'case_diagnostics.jsonl')}
            previous={r['pair_id']:r for r in read(root/'baseline'/('matched_tokens_real_cv.json' if split=='dunhuang_cv' else 'matched_tokens_turufan.json'))['rows']}
            folds={key:{r['pair_id']:r for r in oof[split][key]['bounded_max_f1']} for key in labels}
            for pair in meta['pairs']:
                r=predictions[pair['pair_id']];b=base[r['pair_id']];wo=r['candidates'][r['winner_index']] if r['has_candidate'] else None
                models={}
                for key in labels:
                    f=folds[key][r['pair_id']]
                    sim=c['splits'][split]['models'][key]['sim_frozen']['threshold']
                    isnew=key=='v3_B22';p=r if isnew else b
                    models[key]=dict(score=f['score'],translation=p['translation'],threshold_cv=f['threshold'],
                        threshold_sim=sim,accepted_cv=f['accepted'],decision_valid=bool(p.get('numeric_valid',p.get('decision_valid'))),
                        layout_error_px=r['error_px'] if isnew else previous[r['pair_id']].get('layout_error_px'),
                        layout20=r['layout20'] if isnew and r['gt_known'] else previous[r['pair_id']].get('layout_good_20'))
                # Cached original scores remain authoritative; exact-runtime rerun supplies masks/pose/heatmaps.
                cases.append(dict(pair_id=r['pair_id'],dataset='敦煌' if split=='dunhuang_cv' else 'Turufan',
                    label=bool(pair['label']),fragment_a=pair['fragment_a_id'],fragment_b=pair['fragment_b_id'],
                    case_name=' + '.join(fragment_name(pair[k]) for k in ('fragment_a_id','fragment_b_id')),fold=pair.get('fold'),
                    gt=r['target_translation_rc'],models=models,candidates=r['candidates'],
                    edge_count=wo['edge_count'] if wo else 0,quality=wo['quality_logit'] if wo else None,
                    null=wo['null_logit'] if wo else None,residual=wo['residual_median_px'] if wo else None))
        q['cases']=dict(rows=cases,source=source('逐例冻结推理 + 原始Mask缓存',[root/'oof_predictions.json',bpath/'predictions.json',root/'report_inputs/masks_and_pairs.json'],
            ['B相对于A绘制位移为-trans_A_to_B_rc。','Turufan无GT Layout，不能把看起来合理当成正确。','原始Pair score不经温度或其它缩放。']))
        q['fragments']=dict(rows=[dict(fragment_id=k,**v) for k,v in inputs['fragments'].items()],
            source=source('既有800px二值输入缓存',[root/'report_inputs/masks_and_pairs.json'],['无新增平滑或缩放。']))
        pp=read(root/'probes_v3/records.json');repair={p['pair_id']:p for p in read(root/'probes_v3_turufan_unique/records.json')}
        pp=[repair.get(p['pair_id'],p) for p in pp]
        bb={p['pair_id']:p for p in read(bpath/'probe_records.json')};heatrows=[]
        for p in pp:
            b=bb[p['pair_id']]
            keys=['q0','q1','arc_context.blocks.0.cross_attn_0','arc_context.blocks.3.cross_attn_0','verifier.blocks.0.cross_attn_0','verifier.blocks.1.cross_attn_0']
            maps={k:heat(p['heatmaps'][k]) for k in keys if k in p['heatmaps']}
            maps['old_q']=heat(b['heatmaps']['q'])
            oldattn=next((k for k in b['heatmaps'] if 'attention' in k and k.endswith('_0')),None)
            if oldattn:maps['old_attention']=heat(b['heatmaps'][oldattn])
            oldlast=next((k for k in reversed(list(b['heatmaps'])) if 'attention' in k and k.endswith('_0')),None)
            if oldlast:maps['old_attention_last']=heat(b['heatmaps'][oldlast])
            heatrows.append(dict(pair_id=p['pair_id'],reason=p['reason'],maps=maps,pooling=p['pooling'],
                interventions=p['diagnostic_interventions'],proxies=p['proxies'],broad_edges=p['broad_edges'],
                raw_array_file=str(root/('probes_v3_turufan_unique' if p['split']=='turufan' else 'probes_v3')/p['raw_array_file'])))
        q['heatmaps']=dict(rows=heatrows,source=source('36例真实前向输出与逐层注意力',[root/'probes_v3/records.json',root/'probes_v3_turufan_unique/records.json',bpath/'probe_records.json'],
            ['按历史失败/新旧差异分层选取，不代表总体发生率。','热图以最大值合并到最多96×96；统一log10色阶[-6,0]。','注意力不等同因果显著性；Q与Attention数值语义不同。']))
    snapshot=dict(title='连续接缝 v3：摆放改善，评分仍漏判',status='reviewed',surface='report',buildStatus='complete',
        report=dict(asOf='2026-09-23'),generatedAt=datetime.now(timezone.utc).isoformat(),filters=[],queries=q)
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        current=read(path)
        if 'id' in current:snapshot['id']=current['id']
    path.write_text(json.dumps(snapshot,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
    print({k:len(v['rows']) for k,v in q.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--full',action='store_true');run(p.parse_args())

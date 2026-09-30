"""Three fixed Matcher checkpoints on the identical paired hard-SIMVAL rows."""
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
RESULTS = OUT.parent.parent
DL = RESULTS.parent.parent
PATHS = {'M12':DL/'matcher_convergence/hard_validation/full_v1_m12',
         'M16':RESULTS/'matcher_m16/hard_validation',
         'M20':RESULTS/'matcher_m20/hard_validation'}
RECIPES = ('wave','local','seam_gaps','partial_curve')
LOSSES = ('assignment_nll','match_nll','dustbin_nll')


def read(path):
    return json.loads(path.read_text())


def mean(values):
    assert values and all(isinstance(x,(float,int)) and math.isfinite(x) for x in values)
    return sum(values)/len(values)


def metrics(rows):
    return dict(count=len(rows),positive_count=sum(r['label'] for r in rows),
        layout20_correct=sum(r['raw_layout20_correct'] for r in rows),
        layout20_denominator=sum(r['label'] for r in rows),
        invalid_layout_count=sum(not r['raw_layout_valid'] for r in rows if r['label']),
        training_valid_count=sum(r['training_valid'] for r in rows),
        supervised_match_pair_count=sum(r['supervised_correspondence_count']>0 for r in rows),
        supervised_A_correspondence_tokens=sum(r['supervised_correspondence_count'] for r in rows),
        pose_supervised_pairs=sum(r['pose_supervised'] for r in rows),
        mean_nll_per_pair={key:mean([r['losses'][key] for r in rows]) for key in LOSSES},
        nll_pair_denominator=len(rows),dustbin_token_denominator=None)


def transitions(old,new,keys):
    cells={x+'_to_'+y:sorted(i for i in keys if old[i]['raw_layout20_correct']==(x=='correct')
        and new[i]['raw_layout20_correct']==(y=='correct')) for x in ('correct','failed') for y in ('correct','failed')}
    assert sum(map(len,cells.values()))==len(keys)
    return dict(count=len(keys),cells=cells,rescued=len(cells['failed_to_correct']),
        lost=len(cells['correct_to_failed']),
        net=len(cells['failed_to_correct'])-len(cells['correct_to_failed']))


loaded,protocols,summaries={}, {}, {}
for name,path in PATHS.items():
    protocols[name],summaries[name]=read(path/'protocol.json'),read(path/'summary.json')
    assert protocols[name]['status']==summaries[name]['status']=='complete'
    assert protocols[name]['completed_count']==summaries[name]['count']==6000
    rows=[json.loads(x) for x in (path/'pair_metrics.jsonl').read_text().splitlines()]
    loaded[name]={r['pair_id']:r for r in rows}
    assert len(loaded[name])==len(rows)==6000
    assert summaries[name]['classifier_metrics_reported'] is False
    assert summaries[name]['model']['trained_scorer_head_called'] is False
    assert summaries[name]['model']['epoch']==int(name[1:])
    for row in rows:
        error=row['raw_translation_l2_px']
        correct=row['label'] and row['raw_layout_valid'] and error is not None and error<=20
        assert row['raw_layout20_correct']==bool(correct)

common=('manifest_sha256','source_val_manifest_sha256','pair_order_sha256','evaluator_sha256',
        'loss_source_sha256','original_evaluator_sha256','decoder_config','device','batch_size','precision')
row_identity=('label','source_pair_id','recipe','changed_pair','pose_supervision_enabled','pose_supervised',
              'fallback_reason','artifact_path','artifact_sha256','source_family_overlap','supervised_correspondence_count')
for name in ('M16','M20'):
    assert loaded[name].keys()==loaded['M12'].keys()
    assert all(protocols[name][k]==protocols['M12'][k] for k in common)
    assert all(all(loaded[name][i][k]==loaded['M12'][i][k] for k in row_identity) for i in loaded['M12'])

reference=loaded['M12']
groups={
    'clean_positive':[i for i,r in reference.items() if r['label'] and r['recipe']=='clean'],
    'requested_positive':[i for i,r in reference.items() if r['label'] and r['recipe']!='clean'],
    'changed_positive':[i for i,r in reference.items() if r['label'] and r['changed_pair']],
    'fallback_requested_positive':[i for i,r in reference.items() if r['label'] and r['recipe']!='clean' and not r['changed_pair']]}
for recipe in RECIPES:
    groups['requested_'+recipe]=[i for i,r in reference.items() if r['label'] and r['recipe']==recipe]
    groups['changed_'+recipe]=[i for i,r in reference.items() if r['label'] and r['recipe']==recipe and r['changed_pair']]
report=dict(status='complete',manifest_sha256=protocols['M12']['manifest_sha256'],source_paths={k:str(v) for k,v in PATHS.items()},
    shared_protocol_fields=list(common),shared_row_identity_fields=list(row_identity),count_per_model=6000,
    models={n:s['model'] for n,s in summaries.items()},groups={},paired_clean_to_requested={},
    missing_candidate_rank_fields=True,model_inference_performed=False,scorer_evaluated=False,
    caveats=['Clean and requested variants share source_pair_id and are not independent samples.',
        'All positive rows remain in Layout20 denominator, including invalid layouts and disabled pose supervision.',
        'NLL means average already-normalized pair losses; dustbin token denominator was not recorded.',
        'Disabled pose/total loss is not compared as a quality improvement.',
        'No GT-candidate ranking fields were saved; no new inference was performed.',
        'This fixed-checkpoint diagnostic does not select a model using REAL/OOD and does not establish Scorer performance.'])
for group,keys in groups.items():
    report['groups'][group]=dict(metrics={n:metrics([rows[i] for i in keys]) for n,rows in loaded.items()},
        transitions={a+'_to_'+b:transitions(loaded[a],loaded[b],keys)
                     for a,b in [('M12','M16'),('M16','M20'),('M12','M20')]})
for name,rows in loaded.items():
    clean={r['source_pair_id']:r for r in rows.values() if r['recipe']=='clean'}
    report['paired_clean_to_requested'][name]={}
    for recipe in RECIPES:
        variants=[r for r in rows.values() if r['recipe']==recipe and r['label']]
        old={r['pair_id']:clean[r['source_pair_id']] for r in variants}
        new={r['pair_id']:r for r in variants}
        report['paired_clean_to_requested'][name][recipe]=transitions(old,new,list(new))
        official=summaries[name]['metrics']['by_recipe'][recipe]['positive_pairs']['mean_losses']
        assert all(math.isclose(mean([r['losses'][k] for r in variants]),official[k],abs_tol=1e-10) for k in LOSSES)
(OUT/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
lines=['# Fixed Matcher M12→M16→M20: hard-SIMVAL','',
    '同一6000条独立仿真验证记录（3000个source，每个clean/requested各一条）；没有Scorer/分类阈值。','',
    '| 正例组 | N | M12 Layout20 | M16 | M20 | M16→20 救回/丢失 |',
    '|---|---:|---:|---:|---:|---:|']
for group,g in report['groups'].items():
    m=g['metrics'];t=g['transitions']['M16_to_M20']
    lines.append('| %s | %d | %d | %d | %d | %d/%d |'%(group,m['M12']['count'],
        m['M12']['layout20_correct'],m['M16']['layout20_correct'],m['M20']['layout20_correct'],t['rescued'],t['lost']))
lines+=['','结果：Layout20没有随12→16→20持续改善。M20配对监督NLL下降，不等于最终解算准确率提高。',
        'clean/requested不是独立数据集，不将二者简单累计解释为新增独立样本。',
        '逐例四格转移、相同source的clean→requested转移、NLL及监督分母见results.json。',
        '未保存正确候选rank字段，不能报告TopK命中；此处不涉及仍未开始的M20 Scorer。','']
(OUT/'FINDINGS.md').write_text('\n'.join(lines))
print('\n'.join(lines))

"""Register source-isolated real development/test roles without reading scores."""
import hashlib
import json
from pathlib import Path

DIAG=Path(__file__).resolve().parents[1]
ROLES={'real_cal':(1,), 'real_select':(2,3,4), 'real_test':(0,)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def partition(meta, excluded):
    pairs=meta['pairs'];source=meta['fragment_source_group']
    if len({r['pair_id'] for r in pairs})!=len(pairs):
        raise ValueError('duplicate real pair')
    if not excluded<=set(r['pair_id'] for r in pairs):
        raise ValueError('GT exclusion missing from source')
    result={};fragment_sets={};source_sets={}
    for role,folds in ROLES.items():
        rr=[r for r in pairs if r['fold'] in folds and r['pair_id'] not in excluded]
        fragments={r[k] for r in rr for k in ('fragment_a_id','fragment_b_id')}
        groups={source[f] for f in fragments}
        if not any(r['label'] for r in rr) or all(r['label'] for r in rr):
            raise ValueError('both classes required in each registered role')
        fragment_sets[role]=fragments;source_sets[role]=groups
        result[role]=dict(folds=list(folds),pair_ids=[r['pair_id'] for r in rr],
            pairs=len(rr),positive=sum(bool(r['label']) for r in rr),
            negative=sum(not r['label'] for r in rr),fragments=len(fragments),
            source_groups=sorted(groups),source_count=len(groups))
    for i,a in enumerate(ROLES):
        for b in list(ROLES)[i+1:]:
            if fragment_sets[a]&fragment_sets[b] or source_sets[a]&source_sets[b]:
                raise ValueError('real role source/fragment leakage')
    if sum(v['pairs'] for v in result.values())!=len(pairs)-len(excluded):
        raise ValueError('partition lost/duplicated pairs')
    return result


def main():
    root=Path('artifacts/consensus_threshold_joint_20260927')
    root.mkdir(parents=True,exist_ok=True)
    cases=json.loads((DIAG/'s7_consensus_eval_v14/case_plan.json').read_text())
    rows={}
    for ds,sub in (('dunhuang_cv','real'),('turufan','ood')):
        local=Path('artifacts/DH_v3_B22_diagnosis_20260923/_incoming')/(ds+'_manifest.json')
        meta=json.loads(local.read_text())
        excluded=set(cases['user_confirmed_gt_exclusions']) if ds=='dunhuang_cv' else set()
        roles=partition(meta,excluded)
        rows[ds]=dict(local_manifest=str(local),manifest_sha256=sha(local),
            remote_manifest='/root/autodl-tmp/rachel_score_design_20260913_001/real_domain_calibration_v1_20260921/'+sub+'/manifest.json',
            prepared=meta['prepared'],excluded_gt_pair_ids=sorted(excluded),
            original_pairs=len(meta['pairs']),layout_gt_available=ds=='dunhuang_cv',roles=roles)
    output=dict(schema='threshold-joint-real-split/1',status='registered',
        development_evaluation=True,previous_real_design_exposure=True,
        model_scores_read=False,new_epoch_results_read=False,source_disjoint=True,
        protocol_sha256=sha(Path(__file__).with_name('PROTOCOL.md')),
        role_folds={k:list(v) for k,v in ROLES.items()},datasets=rows,
        gt_path='/root/autodl-tmp/rachel_same_data_final_eval_exact6_20260906_004/real/translation-gt-attempt-001.json')
    dest=root/'real_split.json'
    if dest.exists() and json.loads(dest.read_text())!=output:
        raise ValueError('do not silently replace a registered real split')
    dest.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status='registered',sha256=sha(dest),datasets={d:{r:{k:v for k,v in s.items() if k not in ('pair_ids','source_groups')} for r,s in x['roles'].items()} for d,x in rows.items()}),ensure_ascii=False))


if __name__=='__main__':main()

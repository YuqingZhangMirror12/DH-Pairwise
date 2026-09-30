"""Only audited, source-isolated 30K may receive a training data contract."""
from collections import Counter
from pathlib import Path
from . import COUNTS,REVISION
from ..s7_compound_v1.materialize import read,save_json,digest
from ..seam_context_v3.prepare import family,sources

KEEP=('pair_id','label','recipe','corrosion_recipe','corrosion_category','source_pair_id','source_root',
      'source_row','source_stratum','negative_kind','offline_paired_mirror','partial_applied',
      'sample_path','artifact_path','sample_sha256','target_metadata','target_metadata_sha256',
      'latent_seam_artifact','inherited_match_count','augmentation_donor_sources','augmentation_revision','v14_fallback')


def inherited_duplicate(first, later, first_split, later_split):
    """Only unchanged same-fold v14 repetitions may survive retention.

    Never resample, perturb or relabel the user's original pairs to make a
    uniqueness counter zero. Cross-fold, newly-created and contradictory
    duplicates remain failures. Compare all numeric fields, not only masks.
    """
    if first_split != later_split:
        raise ValueError('duplicate content crosses data splits')
    if not (first.get('v14_fallback') is True and later.get('v14_fallback') is True):
        raise ValueError('newly-created duplicate content is not inherited v14')
    if (first['source_pair_id'] != later['source_pair_id'] or first['label'] != later['label']
            or first['pair_id'] == later['pair_id']):
        raise ValueError('duplicate original pair/label/ID lineage differs')
    from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
    from .fallback import numerical_identity
    a,_=load_sample(first['sample_path']);b,_=load_sample(later['sample_path'])
    old_a,_=load_sample(first['baseline_sample_path']);old_b,_=load_sample(later['baseline_sample_path'])
    numerical_identity(a,old_a);numerical_identity(b,old_b);numerical_identity(a,b)
    return dict(first_pair_id=first['pair_id'],later_pair_id=later['pair_id'],split=later_split,
        source_pair_id=later['source_pair_id'],label=later['label'],
        first_baseline_path=first['baseline_sample_path'],later_baseline_path=later['baseline_sample_path'],
        first_baseline_sha256=digest(first['baseline_sample_path']),later_baseline_sha256=digest(later['baseline_sample_path']),
        same_original_v14_numerical_content=True,model_inputs_labels_weights_unchanged=True)


def finish(config):
    out=Path(config['out']);specs={};used={};donors={};audit_shas={};summaries={}
    all_ids=set();seen_inputs={};inherited_duplicates=[]
    for split,count in COUNTS.items():
        d=out/split;plan=read(out/'plans'/(split+'.json'));entries=[];source=set();donor=set()
        recipe=Counter();sides=Counter();negative=Counter();notches=Counter();group_audits=[];fallbacks=0;planned_sides=Counter();planned_k=Counter()
        for task in plan['tasks']:
            name=f'{task["slot"]:05d}.json';group=read(d/'groups'/name);audit=read(d/'audits'/name)
            if group['status']!='committed' or audit['status']!='passed' or audit['group_sha256']!=digest(d/'groups'/name):
                raise ValueError('unverified committed group')
            if group['task']!=task or group['plan_sha256']!=config['plan_hashes'][split]:
                raise ValueError('fixed slot quotas changed')
            if len(audit['rows'])!=2:raise ValueError('incomplete pixel/target audit')
            planned_sides[task['size_class']]+=1
            if task['k']:planned_k[str(task['k'])]+=1
            fallbacks+=int(group['v14_fallback'])
            group_audits.append(dict(path=str(d/'audits'/name),sha256=digest(d/'audits'/name)))
            for record,receipt in zip(group['records'],audit['rows']):
                if receipt['status']!='passed' or record['id']!=receipt['id']:
                    raise ValueError('audit pair identity differs')
                for key in ('sample','proof'):
                    if digest(record[key+'_path'])!=record[key+'_sha256'] or receipt[key+'_sha256']!=record[key+'_sha256']:
                        raise ValueError('archive changed after audit')
                if record['pair_id'] in all_ids:raise ValueError('duplicate Pair ID')
                all_ids.add(record['pair_id'])
                content=receipt['model_input_sha256']
                if content in seen_inputs:
                    first_split,first_record=seen_inputs[content]
                    inherited_duplicates.append(inherited_duplicate(first_record,record,first_split,split))
                else:seen_inputs[content]=(split,record)
                source |= sources(record['source_row']);donor |= {family(x) for x in record['augmentation_donor_sources']}
                if record['label']:
                    recipe[record['recipe']]+=1
                    if not record['v14_fallback']:sides[record['detail']['trim']['size_class']]+=1
                    if record['requested_gap_count']:notches[str(record['requested_gap_count'])]+=1
                else:negative[record['negative_kind']]+=1
                entries.append({k:record[k] for k in KEEP})
        if len(entries)!=count or sum(e['label'] for e in entries)!=count//2:
            raise ValueError('exact balanced split count failed')
        if dict(recipe)!=plan['baseline_recipe_counts'] or dict(planned_sides)!=plan['side_counts'] or dict(planned_k)!=plan['notch_counts']:
            raise ValueError('accepted sample quotas changed')
        old_contract=read(config['baseline_contract'])
        allowed=set(old_contract['source_families'][split if split=='train' else split+'_mixed'])
        if not source<=allowed:raise ValueError('source left registered fold')
        if split!='train' and not donor<=allowed:raise ValueError('heldout donor crossed folds')
        if split=='train' and donor & set().union(*(set(old_contract['source_families'][s+'_mixed']) for s in ('cal','select','test'))):
            raise ValueError('TRAIN donor heldout leak')
        used[split]=source;donors[split]=donor
        record=dict(schema_version='rachel-materialized-e1-train/1',split=split,artifact_root=str(d),
            augmentation_revision=REVISION,not_full_training_dataset=False,entries=entries)
        manifest=d/(split+'.json');save_json(manifest,record)
        archive=d/'archive_manifest.json';save_json(archive,record)
        aud=d/'full_pixel_audit.json';save_json(aud,dict(status='passed',checked_pairs=count,
            manifest_sha256=digest(manifest),group_receipts=group_audits,
            actual_loader_and_supervision_checked_all=True,all_pixels_and_endpoints_checked=True))
        audit_shas[split]=dict(path=str(aud),sha256=digest(aud))
        specs[split]=dict(path=str(manifest),sha256=digest(manifest),pair_count=count,
            archive_manifest=str(archive),archive_manifest_sha256=digest(archive))
        summaries[split]=dict(pairs=count,positives=count//2,negatives=count//2,
            recipe_counts=dict(recipe),cut_side_counts=dict(sides),notch_counts=dict(notches),
            v14_fallback_pairs=2*fallbacks,v17_applied_pairs=count-2*fallbacks,
            planned_cut_side_counts=dict(planned_sides),planned_notch_counts=dict(planned_k),
            negative_kind=dict(negative),source_families=len(source),donor_families=len(donor),
            inherited_v14_duplicate_occurrences=sum(x['split']==split for x in inherited_duplicates))
    all_sources=set()
    for split in COUNTS:
        if all_sources & used[split]:raise ValueError('manuscript leakage between splits')
        all_sources |= used[split]
    old=read(config['baseline_contract'])
    historical=read(old['historical_s7_train']['path'])
    history=set().union(*(sources(e['source_row']) for e in historical['entries']))
    heldout=set().union(*(used[s] for s in ('cal','select','test')))
    if heldout & (history|donors['train']):raise ValueError('historical TRAIN/donor leakage')
    audit_path=out/'full_audit.json'
    save_json(audit_path,dict(schema='aggressive-v17-full-audit/1',status='passed',checked_pairs=30000,
        failures=0,manifest_sha256={s:v['sha256'] for s,v in specs.items()},
        review_approval_sha256=digest(config['approval']),source_disjoint=True,
        endpoint_and_area_pixel_audit=True,target_and_donor_audit=True,split_audits=audit_shas,fallback_numerical_identity_audit=True,
        exact_numerical_duplicate_count=len(inherited_duplicates),
        unique_numerical_sample_count=30000-len(inherited_duplicates),
        inherited_v14_duplicate_occurrences=inherited_duplicates,
        new_or_cross_split_duplicate_count=0,
        duplicate_policy='retain only exact original v14 repetitions in their original split; report counts; no reweighting',
        summaries=summaries))
    families={s:sorted(used[s]) for s in COUNTS};families.update(historical_s7_train=sorted(history),donor_train=sorted(donors['train']))
    contract=dict(schema='s7-consensus-data-contract/4',status='passed',augmentation_revision=REVISION,
        train=dict(specs['train'],pairs=24000,positives=12000,negatives=12000),
        validation={'cal_mixed':specs['cal'],'select_mixed':specs['select']},test={'mixed':specs['test']},
        review_approval_sha256=digest(config['approval']),source_disjoint=True,source_families=families,
        aggressive_full_audit=dict(path=str(audit_path),sha256=digest(audit_path)),
        validation_hard_caveat='6000 augmented instances, not 6000 independent manuscripts; original heldout scale',
        online_mirror_probability=0.,offline_mirror_probability=.15,no_new_base_tears=True,
        validation_design=old['validation_design'],
        fallback_policy='same original v14 pair retained unchanged if bounded new augmentation fails; reported separately',
        inherited_v14_duplicate_count=len(inherited_duplicates),
        duplicate_policy='same-fold inherited v14 only, all numeric fields verified; unchanged row weights; not independent new examples',
        baseline_contract_sha256=digest(config['baseline_contract']),full_generation_config_sha256=digest(out/'config.json'),
        augmentation=dict(weak_peak=[3,8],major_peak=[5,15],major_upper_exclusive=True,notches=[1,4],
            primary_final_bilateral_gap_peak=[5,35],light_peak=[1,3],light_untouched_coverage=.70,
            endpoint_shortening=.20,area_loss_cap=.20,planned_smaller_cut_fraction=.70,applied_statistics=summaries))
    save_json(out/'data_contract.json',contract)
    return summaries

"""Full native-proposal source-isolated simulation evaluation. No GT seeds."""
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,Subset

from .data import Dataset,collate,to_device
from .evidence import PairEvidence
from .matcher import INPUTS
from .metrics import choose_threshold,summarize
from .scratch_matcher import matching_loss
from .validation_protocol import layout,check_rows


@torch.no_grad()
def evaluate_view(model,manifest,expected_sha,stage,device,config,cache=None):
    if stage not in ('matcher','scorer'):
        raise ValueError('stage must be matcher/scorer')
    rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    dataset=Dataset(manifest,expected_sha)
    loader=DataLoader(Subset(dataset,list(range(rank,len(dataset),world))),batch_size=config.microbatch,
        num_workers=config.workers_per_rank,collate_fn=collate,pin_memory=True)
    model.eval();rows=[]
    for batch in loader:
        batch=to_device(batch,device)
        output=model.matcher(*(batch[k] for k in INPUTS))
        matcher_loss,_,_=matching_loss(model.matcher,output,batch)
        for i,pair_id in enumerate(batch['pair_ids']):
            pair=PairEvidence.from_matcher(output,i,batch['mask_a'],batch['mask_b'])
            proposals=model.builder(pair) if cache is None else cache.get(pair_id,model.builder,pair)
            known=bool(batch['translation_valid'][i]);gt=batch['translation_a_to_b_rc'][i]
            before=[float((c.translation.to(device)-gt).norm()) for c in proposals.clusters] if known else []
            if stage=='scorer':
                pred=model.score_pair(pair,proposals=proposals)
                translation=pred.translation_a_to_b_rc
                score=float(pred.score);numeric=pred.numeric_valid;has=pred.has_candidate
                after=[float((c.translation-gt).norm()) for c in pred.clusters] if known else []
                winner=pred.selected_cluster_id;uncertainty=pred.pose_uncertainty
                candidate_scores=[float(c.readout.score) for c in pred.clusters]
            else:
                translation=proposals.clusters[0].translation.to(device) if proposals.clusters else None
                score=0.;numeric=pair.numeric_valid;has=bool(proposals.clusters);after=before
                winner=0 if has else -1;uncertainty={};candidate_scores=[]
            error=float((translation-gt).norm()) if known and translation is not None else None
            valid_match=batch['target_a'][i]>=0
            ids=valid_match.nonzero(as_tuple=False).flatten()
            gt_nll=float(-output.assignment[i,ids,batch['target_a'][i,ids]].clamp_min(1e-8).log().mean()) if len(ids) else None
            rows.append(dict(pair_id=pair_id,label=bool(batch['labels'][i]),gt_known=known,
                recipe=batch['recipes'][i],
                has_candidate=has,numeric_valid=numeric,score=score,
                translation=None if translation is None else translation.detach().cpu().tolist(),
                selected_cluster_id=winner,error_px=error,layout20=bool(error is not None and error<=20),
                candidate_coverage=bool(known and any(e<=20 for e in after)),
                proposal_coverage=bool(known and any(e<=20 for e in before)),
                proposal_errors_px=before,candidate_errors_px=after,candidate_scores=candidate_scores,
                candidate_count=len(proposals.clusters),seed_count=len(proposals.seeds),merged_count=len(proposals.merge_trace),
                valid_points_a=len(pair.local_a),valid_points_b=len(pair.local_b),
                pose_uncertainty=uncertainty,gt_match_nll=gt_nll,
                # Matching loss is averaged over this physical batch solely for
                # aggregate auxiliary diagnostics; GT matchNLL above is per-pair.
                matcher_batch_loss=float(matcher_loss),
                transport_converged=bool(output.transport.diagnostics.converged[i])))
    if world>1:
        gathered=[None]*world;dist.all_gather_object(gathered,rows);rows=[r for group in gathered for r in group]
    if len(rows)!=len(dataset) or len({r['pair_id'] for r in rows})!=len(dataset):
        raise AssertionError('duplicated/missing validation Pair')
    return sorted(rows,key=lambda r:r['pair_id'])


def summarize_validation(data,contract,stage,config):
    if stage not in ('matcher','scorer'):
        raise ValueError('stage must be matcher/scorer')
    population=check_rows(data,contract)
    kind,views=layout(contract)
    threshold=(choose_threshold(data['cal'],config,paired_views=kind=='paired_clean_hard')
               if stage=='scorer' else config.threshold_tie_preference)
    metrics={v:summarize(data['select'][v],threshold) for v in views}
    primary=float(np.mean([m['joint_f1' if stage=='scorer' else 'candidate_coverage'] for m in metrics.values()]))
    overall_layout=float(np.mean([m['layout20'] for m in metrics.values()]))
    if stage=='matcher':
        auxiliary=-float(np.mean([r['matcher_batch_loss'] for rows in data['select'].values() for r in rows]))
    else:
        auxiliary=float(np.mean([m['ap'] for m in metrics.values()]))
    result=dict(stage=stage,threshold=threshold,key=[primary,overall_layout,auxiliary],selection_value=primary,
        selected=metrics,fixed03={v:summarize(data['select'][v],.3) for v in views},real_used=False,**population)
    if kind=='paired_clean_hard':
        # Historical compatibility aliases, now calculated rather than hardcoded.
        result.update(cal_independent_pairs=population['cal_distinct_pair_ids'],
                      select_independent_pairs=population['select_distinct_pair_ids'])
    else:
        recipes=sorted({r['recipe'] for r in data['select']['mixed']})
        result['recipe_diagnostics']={recipe:summarize(
            [r for r in data['select']['mixed'] if r['recipe']==recipe],threshold) for recipe in recipes}
        result['recipe_diagnostics_affect_selection']=False
    return result


def validate(model,contract,stage,device,config,caches=None):
    start=time.time();data={}
    _,views=layout(contract)
    for split in ('cal','select'):
        data[split]={}
        for view in views:
            name=split+'_'+view;record=contract['validation'][name]
            data[split][view]=evaluate_view(model,record['path'],record['sha256'],stage,device,config,
                                           None if caches is None else caches[name])
    report=summarize_validation(data,contract,stage,config)
    report['elapsed_seconds']=time.time()-start
    return report,data

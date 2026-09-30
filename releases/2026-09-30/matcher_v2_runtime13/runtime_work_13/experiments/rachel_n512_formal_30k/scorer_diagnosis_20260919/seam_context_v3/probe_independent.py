"""Post-completion layer probes of the independently selected F/I checkpoints.

Matcher and Scorer feature paths are exported separately. No epoch selection,
threshold fitting or feature updates occur here. The frozen external evaluation
must have completed first so each probe is checked against its exact winner.
"""
import argparse
import json
from pathlib import Path
from .evaluate_independent import load_winner, setup
from .prepare import read, save, sha
from .probe_external import run as probe


def validate_reference(cases, reference, arm, checkpoint_sha):
    keys=[(r['split'],r['pair_id']) for r in cases]
    if not keys or len(set(keys))!=len(keys):
        raise ValueError('probe requests must be nonempty and unique')
    for split in sorted({k[0] for k in keys}):
        if split not in ('dunhuang_cv','turufan'):
            raise ValueError('probe split must match the independent real evaluation')
        path=Path(reference)/split
        protocol=read(path/'protocol.json')
        if (protocol['status'],protocol['arm'],protocol['checkpoint_sha256'])!=('complete',arm,checkpoint_sha):
            raise ValueError('reference evaluation belongs to another checkpoint or is incomplete')
        rows=[json.loads(line) for line in (path/'case_diagnostics.jsonl').read_text().splitlines()]
        ids=[r['pair_id'] for r in rows]
        if len(ids)!=len(set(ids)) or not {k[1] for k in keys if k[0]==split}<=set(ids):
            raise ValueError('requested cases absent from the reference population')
    return set(keys)


def run(a):
    setup(260923)
    model,cp,selection=load_winner(a.run,a.base)
    a.checkpoint=str(Path(a.run)/'best.pt')
    checkpoint_sha=sha(a.checkpoint)
    requested=validate_reference(read(a.cases)['cases'],a.reference,cp['arm'],checkpoint_sha)
    provenance=dict(arm=cp['arm'],selected_epoch=selection['epoch'],
        checkpoint_sha256=checkpoint_sha,base_checkpoint_sha256=cp['binding']['base_sha256'],
        scorer_context_is_independent=cp['arm']=='independent_features',
        matcher_frozen=True,threshold_fitting=False,checkpoint_selected_on_real=False,
        script_sha256=sha(__file__))
    probe(a,supplied_model=model,supplied_config=model.cfg,provenance=provenance)
    records=read(Path(a.out)/'records.json')
    actual={(r['split'],r['pair_id']) for r in records}
    parity_ok=all(r['parity']['score_delta']<=2e-5 and
        (r['parity']['translation_delta_px'] is None or r['parity']['translation_delta_px']<=.01) for r in records)
    if actual!=requested or len(records)!=len(requested) or not parity_ok:
        save(Path(a.out)/'status.json',dict(status='failed_parity_or_coverage',requested=len(requested),actual=len(records),
            actual_ids_match=actual==requested,parity_ok=parity_ok,model_provenance=provenance))
        raise ValueError('probe failed full-population input/output parity or case coverage')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','base','reference','cases','out'):parser.add_argument('--'+name,required=True)
    run(parser.parse_args())

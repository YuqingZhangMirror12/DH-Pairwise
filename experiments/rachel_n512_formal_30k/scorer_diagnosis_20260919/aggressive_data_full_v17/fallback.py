"""Independent identity check: retained v14 is never relabelled as a new cut."""
from dataclasses import fields
import hashlib
import numpy as np
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
from ..s7_compound_v1.materialize import digest
from ..s7_consensus_v1.check_data_compatibility import check_batch
from ..s7_consensus_v1.data import collate
from ..seam_context_v3.targets import known_gap_links
from .source import baseline

def numerical_identity(actual,original):
    for f in fields(actual):
        a,b=getattr(actual,f.name),getattr(original,f.name)
        if isinstance(a,np.ndarray):
            if not np.array_equal(a,b):raise ValueError('retained v14 array changed:'+f.name)
        elif f.name not in ('pair_id','fragment_a_token','fragment_b_token') and a!=b:
            raise ValueError('retained v14 scalar changed:'+f.name)

def audit(record):
    if record['detail']['trim'] is not None or record['requested_gap_count']!=0:
        raise ValueError('fallback must not claim v17 cut/notches')
    sample,report=load_sample(record['sample_path']);original,prior=load_sample(record['baseline_sample_path'])
    numerical_identity(sample,original)
    if report['compound']!=prior['compound'] or report['pair_shared_scale']!=prior['pair_shared_scale']:
        raise ValueError('retained v14 recipe/scale changed')
    if not report.get('v14_fallback') or digest(record['sample_path'])!=record['sample_sha256']:
        raise ValueError('fallback archive binding')
    if digest(record['proof_path'])!=record['proof_sha256']:raise ValueError('fallback proof binding')
    with np.load(record['proof_path'],allow_pickle=False) as z:
        for side in 'ab':
            if not np.array_equal(np.unpackbits(z['packed_final_'+side],axis=1),getattr(sample,'mask_'+side)[0]):
                raise ValueError('retained mask proof differs')
    source=baseline(record['baseline_slot'],record['baseline_ordinal'])
    expected=known_gap_links(sample,source['original'],prior)
    if digest(record['target_metadata'])!=record['target_metadata_sha256']:raise ValueError('fallback target binding')
    with np.load(record['target_metadata'],allow_pickle=False) as z:
        if set(z.files)!=set(expected) or any(not np.array_equal(z[k],expected[k]) for k in expected):
            raise ValueError('retained supervision changed')
    items=[(sample,report,record)];supervision=check_batch(items,collate(items))[0]
    h=hashlib.sha256()
    for name in ('mask_a','mask_b','points_rc_a','points_rc_b','target_a','target_b','translation_a_to_b_rc'):
        h.update(name.encode());h.update(getattr(sample,name).tobytes())
    return dict(id=record['id'],status='passed',v14_fallback=True,baseline_numerical_identity=True,
        no_v17_cut_claimed=True,supervision=supervision,light_coverages=None,paired_gap=None,
        sample_sha256=record['sample_sha256'],proof_sha256=record['proof_sha256'],model_input_sha256=h.hexdigest())

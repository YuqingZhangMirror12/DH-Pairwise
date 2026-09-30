"""Version-aware export of already-computed evidence; no network/source edits.

The bound model's generic exporter predates exact-union scoring. Extend only
its detached output, preserving actual tensors and giving K an honest meaning.
"""
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1 import diagnostics

from .frozen import TrainingConfig, registered_protocol
from .attention_trace import attach_trace


def snapshot_prediction(pair_id, pair, prediction, *, threshold, provenance, attention_trace=None):
    protocol = registered_protocol(TrainingConfig().record())
    for key in ('variant', 'evidence_mode'):
        if key in provenance and provenance[key] != protocol[key]:
            raise ValueError('snapshot provenance differs from bound implementation: ' + key)
    meta, values = diagnostics.snapshot_prediction(pair_id, pair, prediction,
        threshold=threshold, provenance=provenance)
    arrays = diagnostics._Arrays()
    arrays.values, arrays.manifest = values, meta['arrays']
    meta['export_adapter'] = 's7-consensus-version-aware-snapshot/1'
    meta['semantics']['evidence_mode'] = protocol['evidence_mode']
    meta['semantics']['variant'] = protocol['variant']
    union_mode = protocol['evidence_mode'] == 'exact_union_q'
    for saved, cluster in zip(meta['clusters'], prediction.clusters):
        for name, encoded in (('initial', cluster.initial_encoded), ('final', cluster.encoded)):
            evidence = encoded.evidence
            is_union = hasattr(evidence, 'union_edge_ids') and hasattr(evidence, 'directional_kernels')
            if is_union != union_mode:
                raise ValueError('actual evidence type differs from registered weighting')
            if is_union:
                stage = saved['stages'][name]
                prefix = f"cluster_{saved['cluster_id']:03d}/{name}"
                for field in ('union_edge_ids', 'directional_kernels'):
                    stage[field] = arrays.add(prefix+'/'+field, getattr(evidence, field))
                stage['directional_q_diagnostic'] = arrays.add(prefix+'/directional_q_diagnostic',
                    pair.q*evidence.directional_kernels)
        if union_mode:
            saved['joint_constraint'] = diagnostics._json_value(cluster.joint_constraint)
    if union_mode:
        meta['semantics'].update(
            kernels='binary exact deduplicated union membership, NOT a directional kernel or attention',
            weights='absolute Q times union membership; no geometric attenuation or cluster normalization',
            directional_kernels='recorded pose-dependent damage compatibility; NOT used to attenuate Scorer Q',
            directional_q_diagnostic='Q times recorded directional kernel for comparison ONLY; not the actual Scorer input',
            localization_kernels='directional precision kernel used by the actual INITIAL localization fit',
            correspondence_ids='all numerically nonzero FINAL Q*union entries, no display cutoff',
            edge_contributions='absolute Q*union times mean A/B observed-arc local-class support',
            joint_constraint='actual bound common-pose refinement guard; original pose diameter remains16px')
    if attention_trace is not None:
        attach_trace(meta, arrays, prediction, attention_trace)
    return meta, arrays.values

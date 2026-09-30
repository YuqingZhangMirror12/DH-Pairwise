"""Readback audit of saved C10 numerical evidence, without another model call.

This verifies serialization and attribution semantics, not layout correctness.
No GT, learned thresholds, figure-dependent selections or model writes occur.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .frozen import read, sha


def close(actual, expected, name, *, rtol=2e-5, atol=2e-6):
    if not np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False):
        raise ValueError('evidence consistency failed: '+name)


def audit_union(cluster, arrays, q):
    """Verify exact union, fixed16 diameter, and pose guard without new fitting."""
    get = lambda key: arrays[key]
    proposal = cluster['proposal']
    ids = get(proposal['edge_ids'])
    if not np.array_equal(ids, np.unique(ids, axis=0)):
        raise ValueError('union point pairs were not deduplicated')
    if not np.array_equal(ids, get(proposal['original_union_edge_ids'])):
        raise ValueError('original union differs from scored cluster')
    mask = np.zeros_like(q)
    mask[ids[:, 0], ids[:, 1]] = 1
    centers = get(proposal['member_translations_rc'])
    actual_diameter = float(np.linalg.norm(centers[:, None, :]-centers[None, :, :], axis=2).max())
    if proposal['pose_diameter_px'] != 16. or actual_diameter > 16.+1e-5:
        raise ValueError('fixed16 complete-link diameter violated')
    close(actual_diameter, proposal['actual_diameter_px'], 'original candidate diameter')
    for stage in cluster['stages'].values():
        close(get(stage['kernels']), mask, 'exact union membership', atol=0, rtol=0)
        if not np.array_equal(get(stage['union_edge_ids']), ids):
            raise ValueError('scoring union identity changed between passes')
        directional = get(stage['directional_kernels'])
        if directional.shape != q.shape or not np.isfinite(directional).all() or (
                (directional < 0) | (directional > 1)).any():
            raise ValueError('invalid recorded directional kernel')
        close(get(stage['directional_q_diagnostic']), q*directional,
              'diagnostic directional Q', atol=1e-37)
        if np.linalg.norm(centers-get(stage['pose']), axis=1).max() > 16.+1e-4:
            raise ValueError('final/initial pose outside common-pose bound')
    constraint = cluster.get('joint_constraint', {})
    if (constraint.get('threshold_px') != 16.
            or constraint.get('original_pose_diameter_px') != proposal['actual_diameter_px']):
        raise ValueError('recorded common-pose constraint differs')
    if constraint.get('physical_refinement_blocked'):
        close(get(cluster['stages']['initial']['pose']), get(cluster['stages']['final']['pose']),
              'blocked physical refinement', atol=0, rtol=0)


def validate_attention(meta, arrays):
    """Readback identity/normalization audit; not a causal or GT assertion."""
    capture=meta.get('attention_capture')
    present=any(s.get('attention') for c in meta['clusters'] for s in c['stages'].values())
    exported=meta['semantics'].get('attention_weights_exported',False)
    if capture is None:
        if present or exported:
            raise ValueError('attention arrays lack an executed-capture protocol')
        return
    if (capture.get('schema')!='s7-consensus-executed-attention/1'
            or capture.get('heads')!=4 or capture.get('no_additional_network_forward') is not True
            or capture.get('attention_calls')!=(16 if meta['clusters'] else 0)
            or exported!=bool(meta['clusters'])):
        raise ValueError('attention capture protocol differs')
    expected_keys={f'{layer}/{kind}/{side}' for layer in (0,1)
                   for kind in ('self_attention','cross_attention') for side in 'ab'}
    for c in meta['clusters']:
        for name,stage in c['stages'].items():
            saved=stage.get('attention',{})
            if set(saved)!=expected_keys:
                raise ValueError('attention stage missing an operator')
            for label,item in saved.items():
                layer,kind,side=label.split('/')
                key_side=side if kind=='self_attention' else ('b' if side=='a' else 'a')
                ordinal=(0 if name=='initial' else 2)+(0 if side=='a' else 1)
                if (item.get('module')!=f'layers.{layer}.{kind}' or item.get('layer')!=int(layer)
                        or item.get('heads')!=4 or item.get('kind')!=kind
                        or item.get('query_side')!=side or item.get('key_side')!=key_side
                        or item.get('call_ordinal')!=ordinal):
                    raise ValueError('attention phase/side/operator identity differs')
                qinput,kinput=stage[side]['input'],stage[key_side]['input']
                qi=np.flatnonzero(arrays[qinput['valid']]);ki=np.flatnonzero(arrays[kinput['valid']])
                if (not np.array_equal(arrays[item['query_compact_ids']],qi)
                        or not np.array_equal(arrays[item['key_compact_ids']],ki)):
                    raise ValueError('attention contour identity differs')
                weights=arrays[item['head_mean']]
                if (weights.shape!=(len(qi),len(ki)) or not np.isfinite(weights).all()
                        or (weights<0).any() or (weights>1.+1e-6).any()):
                    raise ValueError('invalid attention head-mean matrix')
                close(weights.sum(-1),np.full(len(qi),float(len(ki)>0)),'attention row normalization')
                close(arrays[item['key_measure']],
                    (arrays[kinput['mass']]*arrays[kinput['observed_arc_px']])[ki],
                    'actual attention key observation measure',rtol=0.,atol=0.)
                if kind=='self_attention':
                    if item['geometry_bias'] is not None:
                        raise ValueError('self-attention has unexpected geometric bias')
                else:
                    kernels=arrays[stage['kernels']]
                    if side=='b':kernels=kernels.T
                    close(arrays[item['geometry_bias']],np.log(np.maximum(kernels[np.ix_(qi,ki)],1e-30)),
                          'actual cross-attention bias')
                error=item.get('reconstruction_max_abs',float('nan'))
                if not np.isfinite(error) or not 0.<=error<=2e-6:
                    raise ValueError('attention reconstruction check failed')


def validate_arrays(meta, arrays):
    if meta.get('schema') != 's7-consensus-evidence/1' or set(arrays) != set(meta['arrays']):
        raise ValueError('evidence schema or array membership differs')
    for key, spec in meta['arrays'].items():
        value = arrays[key]
        if (list(value.shape) != spec['shape'] or value.dtype.str != spec['dtype']
                or hashlib.sha256(value.tobytes(order='C')).hexdigest() != spec['sha256']
                or int((~np.isfinite(value)).sum()) != spec['nonfinite_count']):
            raise ValueError('evidence array content differs: '+key)
    validate_attention(meta, arrays)
    get = lambda key: arrays[key]
    p = meta['pair']; q = get(p['q']); n,m = q.shape
    if n != len(get(p['local_a'])) or m != len(get(p['local_b'])):
        raise ValueError('compact Q/feature size mismatch')
    if bool(meta['clusters']) != meta['has_candidate']:
        raise ValueError('candidate presence differs')
    accepted = bool(meta['has_candidate'] and meta['numeric_valid']
                    and meta['score'] >= meta['threshold'])
    if accepted != meta['accepted']:
        raise ValueError('acceptance is not the recorded frozen threshold')
    if not meta['has_candidate']:
        if meta['selected_cluster_id'] != -1 or meta['translation_a_to_b_rc'] is not None:
            raise ValueError('empty evidence invents a pose')
        return dict(status='passed', pair_id=meta['pair_id'], clusters=[], has_candidate=False,
                    layout_correctness_verified=False)
    if not meta['numeric_valid'] or not np.isfinite(q).all():
        raise ValueError('nonfinite candidate cannot support a numerical diagnosis')
    selected = [c for c in meta['clusters'] if c['selected']]
    if len(selected) != 1 or selected[0]['cluster_id'] != meta['selected_cluster_id']:
        raise ValueError('selected cluster identity differs')
    winner = selected[0]
    close(get(winner['refinement']['translation']), meta['translation_a_to_b_rc'], 'selected pose',atol=0,rtol=0)
    close(-get(winner['refinement']['translation']), meta['canvas_b_shift_rc'], 'canvas sign',atol=0,rtol=0)
    close(get(winner['readout']['score']),meta['score'],'selected score',atol=0,rtol=0)
    reports = []
    mode = meta.get('semantics', {}).get('evidence_mode', 'directional_full_q')
    if mode not in ('directional_full_q', 'exact_union_q'):
        raise ValueError('unknown evidence weighting mode')
    if ('evidence_mode' in meta.get('provenance', {})
            and meta['provenance']['evidence_mode'] != mode):
        raise ValueError('provenance and evidence meaning differ')
    for c in meta['clusters']:
        if ('union_edge_ids' in c['stages']['final']) != (mode == 'exact_union_q'):
            raise ValueError('actual recorded evidence type differs from its meaning')
        if mode == 'exact_union_q':
            audit_union(c, arrays, q)
        initial,final = c['stages']['initial'],c['stages']['final']
        close(get(initial['pose']),get(c['proposal']['translation']),'initial proposal pose',atol=0,rtol=0)
        close(get(final['pose']),get(c['refinement']['translation']),'final re-encoded pose',atol=0,rtol=0)
        for stage in (initial,final):
            w,k = get(stage['weights']),get(stage['kernels'])
            if w.shape != q.shape or k.shape != q.shape:
                raise ValueError('full Q was truncated')
            close(w,q*k,'absolute Q times registered membership/kernel',atol=1e-37)
            for side,axis in (('a',1),('b',0)):
                inp,out = stage[side]['input'],stage[side]['output']
                mass=get(inp['mass']); observed=get(inp['observed_arc_px'])
                close(mass,w.sum(axis),'endpoint mass')
                close(get(inp['unmatched']),get(p['unmatched_'+side]),'dustbin preserved')
                close(mass+get(inp['other_mass']),q.sum(axis),'recalled versus other-pose mass')
                valid=get(inp['valid']).astype(bool)
                if not np.array_equal(valid,mass>0):
                    raise ValueError('nonzero evidence was silently thresholded')
                probability=get(out['local_probabilities'])
                close(probability.sum(1),valid.astype(float),'support/unknown/conflict partition')
                if (probability<0).any() or (observed<0).any():
                    raise ValueError('negative class mass or observation measure')
        w=get(final['weights'])
        ids=get(c['correspondence_ids'])
        if not np.array_equal(ids,np.argwhere(w>0)):
            raise ValueError('final sparse export omits nonzero correspondences')
        original=np.stack([get(p['original_a'])[ids[:,0]],get(p['original_b'])[ids[:,1]]],axis=1)
        if not np.array_equal(original,get(c['correspondence_original_ids'])):
            raise ValueError('original endpoint mapping differs')
        old={tuple(x) for x in get(c['proposal']['edge_ids'])}
        added=np.array([tuple(x) not in old for x in ids],bool)
        if (not np.array_equal(added,get(c['added_to_sparse_proposal']))
                or int(added.sum()) != c['added_correspondence_count']):
            raise ValueError('new recall attribution differs')
        side_support=[]
        for side in 'ab':
            inp,out=final[side]['input'],final[side]['output']
            measure=get(inp['mass'])*get(inp['observed_arc_px'])
            prob=get(out['local_probabilities'])
            support=measure*prob[:,0]
            close(support,get(c['readout']['support_weights_'+side]),'final score support')
            side_support.append(support.sum())
        for col,name in enumerate(('support','unknown','conflict')):
            a,b=final['a'],final['b']
            ca=get(a['input']['observed_arc_px'])*get(a['output']['local_probabilities'])[:,col]
            cb=get(b['input']['observed_arc_px'])*get(b['output']['local_probabilities'])[:,col]
            expected=.5*w*(ca[:,None]+cb[None])
            close(get(c['edge_contributions'][name]),expected,name+' edge contribution',atol=1e-36)
        support_total=float(get(c['edge_contributions']['support']).sum())
        close(support_total,get(c['readout']['positive_evidence_px']),'support readout total')
        close(.5*sum(side_support),support_total,'no A/B double counting')
        close(get(c['edge_contributions']['conflict']).sum(),
              get(c['readout']['conflict_evidence_px']),'conflict readout total')
        # These are the weights ACTUALLY used before refinement; do not replace
        # them by final-pass reliability or by a normalized picture heatmap.
        a,b=initial['a'],initial['b']
        measure=.5*(get(a['input']['observed_arc_px'])[:,None]+get(b['input']['observed_arc_px'])[None])
        support=.5*(get(a['output']['local_probabilities'])[:,0,None]+get(b['output']['local_probabilities'])[None,:,0])
        reliable=.5*(get(a['output']['localization_reliability'])[:,None]+get(b['output']['localization_reliability'])[None])
        base=get(initial['weights'])*measure*support
        anchor=reliable*get(initial['localization_kernels'])
        close(get(c['refinement']['localization_weights']),base*anchor,'initial localization weights',atol=1e-36)
        close(get(c['refinement']['compatibility_weights']),base*(1-anchor),'initial compatibility weights',atol=1e-36)
        recall=get(c['added_to_sparse_proposal'])
        contribution=get(c['edge_contributions']['support'])[ids[:,0],ids[:,1]]
        report = dict(cluster_id=c['cluster_id'],selected=c['selected'], evidence_mode=mode,
            retained_correspondences=len(ids),added_correspondences=int(recall.sum()),
            support_total_px=support_total,added_support_contribution_px=float(contribution[recall].sum()),
            evidence_at_final_pose=True,localization_weights_from_initial_pass=True)
        if mode == 'exact_union_q':
            report['union_absolute_q_mass'] = float(w.sum())
            report['directionally_weighted_union_q_mass_diagnostic'] = float(
                (w*get(final['directional_kernels'])).sum())
            report['directional_mass_used_for_scorer'] = False
        reports.append(report)
    return dict(status='passed',pair_id=meta['pair_id'],clusters=reports,has_candidate=True,
                layout_correctness_verified=False,
                note='Checks exact numerical provenance and accounting, not causal benefit or true seam length.')


def audit_snapshot(path):
    path=Path(path);meta=read(path)
    sidecar=path.parent/meta['sidecar']['path']
    if (sidecar.parent.resolve()!=path.parent.resolve() or sha(sidecar)!=meta['sidecar']['sha256']
            or sidecar.stat().st_size!=meta['sidecar']['bytes']):
        raise ValueError('snapshot sidecar identity differs')
    with np.load(sidecar,allow_pickle=False) as values:
        arrays={k:values[k] for k in values.files}
    return validate_arrays(meta,arrays)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot')
    args=parser.parse_args()
    print(json.dumps(audit_snapshot(args.snapshot),ensure_ascii=False,indent=2,allow_nan=False))

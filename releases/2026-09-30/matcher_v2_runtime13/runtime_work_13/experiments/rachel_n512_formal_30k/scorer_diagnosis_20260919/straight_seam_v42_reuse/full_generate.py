"""Approved full-size v4.2 geometry, exact quotas, corrected mismatch targets.

Reference files and older runs stay read-only. No model inference or training.
Pixel generation is unchanged; healthy-target admission may reject an attempt.
"""
import argparse
from collections import Counter
import json
import multiprocessing
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy import ndimage

if __package__:
    from . import generate as g, supervision as s
else:
    import generate as g
    import supervision as s


COUNTS = {'train': {'M': 600, 'J': 1200, 'R': 1200},
          'select': {'M': 100, 'J': 200, 'R': 150},
          'test': {'M': 100, 'J': 200, 'R': 150}}
SEEDS = {'train': 26093084, 'select': 26093085, 'test': 26093086}
REVISION = 'codex-v42-geometry-explicit-targets/1'


def j_bases(count, seed, split, label, split_seed):
    if count % 25:
        raise ValueError('J count must allow exact 80/12/8 percent quotas')
    bases = np.array(['torn_rachel']*(count*20//25) +
                     ['margin_fragment']*(count*3//25) + ['torn_strip']*(count*2//25))
    np.random.default_rng([seed, split_seed, 7718, int(label)]).shuffle(bases)
    return bases.tolist()


def interval_damage(along, intervals, margin_px=3.):
    """Same closed-interval convention as the audited target helper."""
    result = np.zeros(np.asarray(along).shape, dtype=bool)
    for start, stop in intervals:
        if not np.isfinite([start, stop]).all() or stop < start:
            raise ValueError('invalid damage interval')
        result |= (along >= start-margin_px) & (along <= stop+margin_px)
    return result


class ParameterRNG:
    """Record first three reference shape draws; otherwise delegate unchanged."""
    def __init__(self, rng):
        self.rng = rng
        self.shape_draws = []

    def uniform(self, *args, **kwargs):
        value = self.rng.uniform(*args, **kwargs)
        if len(self.shape_draws) < 3:
            self.shape_draws.append(float(value))
        return value

    def __getattr__(self, name):
        return getattr(self.rng, name)


def initialize(reference_dir, rows, counts):
    g.initialize(reference_dir, rows)
    ref = g.CONTEXT['reference']
    base_of, piece, misfit, finalize, save_sample = (
        ref.base_of, ref.piece_pair, ref.misfit, ref.finalize, ref.save_sample)
    quota_cache = {}

    def quota_base(kind, seed, split, label, idx):
        if kind != 'J':
            return base_of(kind, seed, split, label, idx)
        key = seed, split, label
        if key not in quota_cache:
            quota_cache[key] = j_bases(counts['J'], seed, split, label, ref.SPLIT_SEED[split])
        return quota_cache[key][idx]

    def record_misfit(a, b, geom, rng, kind):
        seam = a & ndimage.binary_dilation(b)
        recorder = s.RecordingRNG(rng)
        result = misfit(a, b, geom, recorder, kind)
        if seam.sum() >= 50:
            along = (np.column_stack(np.nonzero(seam))-geom[0]) @ geom[1]
            trace = s.parse_events(recorder.events, ref.MIS[kind], float(along.min()), float(along.max()))
        else:
            assert not recorder.events
            trace = dict(intervals=[], wear=[0., 0.], gap_coverage_draw=None,
                         overlap_active=False, overlap_coverage_draw=None)
        g.CONTEXT['trace'] = trace
        return result

    def record_piece(kind, base, rng, srcs):
        proxy = ParameterRNG(rng)
        g.CONTEXT['trace'] = None
        a, b, meta = piece(kind, base, proxy, srcs)
        if len(proxy.shape_draws) != 3:
            raise ValueError('reference shape draw protocol changed')
        record = dict(base=base, shape=dict(zip(
            ('rough_std_px', 'rough_corr_px', 'bend_amp_px'), proxy.shape_draws)),
            piece_rejection=str(meta) if a is None else None,
            misfit_trace=g.CONTEXT['trace'])
        g.CONTEXT['piece_attempts'].append(record)
        return a, b, meta

    def corrected_finalize(a, b, meta, label=True):
        result, reason = finalize(a, b, meta, label)
        if result is None:
            g.CONTEXT['finalization_attempts'].append(dict(reason=reason))
            return result, reason
        if not label:
            g.CONTEXT['finalization_attempts'].append(dict(reason=None))
            return result, None
        ia, ib, t, pa, va, pb, vb, ta, tb = result
        proof = g.CONTEXT['proof']
        sample = SimpleNamespace(points_rc_a=pa, points_rc_b=pb,
                                 contour_valid_a=va, contour_valid_b=vb,
                                 target_a=ta, target_b=tb)
        targets, report = s.build_supervision(sample, proof, g.CONTEXT['trace'], interval_damage)
        healthy = report['correspondence_count']
        reason = None if healthy >= 8 else 'few_healthy_corr'
        g.CONTEXT['finalization_attempts'].append(dict(reason=reason,
            original_correspondences=report['original_correspondences'], healthy_correspondences=healthy))
        if reason:
            g.CONTEXT['rejections'][reason] += 1
            return None, reason
        proof['original_target_a'] = ta.copy()
        proof['original_target_b'] = tb.copy()
        meta['reference_correspondences_before_damage_exclusion'] = meta['correspondences']
        meta['correspondences'] = healthy
        meta['supervision_revision'] = REVISION
        g.CONTEXT['target_audit'] = report
        return (ia, ib, t, pa, va, pb, vb, targets['target_a'], targets['target_b']), None

    def corrected_save(*args, **kwargs):
        args = list(args)
        report = dict(args[8])
        report['supervision_revision'] = REVISION
        report['inheritance_rule'] = (
            'Filter reference reciprocal targets by the intended common seam; '
            'ignore (-2) both sides of any gap/overlap/nick interval including 3px shoulders; '
            'other unmatched points -1. R uniform wear retains its adaptive tolerance. '
            'GT, intervals and provenance are supervision only, never model inputs.')
        args[8] = report
        return save_sample(*args, **kwargs)

    ref.base_of = quota_base
    ref.misfit = record_misfit
    ref.piece_pair = record_piece
    ref.finalize = corrected_finalize
    ref.save_sample = corrected_save


def generate_one(job):
    g.CONTEXT.update(piece_attempts=[], finalization_attempts=[], target_audit=None, trace=None)
    entry = g.generate_one(job)
    entry.update(wrapper_revision=REVISION,
                 piece_attempts=g.CONTEXT['piece_attempts'],
                 finalization_attempts=g.CONTEXT['finalization_attempts'],
                 target_audit=g.CONTEXT['target_audit'] if entry.get('label') else None,
                 accepted_damage_trace=g.CONTEXT['trace'] if entry.get('label') else None)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-dir', 'source-admission', 'real-audit', 'output-new'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--split', choices=list(COUNTS), required=True)
    parser.add_argument('--workers', type=int, choices=[1, 2, 3, 4], default=3)
    parser.add_argument('--preflight', action='store_true', help='70 TRAIN-source pairs; distinct seed; not part of the full catalog')
    a = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('explicit CPU-only environment required')
    ref, hashes = g.load_reference(a.reference_dir)
    admission = json.loads(a.source_admission.read_text())
    real = json.loads(a.real_audit.read_text())
    assert real['source_admission_sha256'] == g.sha(a.source_admission)
    root, rows, excluded = g.source_pool(admission, real, a.split)
    assert str(root) == ref.SRC
    counts, seed = COUNTS[a.split], SEEDS[a.split]
    if a.preflight:
        if a.split != 'train':
            raise ValueError('preflight is TRAIN-only')
        counts, seed = {'M':5,'J':25,'R':5}, 26093087
    out = a.output_new.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'samples').mkdir(); (out/'proof').mkdir()
    jobs = [(kind, label, i, a.split, str(out), seed)
            for kind, count in counts.items() for label in (1, 0) for i in range(count)]
    g.save(out/'plan.json', dict(schema=REVISION, split=a.split, seed=seed, counts_per_label=counts,
        expected=len(jobs), source_admission_sha256=g.sha(a.source_admission),
        real_audit_sha256=g.sha(a.real_audit), reference_sha256=hashes,
        code_sha256={Path(p).name:g.sha(p) for p in (__file__, g.__file__, s.__file__)},
        source_count=len(rows), excluded_donors=excluded, SEAM=ref.SEAM, MIS=ref.MIS,
        J_BASES=ref.J_BASES, quota_policy='exact 80/12/8 for J within each label; shuffle before generation',
        acceptance='reference pixel constraints plus >=8 healthy reciprocal targets after explicit damage ignore',
        retries='At most 300 per pair; same base; reference redraws shape/damage and donor. All attempts logged.',
        preflight=a.preflight, no_real_masks_for_generation=True, training_admitted=False, gpu=False))
    entries=[]; started=time.time()
    with (out/'records.jsonl').open('x') as stream:
        with multiprocessing.get_context('spawn').Pool(a.workers, initializer=initialize,
                initargs=(a.reference_dir, rows, counts)) as pool:
            for entry in pool.imap_unordered(generate_one, jobs, chunksize=1):
                entries.append(entry)
                stream.write(json.dumps(entry, ensure_ascii=False, sort_keys=True)+'\n'); stream.flush()
                if len(entries)%100 == 0 or len(entries)==len(jobs):
                    print(json.dumps(dict(done=len(entries), expected=len(jobs), seconds=time.time()-started)), flush=True)
    good=sorted([x for x in entries if not x.get('failed')],key=lambda x:x['id'])
    failed=[x for x in entries if x.get('failed')]
    g.save(out/'manifest.json', dict(schema=REVISION, split=a.split, seed=seed, entries=good,
        failed=failed, plan_sha256=g.sha(out/'plan.json'), training_admitted=False))
    receipt=dict(status='generated' if not failed else 'incomplete', generated=len(good),
        expected=len(jobs), failed=len(failed), manifest_sha256=g.sha(out/'manifest.json'),
        records_sha256=g.sha(out/'records.jsonl'), seconds=time.time()-started,
        training_admitted=False, gpu_training_started=False)
    g.save(out/'generation_complete.json', receipt)
    print(json.dumps(receipt),flush=True)
    if failed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()

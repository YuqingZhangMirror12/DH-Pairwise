"""Full SELECT900 calibration on the original frozen E32/Stats source.

Run directly, not as a package: importing a generator or a different experiments
tree before the original model would defeat the model-source identity guard.
GT is attached only after predictions. No score controls sample selection.
"""
import argparse
from collections import Counter
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback


EXPECTED = {(kind, label): n for kind, n in (('M', 100), ('J', 200), ('R', 150))
            for label in (True, False)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def validate_population(manifest, audit, manifest_sha):
    entries = manifest['entries']
    counts = Counter((r['recipe'].removeprefix('straight_'), r['label']) for r in entries)
    if (manifest.get('split') != 'select' or counts != EXPECTED or len(entries) != 900
            or manifest.get('failed') or any(type(r['label']) is not bool for r in entries)
            or any(r['recipe'] not in ('straight_M', 'straight_J', 'straight_R') for r in entries)
            or audit.get('status') != 'passed_integrity_and_supervision'
            or audit.get('rows') != 900 or len(audit.get('records', [])) != 900
            or audit.get('source_manifest_sha256') != manifest_sha):
        raise ValueError('Complete independently audited v4.2 SELECT900 required; TEST is forbidden')
    lookup = {r['pair_id']: r for r in audit['records']}
    if len(lookup) != 900 or len({r['pair_id'] for r in entries}) != 900:
        raise ValueError('Duplicate audited/manifest identities')
    for entry in entries:
        row = lookup.get(entry['pair_id'], {})
        if (row.get('sample_sha256') != entry['sample_sha256']
                or row.get('positive') != entry['label']
                or row.get('kind') != entry['recipe'][-1]
                or row.get('base') != entry['meta']['base']
                or (entry['label'] and row['target_audit']['correspondence_count'] < 8)):
            raise ValueError('Manifest row differs from independent pixel/target audit')
    return entries


def load_helper(path, expected_sha):
    if sha(path) != expected_sha:
        raise ValueError('Original frozen-model diagnostic helper changed')
    spec = importlib.util.spec_from_file_location('v42_original_model_diagnostic', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def label_predictions(entries, predictions, targets):
    import numpy as np
    if [r['pair_id'] for r in predictions] != [r['pair_id'] for r in entries]:
        raise ValueError('Missing, reordered or duplicate predictions')
    result = []
    for entry, row in zip(entries, predictions):
        gt = targets[row['pair_id']]
        errors = ([float(np.linalg.norm(np.asarray(c['translation']) - gt)) for c in row['candidates']]
                  if gt is not None else None)
        if row['winner'] != -1 and not 0 <= row['winner'] < len(row['candidates']):
            raise ValueError('Selected candidate outside prediction list')
        result.append(dict(**row, recipe=entry['recipe'][-1], base_kind=entry['meta']['base'],
            label=entry['label'], errors=errors,
            layout20=(row['winner'] >= 0 and errors[row['winner']] <= 20) if errors is not None else None,
            coverage=any(e <= 20 for e in errors) if errors is not None else None))
    return result


def run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not 1 <= args.threads <= 4:
        raise ValueError('Explicitly disable CUDA and bound CPU threads')
    manifest_path = args.data_root.resolve() / 'manifest.json'
    manifest_sha = sha(manifest_path)
    audit_sha = sha(args.audit)
    entries = validate_population(read(manifest_path), read(args.audit), manifest_sha)
    out = args.output_new.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    save(out / 'launch.json', dict(pid=os.getpid(), pgid=os.getpgrp(),
        start_ticks=Path('/proc/self/stat').read_text().split()[21], started_unix=time.time(),
        script_sha256=sha(__file__), helper_sha256=args.helper_sha256,
        expected=900, split='select', gpu_used=False, automatic_retries=0,
        data_manifest_sha256=manifest_sha, audit_sha256=audit_sha))
    try:
        helper = load_helper(args.helper_code.resolve(), args.helper_sha256)
        import torch
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        model, provenance = helper.load_model(args.model_root.resolve(), args.reference.resolve())
        data = importlib.import_module(helper.PACKAGE + '.s7_consensus_v1.data')
        matcher = importlib.import_module(helper.PACKAGE + '.s7_consensus_v1.matcher')
        evidence = importlib.import_module(helper.PACKAGE + '.s7_consensus_v1.evidence')
        before = helper.state_sha(model)
        protocol = dict(schema='v42-select-frozen-diagnostic/1', model=provenance,
            manifest_sha256=manifest_sha, audit_sha256=audit_sha, script_sha256=sha(__file__),
            helper_sha256=args.helper_sha256, split='select', expected=900, cpu_threads=args.threads,
            threshold_refitted=False, model_score_used_for_data_acceptance=False,
            gt_used_for_prediction=False, training_started=False, test_inferred=False)
        save(out / 'protocol.json', protocol)
        predictions, targets = [], {}
        with (out / 'predictions.jsonl').open('x') as stream, torch.no_grad():
            for start in range(0, len(entries), 4):
                chunk = entries[start:start + 4]
                items = []
                for entry in chunk:
                    if sha(entry['sample_path']) != entry['sample_sha256']:
                        raise ValueError('Audited sample changed')
                    sample, report = data.load_sample(entry['sample_path'])
                    if (sample.pair_id != entry['pair_id'] or bool(sample.label) != entry['label']
                            or bool(report['pose_supervision_enabled']) != (bool(sample.label) and not report['changed_pair'])):
                        raise ValueError('Actual loader identity or supervision policy differs')
                    items.append((sample, report, entry))
                    targets[entry['pair_id']] = sample.translation_a_to_b_rc.copy() if entry['label'] else None
                batch = data.collate(items)
                ev = model.matcher(*(batch[k] for k in matcher.INPUTS))
                for i, entry in enumerate(chunk):
                    pair = evidence.PairEvidence.from_matcher(ev, i, batch['mask_a'], batch['mask_b'])
                    pred = model.score_pair(pair, threshold=.41)
                    candidates = []
                    for candidate in pred.clusters:
                        ids = torch.unique(candidate.proposal.edge_ids, dim=0)
                        candidates.append(dict(translation=candidate.translation.detach().tolist(),
                            score=float(candidate.readout.score), edges=len(ids),
                            sum_q=float(pair.q[ids[:, 0], ids[:, 1]].sum())))
                    row = dict(pair_id=entry['pair_id'], score=float(pred.score), accepted=bool(pred.accepted),
                        winner=pred.selected_cluster_id if pred.has_candidate else -1, candidates=candidates,
                        absolute_q=float(pair.q.sum()))
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    predictions.append(row)
                stream.flush()
                if (start + len(chunk)) % 100 == 0 or start + len(chunk) == len(entries):
                    print(json.dumps(dict(processed=start + len(chunk), total=900,
                        seconds=time.monotonic() - started)), flush=True)
            os.fsync(stream.fileno())
        if (before != helper.state_sha(model) or torch.cuda.is_initialized()
                or sha(manifest_path) != manifest_sha or sha(args.audit) != audit_sha
                or sha(args.helper_code) != args.helper_sha256
                or sha(provenance['checkpoint']) != provenance['checkpoint_sha256']):
            raise ValueError('Model/input/helper changed or CUDA initialized')
        labeled = label_predictions(entries, predictions, targets)
        groups = {kind: helper.summarize([r for r in labeled if r['recipe'] == kind]) for kind in 'MJR'}
        subtypes = {base: helper.summarize([r for r in labeled if r['base_kind'] == base])
                    for base in sorted({r['base_kind'] for r in labeled})}
        flagged = [base for base, stats in subtypes.items()
                   if stats['candidate_coverage'] is not None and stats['candidate_coverage'] < .2]
        summary = dict(**protocol, status='complete', groups=groups, subtypes=subtypes,
            low_coverage_subtypes=flagged, exclusion_by_model_score=False, model_unchanged=True,
            model_state_sha256=before, predictions_sha256=sha(out / 'predictions.jsonl'),
            elapsed_seconds=time.monotonic() - started, cuda_initialized=False)
        save(out / 'labeled_predictions.json', labeled)
        save(out / 'diagnostic_flags.json', dict(low_coverage_subtypes=flagged,
            pair_ids=[r['pair_id'] for r in labeled if r['base_kind'] in flagged], excluded=False))
        save(out / 'summary.json', summary)
        save(out / 'complete.json', dict(status='complete', rows=900,
            summary_sha256=sha(out / 'summary.json'), predictions_sha256=summary['predictions_sha256'],
            labeled_predictions_sha256=sha(out / 'labeled_predictions.json'),
            model_unchanged=True, model_state_sha256=before))
        print(json.dumps(dict(status='complete', groups=groups, low_coverage_subtypes=flagged), sort_keys=True))
    except BaseException as exc:
        save(out / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic() - started))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('data-root', 'audit', 'model-root', 'reference', 'helper-code', 'output-new'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--helper-sha256', required=True)
    p.add_argument('--threads', type=int, choices=(1, 2, 3, 4), default=2)
    run(p.parse_args())


if __name__ == '__main__':
    main()

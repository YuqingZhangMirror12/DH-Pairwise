"""Read-only remote curve extraction; stdout JSON only, no model loads/runs."""
import datetime
import json
import math
import pathlib
import re
import subprocess

ROOT = pathlib.Path('/root/autodl-tmp/rachel_score_design_20260913_001')
RUNS = {
    'shared_S4_S6_M12': ROOT / 'new_s345_20260914/matrix_bnfix_20260914/archive/s3_matrix/training',
    'S4': ROOT / 'new_s345_20260914/s4_cross_attention/training',
    'S6_D2': ROOT / 'attention_depth_20260915/s4_cross_attention_depth2/training',
    'S6_D4': ROOT / 'attention_depth_20260915/s4_cross_attention_depth4/training',
    'S7': ROOT / 's6_s7_20260915/priority_after_s5/s7_augmented_full24/training',
}


def read(path):
    return json.loads(path.read_text())


def subset(obj, keys):
    return {key: obj.get(key) for key in keys}


def val_score_diagnostics(path):
    rows = read(path)
    if isinstance(rows, dict):
        rows = rows['rows']
    bce = brier = 0.0
    pos = neg = 0
    positive_scores, negative_scores = [], []
    for row in rows:
        y = int(row['label'])
        probability = float(row['classification']['fused'])
        clipped = max(1e-12, min(1 - 1e-12, probability))
        bce -= y * math.log(clipped) + (1 - y) * math.log1p(-clipped)
        brier += (probability - y) ** 2
        (positive_scores if y else negative_scores).append(probability)
        pos += y
        neg += 1 - y
    return dict(source=str(path), sample_count=len(rows), positive_count=pos,
                negative_count=neg, pair_bce_from_saved_probabilities=bce / len(rows),
                brier=brier / len(rows),
                positive_mean_score=sum(positive_scores) / pos,
                negative_mean_score=sum(negative_scores) / neg,
                bce_clip_epsilon=1e-12)


def extract(name, root):
    if not root.exists():
        return dict(path=str(root), missing=True)
    protocol = read(root / 'protocol.json')
    args = protocol.get('arguments', {})
    segments = []
    for path in sorted(root.glob('segment_*.json')):
        obj = read(path)
        segment, training = obj['segment'], obj['training']
        if name == 'shared_S4_S6_M12' and segment['epoch'] > 12:
            continue
        segments.append(dict(source=str(path), **segment,
            loss_components=training['loss_components'], samples=training['samples'],
            optimizer_updates=training.get('optimizer_updates'),
            elapsed_s=training.get('elapsed_s'),
            pair_bce_weight=training.get('pair_bce_weight'),
            runtime_batching=training.get('runtime_batching'),
            valid_pair_count=training.get('valid_pair_count')))
    epochs = []
    for epoch in sorted({s['epoch'] for s in segments}):
        selected = [s for s in segments if s['epoch'] == epoch]
        n = sum(s['samples'] for s in selected)
        losses = {key: sum(s['loss_components'][key] * s['samples'] for s in selected) / n
                  for key in selected[0]['loss_components']}
        epochs.append(dict(epoch=epoch, phase=selected[0]['phase'], samples=n,
            segment_count=len(selected), source_segments=[pathlib.Path(s['source']).name for s in selected],
            learning_rates=sorted({s['learning_rate'] for s in selected}),
            weighted_mean_training_losses=losses,
            summed_train_seconds=sum(s['elapsed_s'] for s in selected),
            optimizer_updates=sum(s['optimizer_updates'] for s in selected)))
    validation = []
    for path in sorted(root.glob('validation_*.json')):
        if not re.match(r'validation_\d{3}\.json$', path.name):
            continue
        v = read(path)
        if name == 'shared_S4_S6_M12':
            continue
        record = dict(source=path.name, epoch=v['epoch'],
            global_exposure=v.get('global_exposure'),
            population=subset(v['validation'], ['sample_count', 'positive_count', 'negative_count', 'decision_coverage']),
            max_f1=v['validation']['methods']['fused'],
            recall95=v['operating_points']['validation']['recall_95'],
            selection_eligible=v.get('selection_eligible'),
            pose_used_for_selection=v['validation'].get('pose_used_for_selection'))
        score_path = root / ('validation_%03d_rows.json' % v['epoch'])
        record['score_diagnostics'] = val_score_diagnostics(score_path)
        record['score_diagnostics']['source'] = score_path.name
        validation.append(record)
    receipt_path = root / 'matcher_pretraining_receipt.json'
    receipt = read(receipt_path) if receipt_path.exists() else None
    status_path = root / 'status.json'
    status = read(status_path) if status_path.exists() else None
    checkpoint_epochs = sorted(int(p.stem.split('_')[-1]) for p in root.glob('epoch_*.pt'))
    return dict(path=str(root), protocol_source=str(root / 'protocol.json'),
        protocol_status=protocol.get('status'), status=status,
        max_existing_checkpoint_epoch=max(checkpoint_epochs) if checkpoint_epochs else None,
        arguments=subset(args, ['head_kind', 'cross_attention_depth', 'sampling', 'matcher_checkpoint',
          'stop_after_epoch', 'microbatch', 'physical_microbatch', 'effective_batch']),
        train_population=subset(protocol.get('populations', {}).get('train', {}), ['count', 'manifest', 'manifest_sha256']),
        val_population=subset(protocol.get('populations', {}).get('val', {}), ['count', 'manifest', 'manifest_sha256']),
        matcher_loss_config=protocol.get('matcher_loss_config'),
        classifier_loss=protocol.get('classifier_loss'),
        frozen_base_eval_in_classifier=protocol.get('frozen_base_eval_in_classifier'),
        matcher_receipt_source=str(receipt_path) if receipt else None, matcher_receipt=receipt,
        committed_segment_count=len(segments), source_paths_relative_to_run=True,
        epochs=epochs, validation=validation)


def main():
    observed = datetime.datetime.now(datetime.timezone.utc).isoformat()
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total',
                          '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
    processes = []
    for path in pathlib.Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            argv = (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace')
            if 'python' in argv and ('train_' in argv or 'run_recall_benchmark_queue' in argv) and ' -c ' not in argv:
                processes.append(dict(pid=int(path.name), argv=argv))
        except OSError:
            pass
    queues = []
    queue_paths = [ROOT / 's6_s7_20260915/priority_after_s5/queues' / name / 'queue_state.json'
                   for name in ['s6_depth_2_4', 's7_augmented_full24', 'original_tail_1', 'original_tail_2', 'original_tail_3']]
    queue_paths.append(ROOT / 's8_step2048_attention_20260916/queue/queue_state.json')
    for path in queue_paths:
        if path.exists():
            q = read(path)
            queues.append(dict(source=str(path), **subset(q, ['status', 'pid', 'active_stage', 'completed_at'])))
    out = dict(schema='scorer-convergence-read-only/1', observed_at_utc=observed,
        extraction_scope='Fixed completed training runs, training segments and clean SIMVAL only; no TEST/REAL/OOD used.',
        no_training_or_inference_performed=True,
        runtime=dict(gpu_csv=gpu, training_or_queue_processes=processes, queue_states=queues),
        runs={name: extract(name, path) for name, path in RUNS.items()})
    print(json.dumps(out, separators=(',', ':')))


if __name__ == '__main__':
    main()

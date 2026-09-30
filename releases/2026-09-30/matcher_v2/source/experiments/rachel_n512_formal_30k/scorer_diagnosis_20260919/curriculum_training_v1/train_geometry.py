"""One immutable TRAIN-only calibration shared by both presentation orders.

Reuse the established inherited-target rules on the admitted unique catalog,
not a fixed24K loop, requested gap limits, validation predictions or real data.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from .checkpoint_io import file_sha, write_json
from .exposure import STAGES, SampleRef, canonical_catalog, digest
from .model_adapter import AdmittedDataset, bound_module
from .verify_validation_preparation import bind_baseline

BASE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.'
DATASET = None
CALIBRATION = None
CATALOG = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def catalog_view(path, expected_sha256):
    require(file_sha(path) == expected_sha256, 'admission changed before geometry preparation')
    admission = read(path)
    require(admission['status'] == 'passed' and admission['schema'] == 'curriculum-data-admission/1'
            and admission['gpu_used'] is False, 'full completed TRAIN admission required')
    catalog = canonical_catalog(SampleRef(**row) for row in admission['catalog'])
    require([asdict(row) for row in catalog] == admission['catalog']
            and digest([asdict(row) for row in catalog]) == admission['catalog_sha256'], 'geometry catalog identity differs')
    require(set(row.stage for row in catalog) == set(STAGES), 'all three admitted TRAIN tiers required')
    for stage in STAGES:
        rows = [row for row in catalog if row.stage == stage]
        require(any(row.label for row in rows) and any(not row.label for row in rows), 'missing TRAIN class in a tier')
    # This is only a catalog view for the loader. It neither creates an exposure
    # ledger nor chooses a training budget to run this non-training audit.
    return admission, SimpleNamespace(catalog=catalog)


def inspect_index(index, dataset, calibration, stage):
    require(stage in STAGES, 'unregistered geometry TRAIN stage')
    entry = dataset.entries[index]
    require(entry['label'] is True, 'only positive inherited correspondences calibrate geometry')
    sample, _report, actual = dataset[index]
    require(actual == entry and sample.label and sample.translation_valid, 'TRAIN sample identity or GT missing')
    edges = calibration.inherited_edges(sample, entry['recipe'])
    require(edges.ndim == 2 and edges.shape[1] == len(calibration.COLUMNS) and len(edges) > 0
            and np.isfinite(edges).all(), 'invalid inherited-edge calibration result')
    record = dict(catalog_index=index, pair_id=entry['pair_id'], stage=stage,
        source_base_key=entry['source_base_key'], recipe=entry['recipe'],
        sample_sha256=entry['sample_sha256'], actual_matcher_input_sha256=entry['actual_matcher_input_sha256'],
        effective_training_target_sha256=entry['effective_training_target_sha256'],
        inherited_edges=len(edges), precise_anchor_edges=int(edges[:, -1].sum()))
    return record, edges


def worker_init(admission_path, expected_sha256, baseline_source):
    global DATASET, CALIBRATION, CATALOG
    torch.set_num_threads(1)
    bind_baseline(Path(baseline_source))
    _, view = catalog_view(admission_path, expected_sha256)
    CATALOG = view.catalog
    DATASET = AdmittedDataset(admission_path, expected_sha256, view, baseline_source)
    CALIBRATION = bound_module(BASE + 'calibrate_geometry', baseline_source)


def worker(index):
    return inspect_index(index, DATASET, CALIBRATION, CATALOG[index].stage)


def summarize_measurements(pairs, blocks, calibration):
    require(pairs and len(pairs) == len(blocks), 'missing geometry measurements')
    keys = [(row['stage'], row['pair_id']) for row in pairs]
    indices = [row['catalog_index'] for row in pairs]
    require(len(keys) == len(set(keys)) and indices == sorted(set(indices)), 'duplicate or shuffled calibration rows')
    for row, block in zip(pairs, blocks):
        require(block.shape == (row['inherited_edges'], len(calibration.COLUMNS)) and np.isfinite(block).all(),
                'per-pair inherited edge count differs')
        require(np.all(block[:, -1] == (row['recipe'] == 'clean')), 'precise anchors must follow the existing clean recipe rule')
    edges = np.concatenate(blocks)
    pair_index = np.concatenate([np.full(len(block), i, np.int32) for i, block in enumerate(blocks)])
    strata = dict(all=list(range(len(pairs))))
    for stage in STAGES:
        strata['stage:' + stage] = [i for i, row in enumerate(pairs) if row['stage'] == stage]
    for recipe in sorted(set(row['recipe'] for row in pairs)):
        strata['recipe:' + recipe] = [i for i, row in enumerate(pairs) if row['recipe'] == recipe]
    strata['corroded'] = [i for i, row in enumerate(pairs) if row['recipe'] != 'clean']
    require(strata.get('recipe:clean') and strata['corroded'],
            'both clean anchors and corroded TRAIN support required; no validation fallback')
    require(all(strata['stage:' + stage] for stage in STAGES), 'positive calibration missing a TRAIN tier')
    summaries = {name: calibration.summarize_edges(edges, pair_index, chosen) for name, chosen in strata.items()}
    parameters = calibration.derive_parameters(summaries)
    return edges, pair_index, summaries, parameters


def save_calibration(output, admission_path, admission, pairs, blocks, calibration, baseline_hashes, elapsed):
    """Write a new standalone result only after every admitted positive is read."""
    output = Path(output)
    require(not output.exists(), 'preserve earlier geometry outputs')
    expected = [(i, row['stage'], row['pair_id']) for i, row in enumerate(admission['catalog']) if row['label']]
    actual = [(row['catalog_index'], row['stage'], row['pair_id']) for row in pairs]
    require(actual == expected, 'must calibrate every admitted positive exactly once')
    edges, indices, strata, parameters = summarize_measurements(pairs, blocks, calibration)
    output.mkdir(parents=True)
    np.savez_compressed(output / 'inherited_edge_audit.npz', values=edges.astype(np.float32),
        pair_index=indices, columns=np.array(calibration.COLUMNS), source_support_known=np.ones(len(edges), bool))
    write_json(output / 'pairs.json', pairs)
    record = dict(status='complete', schema='s7-consensus-train-geometry/2',
        curriculum_training_catalog_sha256=admission['catalog_sha256'],
        curriculum_data_admission_sha256=file_sha(admission_path),
        curriculum_data_admission=str(Path(admission_path).resolve()),
        all_unique_train_positives_read=True, positive_counts_by_stage={stage:sum(row['stage'] == stage for row in pairs) for stage in STAGES},
        pairs=len(pairs), inherited_edges=len(edges), parameters=parameters, strata=strata,
        pair_weighting='each unique positive pair has equal total weight; not weighted by future repeated exposures',
        reused_baseline_calibration_sha256=file_sha(calibration.__file__), baseline_python_sha256=baseline_hashes,
        implementation_sha256=file_sha(__file__), artifacts={name:file_sha(output / name) for name in ('pairs.json', 'inherited_edge_audit.npz')},
        real_used=False, test_used=False, cal_used=False, select_used=False, predictions_used=False,
        gpu_used=False, optimizer_updates=0, budget_locked=False,
        fixed_pose_diameter_px=16., layout_correctness_label_px=20., pose_threshold_fitted=False,
        calibration_rule='Unchanged baseline derive_parameters: pair-equal reliable clean p95 localization Sigma, corroded tangential p95 evidence Sigma, ceil inherited corroded signed-normal p99 extent.',
        precise_anchor_note='Only inherited targets with recipe clean; no GT residual threshold relabels damaged anchors as precise.',
        shared_scope='one frozen geometry artifact for curriculum and mixed Matcher and both downstream heads',
        missing='No invented per-token ancestor IDs or dense-projection correspondences; no real-based tolerance fitting.',
        elapsed_seconds=elapsed)
    write_json(output / 'geometry_calibration.json', record)
    write_json(output / 'calibration_complete.json', dict(status='complete',
        geometry_sha256=file_sha(output / 'geometry_calibration.json'),
        admission_sha256=file_sha(admission_path), all_admitted_positives=True, gpu_used=False, training_started=False))
    return record


def run(admission_path, expected_sha256, baseline_source, preparation_path, output, workers=4):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1'), 'hide CUDA for TRAIN-only CPU calibration')
    require(type(workers) is int and 1 <= workers <= 4, 'at most four single-thread CPU workers')
    require(not Path(output).exists(), 'preserve earlier geometry outputs')
    preparation = read(preparation_path); source = Path(__file__).parent; baseline_source = Path(baseline_source).resolve()
    own_hashes = {p.name:file_sha(p) for p in source.glob('*.py')}
    baseline_hashes = {str(p.relative_to(baseline_source)):file_sha(p) for p in baseline_source.rglob('*.py')}
    require(preparation['status'] == 'passed' and preparation['source_sha256'] == own_hashes
            and preparation['baseline_python_sha256'] == baseline_hashes, 'calibration source preparation changed')
    admission, view = catalog_view(admission_path, expected_sha256)
    bind_baseline(baseline_source)
    calibration = bound_module(BASE + 'calibrate_geometry', baseline_source)
    positive = [i for i, row in enumerate(view.catalog) if row.label]
    began = time.time(); pairs, blocks = [], []
    # Explicit spawn prevents inherited torch-thread/CUDA process state. Each
    # worker reopens only the admitted immutable TRAIN loader and file bindings.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=worker_init, initargs=(str(admission_path), expected_sha256, str(baseline_source))) as pool:
        for pair, block in pool.map(worker, positive, chunksize=8):
            pairs.append(pair); blocks.append(block)
    require(file_sha(admission_path) == expected_sha256
            and own_hashes == {p.name:file_sha(p) for p in source.glob('*.py')}
            and baseline_hashes == {str(p.relative_to(baseline_source)):file_sha(p) for p in baseline_source.rglob('*.py')},
            'source or admitted data binding changed during calibration')
    return save_calibration(output, admission_path, admission, pairs, blocks, calibration, baseline_hashes, time.time() - began)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--admission-sha256', required=True)
    parser.add_argument('--baseline-source', type=Path, required=True)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    result = run(args.admission, args.admission_sha256, args.baseline_source, args.preparation, args.out, args.workers)
    print(json.dumps({name:result[name] for name in ('status', 'pairs', 'inherited_edges', 'parameters', 'gpu_used', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()

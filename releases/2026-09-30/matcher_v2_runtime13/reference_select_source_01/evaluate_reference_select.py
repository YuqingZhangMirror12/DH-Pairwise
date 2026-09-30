"""CPU-only, terminal-verified native Matcher evaluation of reference SELECT.

No live-source edits, training, threshold fitting, checkpoint selection, retries
or GPU use. This companion intentionally does not claim Scorer accuracy.
"""
import argparse
import importlib
import os
from pathlib import Path
import random
import sys
import traceback

import numpy as np
import torch

from admit_reference_select import (PACKAGE, BOUND, binding, inspect_entry, read,
    read_bound, require, save, sha, validate_manifest)

SPLIT = 'sim_reference_straight_select'


def verify_admission(root, complete_sha):
    root = Path(root)
    require(not (root/'failure.json').exists(), 'admission failure precedes completion')
    complete = read_bound(root/'complete.json', complete_sha)
    require(complete['status'] == 'complete' and complete['pairs'] == 900
            and complete['training_admitted'] is False, 'supplemental-only complete admission required')
    require(set(complete['files']) == {'protocol.json', 'duplicate_audit.json', 'admission.json'},
            'complete admission file membership differs')
    for name, signature in complete['files'].items():
        require(sha(root/name) == signature, 'admission artifact changed')
    plan = read(root/'admission.json')
    require(plan['schema'] == 'reference-select-admission/1'
            and plan['status'] == 'admitted_for_supplemental_evaluation_only'
            and plan['training_admitted'] is False and plan['reference_samples_modified'] is False
            and plan['model_scores_used_for_filtering'] is False
            and plan['exact_training_input_duplicates'] == plan['within_reference_duplicates'] == 0,
            'wrong reference admission policy')
    require(len(plan['rows']) == len(set(plan['groups']['all'])) == 900
            and [r['pair_id'] for r in plan['rows']] == plan['groups']['all'], 'complete fixed reference order required')
    for key, (path, signature) in BOUND.items():
        require(plan['input_bindings'][key] == dict(path=str(path), sha256=signature), 'wrong reference binding')
    for item in plan['input_bindings'].values():
        require(sha(item['path']) == item['sha256'], 'admission bound source changed')
    record = read(plan['input_bindings']['reference']['path'])
    entries = validate_manifest(record)
    require([e['pair_id'] for e in entries] == plan['groups']['all'], 'manifest/admission order differs')
    expected = {'all': [e['pair_id'] for e in entries]}
    for recipe in ('straight_M', 'straight_J', 'straight_R'):
        expected[recipe] = [e['pair_id'] for e in entries if e['recipe'] == recipe]
    require(plan['groups'] == expected, 'group membership changed')
    return plan, entries


def terminal_model(args):
    """A true terminal receipt/export is required before reading heldout samples."""
    require(sha(args.spec) == args.spec_sha, 'bound execution changed')
    source = args.source_root.resolve(); sys.path.insert(0, str(source))
    spec = read(args.spec)
    if args.arm == 'B0':
        execution = importlib.import_module(PACKAGE + '.curriculum_training_v1.execution')
        inputs = execution.load_inputs(spec)
        terminal = importlib.import_module(PACKAGE + '.curriculum_training_v1.matcher_terminal')
        saved, origin = terminal.verified_export(args.training_root, args.spec, inputs['plan'], 'curriculum', args.selection)
        importer = importlib.import_module(PACKAGE + '.curriculum_training_v1.verify_validation_preparation')
        importer.bind_baseline(inputs['baseline'])
        inference_source = inputs['baseline']
        model, geometry, origin = terminal.load_matcher(saved, origin, inference_source)
    else:
        runtime = importlib.import_module(PACKAGE + '.matcher_v2_v1.runtime_inputs')
        inputs = runtime.load_inputs(spec)
        require(spec['arm'] == args.arm and spec['module'] == 'matcher', 'wrong B-arm/native module')
        preparation = importlib.import_module(PACKAGE + '.matcher_v2_v1.evaluate')
        require(args.preparation is not None, 'tested B-arm runtime preparation required')
        preparation.check_preparation(args.preparation, inputs['source'])
        terminal = importlib.import_module(PACKAGE + '.matcher_v2_v1.terminal')
        saved, origin = terminal.verified_export(args.training_root, args.spec, inputs['plan'], args.selection)
        inference_source = inputs['source']
        model, geometry, origin = terminal.load_model(saved, origin, inference_source)
    require(source in Path(terminal.__file__).resolve().parents, 'wrong terminal verifier imported')
    require(saved['module'] == 'matcher' and origin['scorer_used'] is False
            and origin['real_used_for_selection'] is False, 'native SIM-preselected completed Matcher only')
    require(model.base.config.canvas_size == 800 and model.base.config.contour_cap == 512,
            'full800/N512 unchanged evaluation required')
    model.to('cpu'); model.eval(); model.requires_grad_(False)
    return model, geometry, origin, inference_source


class ReferenceDataset:
    def __init__(self, entries, plan, loader, catalog):
        self.entries = entries; self.plan = plan; self.loader = loader; self.catalog = catalog

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        entry, admitted = self.entries[i], self.plan['rows'][i]
        # Every read is checked, including the post-prediction target join.
        row = inspect_entry(entry, self.loader, self.catalog, reference_root=BOUND['reference'][0].parent)
        require(all(admitted[k] == v for k, v in row.items()), 'actual reference sample differs from admission')
        sample, report = self.loader(entry['sample_path'])
        return sample, report, entry


def targets_after_freeze(out, dataset):
    """Read labels/translations for metrics only after every prediction is durable."""
    proof = read(Path(out)/'prediction_complete.json')
    require(proof['pairs'] == len(dataset) and proof['targets_joined'] is False
            and proof['model_state_unchanged'] is True
            and proof['prediction_sha256'] == sha(Path(out)/'pair_predictions.jsonl'),
            'durable full predictions required before GT join')
    result = []
    for i in range(len(dataset)):
        sample, _, entry = dataset[i]
        require(bool(sample.translation_valid) == bool(sample.label), 'GT/label presence differs')
        result.append(dict(pair_id=entry['pair_id'], label=bool(sample.label),
            gt_pose=sample.translation_a_to_b_rc.tolist() if sample.label else None))
    return result


def run(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only evaluation required')
    require(not args.out.exists(), 'exclusive new evaluation output required')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    random.seed(26093041); np.random.seed(26093041); torch.manual_seed(26093041)
    model, geometry, origin, inference_source = terminal_model(args)
    # Holding a live/intermediate checkpoint never passes the terminal above.
    plan, entries = verify_admission(args.admission, args.admission_sha)
    loader = importlib.import_module('staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset')
    catalog = importlib.import_module(PACKAGE + '.curriculum_training_v1.catalog')
    require(sha(loader.__file__) == plan['input_bindings']['official_loader']['sha256']
            and sha(catalog.__file__) == plan['input_bindings']['canonical_inputs']['sha256'],
            'terminal model uses different actual-input/loader semantics')
    dataset = ReferenceDataset(entries, plan, loader.load_sample, catalog)
    data = importlib.import_module(PACKAGE + '.s7_consensus_v1.data')
    checkpoint = importlib.import_module(PACKAGE + '.curriculum_training_v1.checkpoint_io')
    runner = importlib.import_module(PACKAGE + '.curriculum_training_v1.matcher_run')
    before = checkpoint.tree_sha(model.state_dict())
    require(before == origin['matcher_state_sha256'], 'loaded Matcher state differs from selected export')
    code = {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}
    def batches():
        for start in range(0, len(dataset), 8):
            indices = range(start, min(start+8, len(dataset)))
            yield [entries[i] for i in indices], data.collate([dataset[i] for i in indices])
    source = dict(reference_manifest=plan['input_bindings']['reference'], admission=binding(args.admission/'complete.json'),
        simulation_revision='colleague-v4.2-original-select', preprocessing='unchanged materialized reference',
        independent_source_population=False, training_admitted=False)
    origin = dict(origin, split=SPLIT, source=source, arm=args.arm, device='cpu',
        code_sha256=code, evaluation_used_for_model_selection=False, threshold_fitting=False)
    rng_before = torch.get_rng_state().clone()
    runner.run_population(model, geometry, inference_source, dict(pairs=entries), batches(), source,
        out=args.out, provenance=origin, wanted_ids=set(plan['fixed_diagnostic_ids']), groups=plan['groups'],
        targets_callback=lambda: targets_after_freeze(args.out, dataset), device=torch.device('cpu'))
    verified = runner.verify_population(args.out)
    require(verified['pairs'] == 900 and verified['diagnostic_cases'] == 30, 'incomplete reference evaluation')
    require(checkpoint.tree_sha(model.state_dict()) == before and torch.equal(rng_before, torch.get_rng_state()),
            'frozen evaluation changed weights or torch RNG')
    require(not torch.cuda.is_initialized(), 'unexpected CUDA use')
    require(code == {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}, 'companion source changed')
    verify_admission(args.admission, args.admission_sha)
    save(args.out/'independent_artifact_audit.json', dict(verified, rng_unchanged=True, cuda_initialized=False,
        reference_unmodified=True, classification_accuracy=None, scorer_used=False))
    print(verified, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'spec', 'training-root', 'admission', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--preparation', type=Path)
    parser.add_argument('--spec-sha', required=True)
    parser.add_argument('--admission-sha', required=True)
    parser.add_argument('--arm', choices=('B0', 'B1', 'B2', 'B3'), required=True)
    parser.add_argument('--selection', choices=('sim_best', 'equal_budget_endpoint'), required=True)
    args = parser.parse_args(); existed = args.out.exists()
    try:
        run(args)
    except BaseException as error:
        if not existed:
            args.out.mkdir(parents=True, exist_ok=True)
            if not (args.out/'failure.json').exists():
                save(args.out/'failure.json', dict(status='failed', error=repr(error), traceback=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()

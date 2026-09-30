"""Supplemental original v4.2 SELECT with completed B0--B3 lightweight heads.

CPU only, SIM-frozen thresholds, no selection/calibration/retry/live-source edit.
The reference admission is reused, not regenerated. The tested shared evaluator
is imported under a separate namespace; its code is not patched or copied.
"""
import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import traceback

import numpy as np
import torch

PACKAGE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919'
ROOT = Path('/root/autodl-tmp/matcher_v2_20260930')
SPLIT = 'sim_reference_straight_select'
MODULES = ('scorer_patch', 'scorer_stats')
REFERENCE_CODE = {
    'admit_reference_select.py': '548acefefbf6788ba43de42129efb2df4b3cb91d64d81044ae26f18e4f39a2de',
    'evaluate_reference_select.py': 'cc76f2b9145a5ae004a3938a1f4d7fa45d5365803d3657cdf2ae5554d1ac9175',
}
READOUT_CODE = {
    '__init__.py': '0a37c27c913ce20d9b4b8969b41ed66fb559ff923493177dc23669541ee9b0cf',
    'evaluate.py': '8720fee8556764b3a6e5e13eee408fc63dfe0572ea5d82df4bec001c32993952',
    'audit.py': '0c2321da21f9a54a074970d78177ede76b7773cdddd6eb2f508523b003dc7217',
    'population.py': '18d8acdabb5f367bed272a255c7a3bf125faa0fdec147b9726b845e0bd2d3269',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def verify_code(root, expected):
    for name, signature in expected.items():
        require(sha(Path(root)/name) == signature, 'bound companion dependency changed: '+name)


def reference_api(source):
    source = Path(source).resolve(); verify_code(source, REFERENCE_CODE)
    sys.path.insert(0, str(source))
    api = importlib.import_module('evaluate_reference_select')
    admission = importlib.import_module('admit_reference_select')
    require(Path(api.__file__).resolve().parent == source
            and Path(admission.__file__).resolve().parent == source, 'wrong reference admission import')
    return api, admission


def readout_api(engine_root, entry):
    directory = Path(engine_root).resolve()/PACKAGE.replace('.', '/')/'curriculum_scorer_eval_v1'
    verify_code(directory, READOUT_CODE)
    name = PACKAGE+'.reference_select_scorer_readout'
    entry.alias(name, directory)
    evaluator = importlib.import_module(name+'.evaluate'); auditor = importlib.import_module(name+'.audit')
    require(Path(evaluator.__file__).resolve().parent == directory
            and Path(auditor.__file__).resolve().parent == directory, 'wrong shared readout import')
    return evaluator, auditor


def frozen_reference_threshold(origin):
    require(origin['selection_kind'] in ('sim_best', 'equal_budget_endpoint')
            and origin['selection_on_test'] is False and origin['threshold_refitted'] is False,
            'SIM-selected or fixed endpoint only; no target-population selection')
    result = copy.deepcopy(origin); threshold = result['thresholds']['sim_test']
    require(type(threshold) in (int, float) and .2 <= threshold <= .8, 'frozen SIM-CAL threshold required')
    result['thresholds'][SPLIT] = threshold
    result.setdefault('threshold_origins', {})[SPLIT] = 'SIM-CAL at the frozen selected update; reference SELECT not fitted'
    return result


def terminal_model(args):
    require(args.module in MODULES and args.arm in ('B0', 'B1', 'B2', 'B3'), 'registered Scorer arm/module required')
    require(sha(args.spec) == args.spec_sha, 'bound execution changed')
    source = args.source_root.resolve(); sys.path.insert(0, str(source)); spec = read(args.spec)
    entry = importlib.import_module(PACKAGE+'.curriculum_scorer_eval_v1.entry')
    package = source/PACKAGE.replace('.', '/')
    if args.arm == 'B0':
        execution = importlib.import_module(PACKAGE+'.curriculum_training_v1.execution')
        inputs = execution.load_inputs(spec)
        entry.check_preparation(args.preparation, package/'s7_consensus_eval_v14',
                                package/'binary_eval_v1', inputs['baseline'])
        terminal = importlib.import_module(PACKAGE+'.curriculum_scorer_eval_v1.terminal')
        saved, origin = terminal.verified_export(args.training_root, args.spec, inputs['plan'], args.selection)
        entry.bind_baseline(inputs['baseline'])
        inference_source = inputs['baseline']; base_spec = spec
        model, origin = terminal.load_model(saved, origin, inference_source)
    else:
        runtime = importlib.import_module(PACKAGE+'.matcher_v2_v1.runtime_inputs')
        inputs = runtime.load_inputs(spec)
        require(spec['arm'] == args.arm and spec['module'] == args.module, 'wrong B-arm/head execution')
        preparation = importlib.import_module(PACKAGE+'.matcher_v2_v1.evaluate')
        preparation.check_preparation(args.preparation, inputs['source'])
        terminal = importlib.import_module(PACKAGE+'.matcher_v2_v1.terminal')
        saved, origin = terminal.verified_export(args.training_root, args.spec, inputs['plan'], args.selection)
        inference_source = inputs['source']; base_spec = inputs['base_execution']
        model, _, origin = terminal.load_model(saved, origin, inference_source)
        require(origin['arm'] == args.arm, 'selected head belongs to another arm')
    require(source in Path(terminal.__file__).resolve().parents
            and source in Path(entry.__file__).resolve().parents, 'terminal verifier imported from another runtime')
    require(saved['module'] == origin['module'] == args.module
            and origin['selection_kind'] == args.selection
            and origin['matcher_updated_during_training'] is False
            and origin['local_conflict_head_present'] is False
            and origin['old_head_imported'] is False, 'completed own-Matcher/new lightweight head required')
    require(model.matcher.base.config.canvas_size == 800 and model.matcher.base.config.contour_cap == 512,
            'full800/N512 evaluation required')
    model.to('cpu'); model.eval(); model.requires_grad_(False)
    entry.bind_evaluation(package/'s7_consensus_eval_v14', package/'binary_eval_v1')
    return model, frozen_reference_threshold(origin), inference_source, base_spec, entry


def population_plan(admitted, base_spec, case_path):
    roles = base_spec['real_split']
    require(binding(roles['path']) == roles, 'role metadata changed')
    return dict(schema='reference-select-scorer-population/1', case_plan=binding(case_path), real_split=roles,
        simulation_revision='colleague-v4.2-original-select', pair_counts={SPLIT:len(admitted['rows'])})


def target_join(out, dataset):
    out = Path(out); frozen = read(out/'prediction_complete.json')
    require(frozen.get('status') == 'all_predictions_frozen' and frozen.get('pairs') == len(dataset)
            and frozen.get('model_state_unchanged') is True and frozen.get('gt_used_for_prediction') is False
            and frozen.get('sha256') == sha(out/'pair_predictions.jsonl'),
            'complete durable unlabeled predictions required before target joining')
    targets = []
    for i in range(len(dataset)):
        sample, _, entry = dataset[i]
        require(bool(sample.translation_valid) == bool(sample.label), 'reference positive/GT presence differs')
        targets.append(dict(pair_id=entry['pair_id'], label=bool(sample.label),
            gt_pose=sample.translation_a_to_b_rc.tolist() if sample.label else None))
    return targets


def group_rows(entries, split, roles):
    require(split == SPLIT, 'this companion only evaluates original reference SELECT')
    return {'all':entries, **{key:[entry for entry in entries if entry['recipe'] == key]
                              for key in ('straight_M', 'straight_J', 'straight_R')}}


def run(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only supplemental Scorer evaluation required')
    require(not args.out.exists(), 'exclusive new evaluation root required; no automatic retries')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    random.seed(26093042); np.random.seed(26093042); torch.manual_seed(26093042)
    # Load and verify a completed head before touching heldout sample contents.
    model, origin, inference_source, base_spec, entry = terminal_model(args)
    api, admission = reference_api(args.reference_source)
    admitted, entries = api.verify_admission(args.admission, args.admission_sha)
    loader = importlib.import_module('staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset')
    catalog = importlib.import_module(PACKAGE+'.curriculum_training_v1.catalog')
    require(sha(loader.__file__) == admitted['input_bindings']['official_loader']['sha256']
            and sha(catalog.__file__) == admitted['input_bindings']['canonical_inputs']['sha256'],
            'official-loader/actual-input semantics differ from reference admission')
    dataset = api.ReferenceDataset(entries, admitted, loader.load_sample, catalog)
    data = importlib.import_module(PACKAGE+'.s7_consensus_v1.data')
    checkpoint = importlib.import_module(PACKAGE+'.curriculum_training_v1.checkpoint_io')
    evaluator, auditor = readout_api(args.engine_root, entry)
    case_path = args.source_root/PACKAGE.replace('.', '/')/'s7_consensus_eval_v14/case_plan.json'
    plan = population_plan(admitted, base_spec, case_path)
    source = dict(reference_manifest=admitted['input_bindings']['reference'], admission=binding(args.admission/'complete.json'),
        simulation_revision=plan['simulation_revision'], reference_samples_modified=False,
        independent_source_population=False, training_admitted=False)
    def batches():
        for start in range(0, len(dataset), 8):
            indices = range(start, min(start+8, len(dataset)))
            yield [entries[i] for i in indices], data.collate([dataset[i] for i in indices])
    def load_population(split, _plan, _root):
        require(split == SPLIT and _plan == plan, 'wrong supplemental population')
        return dict(pairs=entries), batches(), source, dataset
    code = {p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    before = checkpoint.tree_sha(model.state_dict()); rng_before = torch.get_rng_state().clone()
    origin.update(arm=args.arm, scorer_used=True, code_sha256=code, reference_code_sha256=REFERENCE_CODE,
        readout_code_sha256=READOUT_CODE, source_runtime=str(args.source_root.resolve()),
        readout_runtime=str(args.engine_root.resolve()), device='cpu', evaluation_used_for_model_selection=False,
        threshold_fitting=False, training_admitted=False)
    evaluator.run_population(model, inference_source, plan, SPLIT, origin, args.out, torch.device('cpu'),
        registered_splits=(SPLIT,), population_loader=load_population, group_builder=group_rows,
        target_loader=lambda meta, split, _plan, ds:target_join(args.out, ds),
        extra_diagnostic_ids=admitted['fixed_diagnostic_ids'])
    proof = auditor.verify_population(args.out)
    require(proof['pairs'] == 900 and proof['diagnostic_cases'] == 30, 'complete reference900/30fixed cases required')
    require(checkpoint.tree_sha(model.state_dict()) == before and torch.equal(rng_before, torch.get_rng_state()),
            'evaluation changed model or CPU RNG')
    require(not torch.cuda.is_initialized(), 'unexpected CUDA allocation')
    require(code == {p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}, 'companion source changed')
    verify_code(args.reference_source, REFERENCE_CODE)
    verify_code(args.engine_root/PACKAGE.replace('.', '/')/'curriculum_scorer_eval_v1', READOUT_CODE)
    api.verify_admission(args.admission, args.admission_sha)
    save(args.out/'independent_artifact_audit.json', dict(proof, rng_unchanged=True, cuda_initialized=False,
        reference_unmodified=True, scorer_used=True, threshold_refitted=False, arm=args.arm, module=args.module))
    print(proof, flush=True)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'engine-root', 'spec', 'preparation', 'training-root', 'reference-source', 'admission', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--spec-sha', required=True); parser.add_argument('--admission-sha', required=True)
    parser.add_argument('--arm', choices=('B0', 'B1', 'B2', 'B3'), required=True)
    parser.add_argument('--module', choices=MODULES, required=True)
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

"""Adapter for the frozen native evaluator, with genuine per-pair loss.

No TRAIN data, architecture, proposal policy, or loss formula is changed.
The historical evaluator repeats its batch mean in each row; forcing an
evaluation microbatch of exactly one makes that quantity the full pair loss.
This adapter never calibrates thresholds or reads real/TEST metrics.
"""
from dataclasses import replace
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys

LOSS_SEMANTICS = 'per_pair_full_matcher_loss/1'
IMPLEMENTATION = 'frozen_native_evaluate_view_microbatch1/1'
MODULE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.evaluation'
HASH_MODULE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.curriculum_training_v1.checkpoint_io'
RUNTIME_PREFIXES = ('experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.',
                    'staging.pairwise_v0_2.')


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def check_loaded_runtime_modules(source_root):
    """Reject a native dependency preloaded from an unfrozen checkout.

    The external v2 orchestration package is intentionally separate and must
    have its own source receipt in the launcher. All loaded historical runtime
    modules must come from the bound native source, including lazy imports.
    """
    root = Path(source_root).resolve()
    for name, module in tuple(sys.modules.items()):
        if not name.startswith(RUNTIME_PREFIXES) or name.startswith(__package__ + '.') or name == __package__:
            continue
        path = getattr(module, '__file__', None)
        if path and root not in Path(path).resolve().parents:
            raise ValueError('native dependency imported outside frozen source: ' + name)


def bind_materialized_select(plan, manifest, expected_sha):
    """Join the native loader's file list to the frozen lineage population.

    The historical Dataset verifies IDs/labels but not NPZ bytes. Require
    absolute paths so the new wrapper and that unmodified loader resolve the
    same files, and hash every actual sample against the protocol. This is not
    a substitute for the upstream pixel/target/lineage admission audit.
    """
    from .protocol import validate_protocol, require
    validate_protocol(plan, verify_files=True)
    manifest = Path(manifest).resolve(strict=True)
    require(file_sha(manifest) == expected_sha, 'materialized SELECT manifest changed')
    data = json.loads(manifest.read_text())
    require(data.get('split') == 'select' and data.get('real_used') is False
            and data.get('test_used') is False, 'materialized SIM SELECT only')
    expected = {e['pair_id']: e for e in plan['manifest_bindings']['select']['manifest']['entries']}
    entries = data.get('entries')
    require(isinstance(entries, list) and all(isinstance(e, dict) for e in entries)
            and len(entries) == len(expected), 'materialized SELECT population differs')
    ids = [e.get('pair_id') for e in entries]
    require(all(isinstance(pid, str) for pid in ids) and len(set(ids)) == len(ids)
            and set(ids) == set(expected), 'materialized SELECT identities differ')
    files = {}
    for entry in entries:
        identity = expected[entry['pair_id']]
        require(type(entry.get('label')) is bool and entry['label'] == identity['label'],
                'materialized SELECT label differs')
        # recipe affects pose_enabled in the frozen collator; it must be
        # explicit and part of the manifest bytes bound to the launch receipt.
        require(isinstance(entry.get('recipe'), str) and bool(entry['recipe']),
                'explicit frozen recipe required')
        path = entry.get('sample_path')
        if not path:
            artifact = Path(entry.get('artifact_path', ''))
            base = Path(data.get('artifact_root', manifest.parent))
            require(base.is_absolute() and bool(entry.get('artifact_path')),
                    'absolute artifact root and sample path required')
            path = base / artifact
        path = Path(path)
        require(path.is_absolute(), 'absolute sample path required for native loader parity')
        resolved = path.resolve(strict=True)
        require(resolved.is_file() and file_sha(resolved) == identity['sample_sha256'],
                'actual SELECT sample bytes differ from frozen lineage')
        for key in ('sample_sha256', 'stage', 'generator'):
            require(key not in entry or entry[key] == identity[key],
                    'materialized SELECT stratum/hash differs')
        # Keep the original absolute path (not only its resolved target) so a
        # symlink retarget during evaluation is caught by the closing check.
        files[entry['pair_id']] = (str(path), identity['sample_sha256'])
    return files


def evaluate_protocol_select(plan, model, manifest, expected_sha, device, config,
                             source_root, source_inventory):
    """Return exact selector rows after binding materialized NPZs to the plan.

    The future process launcher must additionally bind this materialized
    manifest SHA, the evaluator source inventory, configuration, committed
    checkpoint, and protocol before invoking this function. This function does
    not write or manufacture launch/actual-process-return receipts.
    """
    files = bind_materialized_select(plan, manifest, expected_sha)
    rows = evaluate_matcher_select(model, manifest, expected_sha, device, config,
                                   source_root, source_inventory)
    if files != bind_materialized_select(plan, manifest, expected_sha):
        raise ValueError('materialized SELECT binding changed during inference')
    expected = {e['pair_id']: e for e in plan['manifest_bindings']['select']['manifest']['entries']}
    fields = ('gt_known', 'layout20', 'candidate_coverage', 'numeric_valid', 'has_candidate',
              'matcher_loss', 'loss_semantics', 'loss_implementation', 'physical_microbatch')
    return [dict({key: expected[row['pair_id']][key]
                  for key in ('pair_id', 'sample_sha256', 'stage', 'generator', 'label')},
                 **{key: row[key] for key in fields}) for row in rows]


def normalize_native_rows(rows, entries):
    """Keep every population member, including failed/no-candidate positives."""
    ids=[e['pair_id'] for e in entries]
    if len(set(ids))!=len(ids) or len(rows)!=len(ids):
        raise ValueError('duplicate manifest identity or incomplete native result')
    by_id={e['pair_id']:e for e in entries}; seen=set(); out=[]
    for row in rows:
        pid=row.get('pair_id')
        if pid in seen or pid not in by_id:
            raise ValueError('duplicate or foreign native result')
        seen.add(pid); entry=by_id[pid]
        if type(row.get('label')) is not bool or row['label']!=entry['label']:
            raise ValueError('native label differs from bound SELECT')
        loss=row.get('matcher_batch_loss')
        if type(loss) not in (int,float) or not math.isfinite(loss) or loss<0:
            raise ValueError('finite nonnegative full pair loss required')
        for k in ('numeric_valid','has_candidate','gt_known','layout20','candidate_coverage'):
            if type(row.get(k)) is not bool:raise ValueError('native boolean missing: '+k)
        if entry['label'] and not row['gt_known']:
            raise ValueError('positive SELECT pair must have original layout ground truth')
        if row['layout20'] and not (row['numeric_valid'] and row['has_candidate'] and row['gt_known'] and row['label']):
            raise ValueError('invalid or absent proposal cannot count as correct layout')
        value={k:v for k,v in row.items() if k!='matcher_batch_loss'}
        value.update(matcher_loss=float(loss),loss_semantics=LOSS_SEMANTICS,
                     loss_implementation=IMPLEMENTATION,physical_microbatch=1)
        out.append(value)
    return sorted(out,key=lambda r:r['pair_id'])


def evaluate_matcher_select(model, manifest, expected_sha, device, config,
                            source_root, source_inventory):
    """Validate an isolated source and SELECT manifest before any native call.

    source_inventory is the previously frozen complete Python inventory, not
    a hash inferred after inference. Publication/export needs the separate
    protocol/lineage and committed-checkpoint admission gates as well.
    """
    manifest=Path(manifest).resolve(); root=Path(source_root).resolve()
    if file_sha(manifest)!=expected_sha:raise ValueError('SELECT manifest changed')
    data=json.loads(manifest.read_text())
    if data.get('split')!='select' or data.get('real_used') is not False or data.get('test_used') is not False:
        raise ValueError('explicit SIM SELECT only; CAL/real/TEST forbidden')
    entries=data.get('entries',[])
    if not entries:raise ValueError('empty SELECT manifest')
    actual={str(p.relative_to(root)):file_sha(p) for p in root.rglob('*.py')}
    if not actual or actual!=source_inventory:raise ValueError('frozen evaluator source differs')
    evaluator=importlib.import_module(MODULE)
    if root not in Path(evaluator.__file__).resolve().parents:
        raise ValueError('native evaluator imported from a different source')
    hashing=importlib.import_module(HASH_MODULE)
    if root not in Path(hashing.__file__).resolve().parents:
        raise ValueError('model hash implementation imported from a different source')
    check_loaded_runtime_modules(root)
    # This is an evaluation-only change, not a change to training batch size.
    eval_config=replace(config,microbatch=1)
    if eval_config.microbatch!=1:raise ValueError('per-pair loss requires evaluation batch one')
    matcher=model.matcher
    if any(p.requires_grad for p in matcher.parameters()):
        raise ValueError('post-hoc Matcher must already be frozen')
    # No dropout/batchnorm/training-root loophole: every submodule is checked.
    if any(m.training for m in matcher.modules()):
        raise ValueError('post-hoc Matcher must already be entirely in eval mode')
    before=hashing.tree_sha(matcher.state_dict())
    rows=evaluator.evaluate_view(model,str(manifest),expected_sha,'matcher',device,eval_config,cache=None)
    check_loaded_runtime_modules(root)
    if {str(p.relative_to(root)):file_sha(p) for p in root.rglob('*.py')} != source_inventory:
        raise ValueError('frozen evaluator source changed during inference')
    if file_sha(manifest)!=expected_sha:raise ValueError('SELECT changed during inference')
    if hashing.tree_sha(matcher.state_dict())!=before:
        raise ValueError('Matcher weights changed during evaluation')
    if any(m.training for m in matcher.modules()) or any(p.requires_grad for p in matcher.parameters()):
        raise ValueError('Matcher mode/freeze changed during evaluation')
    return normalize_native_rows(rows,entries)

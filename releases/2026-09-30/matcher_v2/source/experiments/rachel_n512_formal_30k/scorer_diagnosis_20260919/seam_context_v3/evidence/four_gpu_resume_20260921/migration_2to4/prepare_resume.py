"""Build an explicit topology-only migration plan and evidence from real checks."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

MODULE = 'experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/seam_context_v3'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--finalize', action='store_true')
args = p.parse_args()
root = Path(args.root)
folder = root / 'migration_2to4'
folder.mkdir(exist_ok=True)
source = root / 'source' / MODULE
checkpoint = root / 'training/paused_A_resume011_step7500_20260921.pt'
cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
binding = {x: sha(root / 'data' / x) for x in cp['binding'] if x != 'implementation'}
binding['implementation'] = {x.name: sha(x) for x in sorted(source.glob('*.py'))}
binding['implementation']['beam_kernel.cpp'] = sha(source / 'beam_kernel.cpp')
old = cp['binding']['implementation']
changed = sorted(k for k in old.keys() | binding['implementation'].keys()
                 if old.get(k) != binding['implementation'].get(k))
assert set(changed) == {'train.py', 'launch.py', 'distributed_check.py'}, changed
assert {k: v for k, v in binding.items() if k != 'implementation'} == {
    k: v for k, v in cp['binding'].items() if k != 'implementation'}
assert (cp['world_size'], cp['stage'], cp['epoch'], cp['offset'], cp['updates']) == (2, 'A', 11, 0, 7500)
digest = hashlib.sha256()
for k, v in cp['model'].items():
    digest.update(k.encode())
    digest.update(v.cpu().contiguous().numpy().tobytes())
plan = dict(authorized_2_to_4_resume=True,
            authorization='User confirmed server reboot and4GPUs ready; requested4GPU resume',
            created_at_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            source_checkpoint=str(checkpoint), source_checkpoint_sha256=sha(checkpoint),
            source_binding=cp['binding'], destination_binding=binding,
            source_world_size=2, target_world_size=4, microbatch={'A': 8, 'B': 8},
            effective_batch=32, changed_runtime_files=changed,
            model_digest=digest.hexdigest(),
            new_rank_seeds={'2': 2268427, '3': 3268430},
            resume_position={k: cp[k] for k in ('stage', 'epoch', 'offset', 'updates', 'exposures')},
            rng_policy='Keep saved ranks0/1 streams; seed new ranks2/3 independently; no bitwise equivalence claim',
            preserved=['all model/loss/data code', 'config', 'model weights', 'optimizer and LR',
                       'selection state', 'epoch and exposures', 'effective batch and updates per epoch'])

# Standard DistributedSampler partitions the same consecutive global update
# groups with either topology; assert that on the exact resumed epoch seed.
generator = torch.Generator().manual_seed(cp['sampler_seed'] + cp['epoch'])
order = torch.randperm(24000, generator=generator).tolist()
for step in range(750):
    two = [order[rank::2][step * 16:(step + 1) * 16] for rank in range(2)]
    four = [order[rank::4][step * 8:(step + 1) * 8] for rank in range(4)]
    assert sorted(sum(two, [])) == sorted(sum(four, []))
plan['sampler_check'] = dict(passed=True, epoch=11, pairs=24000, updates=750,
                             same_global_pair_membership_each_update=True)
if not args.finalize:
    if sha(root / 'training/last.pt') != plan['source_checkpoint_sha256']:
        raise RuntimeError('Last checkpoint advanced; re-plan rather than overwrite')
    save(folder / 'resume_manifest.json', plan)
else:
    existing = read(folder / 'resume_manifest.json')
    assert existing['source_checkpoint_sha256'] == plan['source_checkpoint_sha256']
    assert existing['destination_binding'] == binding
    distributed = read(folder / 'distributed_check_4gpu.json')
    assert distributed['passed'] and not distributed['weights_saved']
    assert {x['stage'] for x in distributed['events']} == {'A', 'B'}
    assert len(distributed['full_arc_stress']) == 4
    for event in distributed['events']:
        assert (event['world_size'], event['microbatch'], event['effective_batch']) == (4, 8, 32)
        assert all(r['max_parameter_difference'] <= 1e-6 for r in event['all_ranks'])
    assert all(x['status'] == 'passed' for x in distributed['full_arc_stress'])
    original = read(root / 'preflight_mirror/preflight.json')
    gpu_tests = read(root / 'gpu_tests_mirror.json')
    assert original['passed'] and gpu_tests['passed']
    hashes = {x.name: sha(x) for x in source.glob('*.py') if x.name not in ('launch.py', 'gpu_tests.py')}
    hashes['beam_kernel.cpp'] = sha(source / 'beam_kernel.cpp')
    evidence = dict(passed=True, microbatch={'A': 8, 'B': 8}, effective_batch=32,
                    intended_world_size=4, precision='fp32',
                    kind='topology-only resumed-checkpoint4GPU check',
                    source_hashes=hashes, distributed_check=str(folder / 'distributed_check_4gpu.json'),
                    reference_original_preflight=str(root / 'preflight_mirror/preflight.json'),
                    original_model_learning_check_reused=True,
                    model_loss_data_code_unchanged=True,
                    original_unit_tests_reused=str(root / 'gpu_tests_mirror.json'),
                    full_arc_stress=distributed['full_arc_stress'],
                    batch_trials=distributed['events'],
                    initialization_weights_saved=False, formal_training_started=False)
    save(folder / 'preflight_4gpu.json', evidence)
print(json.dumps({'prepared': True, 'finalized': args.finalize, 'changed_files': changed,
                  'resume_position': plan['resume_position'], 'target_world_size': 4,
                  'effective_batch': 32, 'microbatch': 8, 'model_digest': plan['model_digest']}, ensure_ascii=False))

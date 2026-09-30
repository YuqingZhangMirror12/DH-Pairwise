"""Mechanical, fail-closed integration into a NEW immutable source snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

PARENT = Path('experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919')


def inventory(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*.py'))}


def change(path, old, new, count=1):
    text = path.read_text()
    if text.count(old) != count:
        raise ValueError(f'Unexpected source in {path}: {old[:70]!r}, count={text.count(old)}')
    path.write_text(text.replace(old, new))


def build(workspace, out):
    workspace, out = Path(workspace).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    baseline = workspace / 'artifacts/binary_micro32_20260928/source'
    before = inventory(baseline)
    shutil.copytree(baseline, out/'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    new = out/'source'/PARENT/'decoder_readout_v1'
    shutil.copytree(workspace/PARENT/'decoder_readout_v1', new,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    pkg = out/'source'/PARENT/'s7_consensus_v1'
    train = pkg/'train.py'
    change(train, 'from ..binary_scorer_v1.head import BinaryClusterHead',
           'from ..decoder_readout_v1.head import EvidenceClusterHead')
    change(train, 'return BinaryClusterHead(variant)',
           'return EvidenceClusterHead(variant, remove_overlap=False)')
    change(train, "choices=('patch','stats')", "choices=('patch_mean','patch_sum')")
    change(train, "binding['real_development']=bind_plan(args.real_split)",
           "binding['pooling_scorer_sha256']={p.name:digest(p) for p in (Path(__file__).parent.parent/'decoder_readout_v1').glob('*.py')}\n        binding['real_development']=bind_plan(args.real_split)")
    change(train, "frozen_hash=state_digest(model.matcher) if stage=='scorer' else None",
           "frozen_hash=state_digest(model.matcher) if stage=='scorer' else None\n    initial_head_hash=state_digest(model.head)")
    change(train, "receipt=dict(status='passed',formal_training=False,updated_weights_discarded=True,",
           "if state_digest(model.head)==initial_head_hash:\n                        raise AssertionError('fresh head did not update')\n                    receipt=dict(status='passed',formal_training=False,updated_weights_discarded=True,\n                        head_updated=True,initial_head_sha256=initial_head_hash,final_head_sha256=state_digest(model.head),")
    config = pkg/'config.py'
    change(config, "schema: str = 'binary-cluster-scorer/1'", "schema: str = 'pooling-cluster-scorer/1'")
    change(config, "scorer_variant: str = 'patch'", "scorer_variant: str = 'patch_mean'")
    old = "head=dict(variant=self.scorer_variant,whole_cluster_binary=True,attention=False,local_conflict=False,learned_refinement=False,edge_mlp=[392,64,32] if self.scorer_variant=='patch' else None,cluster_mlp=[80 if self.scorer_variant=='patch' else 16,64,32,1])"
    newhead = "head=dict(variant=self.scorer_variant,whole_cluster_binary=True,attention=False,local_conflict=False,learned_refinement=False,edge_mlp=[392,64,32],cluster_mlp=[48,64,32,1],overlap_statistic_retained=True,pooling=('q_arc_conditional_mean' if self.scorer_variant=='patch_mean' else 'log1p_sum_q_arc_sigmoid_embedding'),head_parameter_count=32481)"
    change(config, old, newhead)
    # Inherited launchers are intentionally not a supported entry for this experiment.
    (pkg/'launch.py').write_text("raise RuntimeError('Use the external pooling priority launcher; inherited controller is disabled')\n")
    for src, dst in [('binary_eval_v1', 'pooling_eval_v1'), ('s7_consensus_eval_v14','s7_consensus_eval_v14')]:
        shutil.copytree(workspace/PARENT/src, out/dst,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    ev = out/'pooling_eval_v1'
    # Adapter mechanics, source binding, role split, terminal checks and arithmetic
    # audits are inherited; only the explicit architecture and pooling arithmetic change.
    for path in ev.glob('*.py'):
        text=path.read_text()
        text=text.replace("('patch','stats')", "('patch_mean','patch_sum')")
        text=text.replace("['patch','stats']", "['patch_mean','patch_sum']")
        text=text.replace("'binary-cluster-scorer/1'", "'pooling-cluster-scorer/1'")
        text=text.replace("34529 if a.variant=='patch' else 3201", '32481')
        text=text.replace("34529 if VARIANT=='patch' else 3201", '32481')
        # Synthetic network fixture uses actual new head with 4-D input features.
        text=text.replace('binary_scorer_v1.test_binary import fixture',
                          'decoder_readout_v1.test_runtime import fixture')
        path.write_text(text)
    loading=ev/'loading.py'
    change(loading, "common.verify_code_binding(binding)",
        "common.verify_code_binding(binding)\n    if inventory(Path(training.__file__).parent.parent/'decoder_readout_v1')!=binding['pooling_scorer_sha256']:\n        raise ValueError('pooling implementation changed')")
    testload=ev/'test_loading.py'
    change(testload, "b.update(implementation_sha256=inventory(code),binary_scorer_sha256=inventory(code.parent/'binary_scorer_v1'),",
        "b.update(pooling_scorer_sha256=inventory(code.parent/'decoder_readout_v1'),implementation_sha256=inventory(code),binary_scorer_sha256=inventory(code.parent/'binary_scorer_v1'),")
    snap=ev/'snapshot.py'
    # Both controls have patch/context inputs and an actual 32-channel edge encoder.
    change(snap, "if variant=='patch':", "if variant in ('patch_mean','patch_sum'):")
    change(snap, "if m['variant']=='patch':", "if m['variant'] in ('patch_mean','patch_sum'):")
    change(snap, "h=get(lookup['edge_mlp.3']['output']);pooled=np.concatenate([(h*cw[:,None]).sum(0),h.max(0)])",
        "h=get(lookup['edge_mlp.3']['output'])\n            pooled=((h*cw[:,None]).sum(0) if m['variant']=='patch_mean' else np.log1p((w[:,None]/(1+np.exp(-h))).sum(0)))")
    change(snap, "'mean/max pooling differs'", "'declared pooling arithmetic differs'")
    change(snap, "pooled_features=None if r.pooled_features is None else a.add(prefix+'/pooled_features',r.pooled_features),",
        "pooled_features=None if r.pooled_features is None else a.add(prefix+'/pooled_features',r.pooled_features),\n            contributions=a.add(prefix+'/edge_contributions',r.contributions),")
    change(snap, "close(get(c['pooled_features']),pooled,'declared pooling arithmetic differs')",
        "close(get(c['pooled_features']),pooled,'declared pooling arithmetic differs')\n            parts=h*cw[:,None] if m['variant']=='patch_mean' else w[:,None]/(1+np.exp(-h))\n            close(get(c['contributions']),parts,'actual edge contribution differs')")
    change(snap, "'FP64 conditional normalization for mean pooling; absolute Q/count/mass retained in statistics'",
        "'FP64 conditional normalization used only by mean arm; sum arm never divides by total mass; absolute Q/count/mass also retained'" )
    verify=ev/'verify_preparation.py'
    change(verify, "names=['consensus_binary_eval_adapter.'+n for n in ('test_contracts','test_snapshot','test_evaluate','test_entry','test_loading')]",
        "names=['consensus_binary_eval_adapter.'+n for n in ('test_contracts','test_snapshot','test_evaluate','test_entry','test_loading')]\n    names += [training.__package__.rsplit('.',1)[0]+'.decoder_readout_v1.test_runtime']")
    shutil.copyfile(workspace/'artifacts/binary_micro32_20260928/real_split.json',out/'real_split.json')
    assert before==inventory(baseline), 'original source changed'
    record=dict(status='prepared_not_gpu_started',baseline_sha256=before,
        training_source_sha256=inventory(out/'source'),evaluation_sha256=inventory(ev),
        common_sha256=inventory(out/'s7_consensus_eval_v14'),baseline_unchanged=True,
        variants=['patch_mean','patch_sum'],search_unchanged=True,overlap_feature_retained=True)
    (out/'source_preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':record['status'],'python_files':len(record['training_source_sha256'])}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();build(a.workspace,a.out)

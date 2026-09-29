"""CPU synthetic tests only. Does not open real weights/data or allocate GPUs."""
import argparse
import importlib
import io
import os
from pathlib import Path
import time
import unittest

from .. import curriculum_training_v1 as curriculum
from ..curriculum_training_v1.checkpoint_io import file_sha, write_json
from ..curriculum_training_v1.verify_validation_preparation import bind_baseline
from .entry import bind_evaluation, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-source','common-source','binary-source','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--scope',choices=('terminal','complete'),default='complete')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('explicitly hide all GPUs for CPU preparation')
    args.baseline_source = args.baseline_source.resolve()
    def bindings():
        return dict(adapter_python_sha256=inventory(Path(__file__).parent),
            curriculum_python_sha256=inventory(Path(curriculum.__file__).parent),
            common_python_sha256=inventory(args.common_source),binary_python_sha256=inventory(args.binary_source),
            baseline_python_sha256={str(p.relative_to(args.baseline_source)):file_sha(p)
                for p in args.baseline_source.rglob('*.py')})
    before = bindings(); bind_baseline(args.baseline_source)
    bind_evaluation(args.common_source,args.binary_source)
    names = ['test_terminal'] + (['test_evaluate','test_audit_controller'] if args.scope == 'complete' else [])
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(__package__+'.'+name) for name in names)
    args.out.mkdir(parents=True,exist_ok=False); start=time.time(); stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (args.out/'tests.log').write_text(stream.getvalue())
    same = before == bindings()
    passed = result.wasSuccessful() and not result.skipped and same
    receipt = dict(schema='curriculum-scorer-evaluation-preparation/1',status='passed' if passed else 'failed',
        scope=args.scope,tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        skipped=len(result.skipped),verified_variants=['patch','stats'] if passed else [],**before,
        source_files_unchanged=same,test_log_sha256=file_sha(args.out/'tests.log'),elapsed_seconds=time.time()-start,
        gpu_used=False,real_inference_performed=False,real_checkpoints_opened=False,formal_training_started=False,
        synthetic_cpu_optimizer_and_terminal_parser_tested=True,
        actual_full_network_state_loading_tested=True,
        synthetic_population_and_actual_head_evidence_tested=args.scope=='complete',
        independent_artifact_recomputation_tested=args.scope=='complete',
        six_job_controller_with_actual_cpu_children_tested=args.scope=='complete',
        gpu_gate_passed=False,automatic_queue_registered=False)
    write_json(args.out/'preparation.json',receipt)
    print({k:receipt[k] for k in ('status','scope','tests','failures','errors','skipped')})
    if not passed:
        print(stream.getvalue()); raise SystemExit(1)


if __name__=='__main__': main()

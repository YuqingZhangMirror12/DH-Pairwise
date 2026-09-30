from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import contracts as direct_contracts
from . import entry
from .contracts import inventory,sha,save


class EntryTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'source';source.mkdir();(source/'bound.py').write_text('BOUND = True\n')
        common=root/'common';common.mkdir();(common/'common.py').write_text('COMMON = True\n')
        real=root/'roles.json';case=root/'cases.json'
        save(real,{'synthetic_roles':True});save(case,{'synthetic_cases':True})
        args=SimpleNamespace(root=root,common_source=common,real_plan=real,case_plan=case,preparation=root/'receipt.json')
        r=dict(schema='binary-evaluation-preparation/1',status='cpu_preparation_passed',errors=0,failures=0,
            skipped=0,tests=2,verified_variants=['patch','stats'],source_files_unchanged=True,
            real_inference_performed=False,real_checkpoints_opened=False,gpu_preflight=False,formal_training_started=False,
            adapter_python_sha256=inventory(Path(entry.__file__).parent),common_python_sha256=inventory(common),
            training_source_sha256=inventory(source,True),real_plan_sha256=sha(real),fixed_case_plan_sha256=sha(case),
            results=[dict(variant=v,status='passed',returncode=0,tests=1,errors=0,failures=0,skipped=0)
                     for v in ('patch','stats')])
        save(args.preparation,r)
        return args,r

    def test_complete_receipt_and_each_binding(self):
        with tempfile.TemporaryDirectory() as t:
            args,r=self.fixture(Path(t))
            with patch.object(direct_contracts,'PLAN_SHA',sha(args.real_plan)):
                self.assertEqual(entry.validate_preparation(args),r)
                for field in ('adapter_python_sha256','common_python_sha256','training_source_sha256',
                              'real_plan_sha256','fixed_case_plan_sha256'):
                    bad=deepcopy(r);bad[field]={'wrong.py':'wrong'} if isinstance(r[field],dict) else 'wrong'
                    save(args.preparation,bad)
                    with self.assertRaises(ValueError):entry.validate_preparation(args)
                save(args.preparation,r)
                (args.root/'source'/'bound.py').write_text('BOUND = False\n')
                with self.assertRaises(ValueError):entry.validate_preparation(args)

    def test_refuses_missing_tests_skips_or_training_claims(self):
        with tempfile.TemporaryDirectory() as t:
            args,r=self.fixture(Path(t))
            with patch.object(direct_contracts,'PLAN_SHA',sha(args.real_plan)):
                for field,value in [('tests',0),('skipped',1),('errors',1),('failures',1),('source_files_unchanged',False),
                                    ('real_inference_performed',True),('real_checkpoints_opened',True),
                                    ('gpu_preflight',True),('formal_training_started',True),('verified_variants',['patch']),
                                    ('results',r['results'][:1]),('training_source_sha256',{})]:
                    bad=deepcopy(r);bad[field]=value;save(args.preparation,bad)
                    with self.assertRaises(ValueError,msg=field):entry.validate_preparation(args)
                for field,value in [('tests',0),('status','failed'),('returncode',1),('skipped',1)]:
                    bad=deepcopy(r);bad['results'][1][field]=value;save(args.preparation,bad)
                    with self.assertRaises(ValueError,msg=field):entry.validate_preparation(args)

    def test_canonical_real_plan_required_even_if_receipt_resealed(self):
        with tempfile.TemporaryDirectory() as t:
            args,r=self.fixture(Path(t))
            with self.assertRaises(ValueError):entry.validate_preparation(args)

    def test_bootstrap_rejects_already_imported_experiment_tree(self):
        # The worker has actually imported the immutable binary training tree.
        with self.assertRaisesRegex(ValueError,'fresh bound-source'):
            entry.bootstrap('unused','unused')

    def test_bootstrap_path_and_distinct_module_aliases(self):
        with tempfile.TemporaryDirectory() as t:
            source=Path(t).resolve();common=source/'common'
            train=SimpleNamespace(__file__=str(source/entry.RELATIVE/'train.py'))
            helper=SimpleNamespace()
            fake_sys=SimpleNamespace(modules={},path=[])
            with patch.object(entry,'sys',fake_sys),patch.object(entry.importlib,'import_module',side_effect=[train,helper]):
                self.assertEqual(entry.bootstrap(source,common),(train,helper))
                self.assertEqual(fake_sys.path[0],str(source))
                self.assertEqual(fake_sys.modules['consensus_binary_eval_common'].__path__,[str(common)])
                self.assertEqual(fake_sys.modules['consensus_binary_eval_adapter'].__path__,[str(Path(entry.__file__).parent.resolve())])
            fake_sys=SimpleNamespace(modules={},path=[])
            with patch.object(entry,'sys',fake_sys),patch.object(entry.importlib,'import_module',return_value=SimpleNamespace(__file__='/wrong/train.py')):
                with self.assertRaisesRegex(ValueError,'wrong training source'):entry.bootstrap(source,common)

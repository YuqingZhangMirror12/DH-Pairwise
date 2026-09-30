from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from . import entry
from .contracts import save


class EntryTests(unittest.TestCase):
    def fixture(self,root):
        for name in ('source','common','binary'):
            d=root/name;d.mkdir();(d/'synthetic.py').write_text('SYNTHETIC=True\n')
        roles=root/'roles.json';save(roles,dict(synthetic_roles=True))
        cases=root/'cases.json';save(cases,dict(synthetic_cases=True))
        a=SimpleNamespace(root=root,preparation=root/'preparation.json',common_source=root/'common',binary_source=root/'binary',real_plan=roles,case_plan=cases)
        r=dict(schema='aggressive-binary-evaluation-preparation/1',status='cpu_preparation_passed',verified_variants=['patch'],
            tests=1,errors=0,failures=0,skipped=0,source_files_unchanged=True,real_inference_performed=False,
            real_checkpoints_opened=False,gpu_preflight=False,formal_training_started=False,
            results=[dict(variant='patch',status='passed',returncode=0,tests=1,errors=0,failures=0,skipped=0)],
            adapter_python_sha256=entry.inventory(Path(entry.__file__).parent),binary_python_sha256=entry.inventory(a.binary_source),
            common_python_sha256=entry.inventory(a.common_source),training_source_sha256=entry.inventory(root/'source',True),
            real_plan_sha256=entry.sha(roles),fixed_case_plan_sha256=entry.sha(cases))
        save(a.preparation,r);return a,r

    def test_cpu_adapter_receipt_binds_all_sources_and_actual_tests(self):
        with tempfile.TemporaryDirectory() as t:
            a,r=self.fixture(Path(t))
            with patch.object(entry,'PLAN_SHA',entry.sha(a.real_plan)):
                self.assertEqual(entry.validate_preparation(a),r)
                for k,v in [('tests',0),('verified_variants',['stats']),('source_files_unchanged',False),('real_checkpoints_opened',True),('results',[])]:
                    bad=deepcopy(r);bad[k]=v;save(a.preparation,bad)
                    with self.assertRaises(ValueError):entry.validate_preparation(a)

    def test_changed_source_or_plan_rejected(self):
        for folder in ('source','binary','common'):
            with tempfile.TemporaryDirectory() as t:
                a,r=self.fixture(Path(t));(Path(t)/folder/'synthetic.py').write_text('CHANGED=True\n')
                with patch.object(entry,'PLAN_SHA',entry.sha(a.real_plan)):
                    with self.assertRaises(ValueError):entry.validate_preparation(a)

    def test_no_bootstrap_over_another_loaded_training_source(self):
        with self.assertRaisesRegex(ValueError,'fresh'):entry.bootstrap('unused','unused','unused')


if __name__=='__main__':unittest.main()

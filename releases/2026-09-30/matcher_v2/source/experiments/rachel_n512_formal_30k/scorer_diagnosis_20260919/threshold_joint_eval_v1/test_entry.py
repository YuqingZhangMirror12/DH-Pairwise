from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from . import entry
from .contracts import save


class EntryTests(unittest.TestCase):
    def fixture(self,root):
        args=SimpleNamespace(root=root/'formal',preparation=root/'preparation.json',
            common_source=root/'common',joint_source=root/'joint',real_plan=root/'plan.json',model_kind='joint',
            operation='evaluate',case_plan=root/'cases.json')
        package=args.root/'source'/entry.RELATIVE;package.mkdir(parents=True)
        helper=args.joint_source/entry.RELATIVE;helper.mkdir(parents=True)
        args.common_source.mkdir()
        (package/'train.py').write_text('# CPU fixture only\n')
        (helper/'real_development.py').write_text('# helper fixture only\n')
        (args.common_source/'frozen.py').write_text('# common fixture only\n')
        save(args.real_plan,{'fixture':True})
        save(args.case_plan,{'fixture_cases':True})
        receipt=dict(status='cpu_preparation_passed',schema='threshold-joint-evaluation-preparation/1',
            errors=0,failures=0,both_implementations_import_verified=True,real_inference_performed=False,
            adapter_python_sha256=entry.inventory(Path(entry.__file__).parent),
            common_python_sha256=entry.inventory(args.common_source),
            implementations={'joint':entry.inventory(package),'frozen':{}},
            real_development_helper_sha256=entry.digest(helper/'real_development.py'),
            real_plan_sha256=entry.digest(args.real_plan),fixed_case_plan_sha256=entry.digest(args.case_plan))
        save(args.preparation,receipt)
        return args

    def test_bound_preparation_passes_and_wrong_model_source_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.fixture(Path(directory));entry.validate_preparation(args)
            args.model_kind='frozen'
            with self.assertRaisesRegex(ValueError,'implementation'):entry.validate_preparation(args)

    def test_changed_common_source_plan_or_helper_rejected(self):
        for part in ('common','plan','helper','cases'):
            with tempfile.TemporaryDirectory() as directory:
                args=self.fixture(Path(directory))
                path={'common':args.common_source/'frozen.py','plan':args.real_plan,'cases':args.case_plan,
                      'helper':args.joint_source/entry.RELATIVE/'real_development.py'}[part]
                path.write_text('changed')
                with self.assertRaises(ValueError):entry.validate_preparation(args)

    def test_fresh_process_required_to_prevent_cross_source_import(self):
        with self.assertRaisesRegex(ValueError,'fresh process'):entry.bootstrap('unopened','unopened','unopened')


if __name__=='__main__':unittest.main()

import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import recover as r

class Tests(unittest.TestCase):
    def test_exact_one_line_delta(self):
        corrected=Path(r.__file__).with_name('audit.py').read_text();old=corrected.replace(r.NEW,r.OLD)
        r.verify_text(old,corrected)
        with self.assertRaises(ValueError):r.verify_text(old,corrected+'\n# accidental second change\n')
    def test_wrong_original_fails(self):
        with self.assertRaises(ValueError):r.verify_text('wrong','wrong')
    def test_real_source_delta_when_available(self):
        local=Path(r.__file__).parent.parent/'aggressive_data_full_v17/audit.py'
        original=local if local.exists() else r.SOURCE/r.REL/'audit.py'
        r.verify_text(original.read_text(),Path(r.__file__).with_name('audit.py').read_text())
    def condition(self,value):
        source=Path(r.__file__).with_name('audit.py').read_text();tree=ast.parse(source)
        nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
            and n.func.id=='need' and len(n.args)==2 and isinstance(n.args[1],ast.Constant)
            and n.args[1].value=='final PRIMARY footprint peak5–35px including weak']
        self.assertEqual(len(nodes),1)
        return eval(compile(ast.Expression(nodes[0].args[0]),'<actual-audit-condition>','eval'),{'max':max,'active_gaps':[value]})
    def test_observed_roundoff_accepted(self):self.assertTrue(self.condition(4.999999999999991))
    def test_genuine_below_five_rejected(self):
        for value in (4.0,4.999,4.999999):self.assertFalse(self.condition(value))
    def test_nominal_range(self):
        for value in (5.,25.,35.):self.assertTrue(self.condition(value))
    def test_upper_bound_unchanged(self):
        self.assertTrue(self.condition(35+1e-5));self.assertFalse(self.condition(35.001))
    def test_nonfinite_rejected(self):
        for v in (float('nan'),float('inf'),-float('inf')):self.assertFalse(self.condition(v))
    def test_preservation_detects_change(self):
        with tempfile.TemporaryDirectory() as t,patch.object(r,'DATA',Path(t)):
            p=Path(t)/'receipt';p.write_text('old');d={'receipt':r.sha(p)};r.check_preserved(d)
            p.write_text('changed')
            with self.assertRaises(ValueError):r.check_preserved(d)
    def test_save_atomic(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'receipt.json';r.save(p,{'status':'passed'});self.assertEqual(r.read(p),{'status':'passed'})
            self.assertEqual(len(list(Path(t).iterdir())),1)
    def test_preparation_rejects_unbound_tests(self):
        with tempfile.TemporaryDirectory() as t,patch.object(r,'ROOT',Path(t)):
            r.save(Path(t)/'cpu_tests_remote.json',dict(status='passed',tests=10,errors=0,failures=0,skipped=0,source_sha256={}))
            with self.assertRaisesRegex(ValueError,'source changed'):r.validate_cpu()

if __name__=='__main__':unittest.main()

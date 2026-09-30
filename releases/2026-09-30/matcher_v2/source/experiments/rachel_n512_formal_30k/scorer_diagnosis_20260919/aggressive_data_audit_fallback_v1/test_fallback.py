from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import fallback_group as f
import recover as r


def group():
    return dict(status='committed',slot=183,task={'slot':183},actual_positive_slot=183,
                positive_replaced=False,v14_fallback=False,
                records=[dict(id='positive',label=True,baseline_slot=183,baseline_ordinal=0),
                         dict(id='negative',label=False,baseline_slot=183,baseline_ordinal=1)])


class Tests(unittest.TestCase):
    def test_only_declared_gap_failure_routes(self):
        self.assertTrue(f.eligible(AssertionError(f.GAP_ERROR),group()))
        self.assertFalse(f.eligible(ValueError(f.GAP_ERROR),group()))
        self.assertFalse(f.eligible(AssertionError('label changed'),group()))

    def test_existing_v14_failure_not_hidden(self):
        g=group();g['v14_fallback']=True
        self.assertFalse(f.eligible(AssertionError(f.GAP_ERROR),g))

    def test_original_pair_checked(self):
        f.check_pair(group())
        g=group();g['actual_positive_slot']=184
        with self.assertRaises(ValueError):f.check_pair(g)

    def test_negative_identity_checked(self):
        g=group();g['records'][1]['baseline_slot']=184
        with self.assertRaises(ValueError):f.check_pair(g)

    def test_label_balance_checked(self):
        g=group();g['records'][1]['label']=True
        with self.assertRaises(ValueError):f.check_pair(g)

    def fixtures(self,root):
        g=group();split=root/'test'
        for i,row in enumerate(g['records']):
            for key,folder in [('sample_path','samples'),('proof_path','proof'),('target_metadata','targets')]:
                path=split/folder/f'00183_{i}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(('old '+key+str(i)).encode());row[key]=str(path)
            row['latent_seam_artifact']=None
        path=split/'groups/00183.json';f.save(path,g)
        return g,path

    def test_archive_all_original_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);g,p=self.fixtures(root);old=f.sha(p);history=root/'history'
            manifest=f.archive(p,history)
            self.assertEqual(len(manifest),7);self.assertEqual(f.sha(p),old)
            for rel,digest in manifest.items():self.assertEqual(f.sha(history/rel),digest)

    def test_no_overwrite_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);g,p=self.fixtures(root);f.archive(p,root/'history')
            with self.assertRaises(FileExistsError):f.archive(p,root/'history')

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);g,p=self.fixtures(root);g['records'][0]['sample_path']=str(root/'outside.npz')
            with self.assertRaises(ValueError):f.artifact_paths(g,p.parent.parent)

    def test_promote_preserves_archive_and_rewrites_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);g,p=self.fixtures(root);history=root/'history';manifest=f.archive(p,history)
            staged,sp=self.fixtures(root/'staged');staged['v14_fallback']=True
            for path in f.artifact_paths(staged,sp.parent.parent):path.write_bytes(b'v14 original fixture')
            result=f.promote(p,staged,sp.parent.parent,history)
            self.assertTrue(f.read(p)['v14_fallback'])
            for path in f.artifact_paths(result,p.parent.parent):self.assertEqual(path.read_bytes(),b'v14 original fixture')
            for rel,digest in manifest.items():self.assertEqual(f.sha(history/rel),digest)

    def test_changed_group_blocks_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);g,p=self.fixtures(root);history=root/'history';f.archive(p,history)
            staged,sp=self.fixtures(root/'staged');f.save(p,{'unexpected':'change'})
            with self.assertRaisesRegex(ValueError,'group changed'):f.promote(p,staged,sp.parent.parent,history)

    def test_cached_audit_binds_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);g,p=self.fixtures(root);out=root/'test/audits/00183.json'
            f.save(out,dict(status='passed',group_sha256=f.sha(p)))
            source=SimpleNamespace(STATE={'split':'test'})
            with patch.object(r,'DATA',root),patch.object(r.importlib,'import_module',return_value=source):
                self.assertTrue(r.audited_group({'slot':183})['cached'])
                f.save(p,dict(changed=True))
                with self.assertRaisesRegex(ValueError,'resume mismatch'):r.audited_group({'slot':183})

    def test_new_audit_records_two_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);g,p=self.fixtures(root);source=SimpleNamespace(STATE={'split':'test','root':'baseline'})
            auditor=SimpleNamespace(audit_record=lambda row,base:dict(id=row['id'],status='passed'))
            with patch.object(r,'DATA',root),patch.object(r.importlib,'import_module',return_value=source),patch.object(r,'audit_module',return_value=auditor):
                r.audited_group({'slot':183})
            self.assertEqual(len(f.read(root/'test/audits/00183.json')['rows']),2)

    def test_failed_audit_not_marked_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);g,p=self.fixtures(root);source=SimpleNamespace(STATE={'split':'test','root':'baseline'})
            def fail(*args):raise AssertionError(f.GAP_ERROR)
            with patch.object(r,'DATA',root),patch.object(r.importlib,'import_module',return_value=source),patch.object(r,'audit_module',return_value=SimpleNamespace(audit_record=fail)):
                with self.assertRaises(AssertionError):r.audited_group({'slot':183})
            self.assertFalse((root/'test/audits/00183.json').exists())

    def test_preservation_rejects_unrelated_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';data.mkdir();p=data/'old.json';f.save(p,{'old':1})
            f.save(root/'preexisting.json',{'old.json':f.sha(p)})
            with patch.object(r,'DATA',data),patch.object(r,'ROOT',root):
                self.assertEqual(r.check_preserved(),{})
                f.save(p,{'old':2})
                with self.assertRaisesRegex(ValueError,'unrelated'):r.check_preserved()

    def test_cpu_gate_requires_bound_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f.save(root/'cpu_tests_remote.json',dict(status='passed',tests=15,errors=0,failures=0,skipped=0,source_sha256={}))
            with patch.object(r,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'not bound'):r.validate_tests()


if __name__=='__main__':unittest.main()

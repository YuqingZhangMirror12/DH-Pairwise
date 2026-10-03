import json
from pathlib import Path
import tempfile
import unittest

from .heldout_augment import digest
from .heldout_extend_run import admit_commit,task_key


class ImportedCommitTests(unittest.TestCase):
    def test_import_keeps_original_receipt_and_artifacts_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            def write(name,value):
                path=root/name
                path.write_text(json.dumps(value,sort_keys=True))
                return dict(path=str(path),sha256=digest(path))
            old_plan=write("old_plan.json",{"plan":"old"})
            old_pilot=write("pilot.json",{"status":"shortfall"})
            task={"slot":7,"role":"select"}
            records=[]
            for ordinal in (0,1):
                row={"pair_id":str(ordinal)}
                for pathkey,hashkey in (("sample_path","sample_sha256"),("proof_path","proof_sha256"),
                    ("target_metadata","target_metadata_sha256"),("unaugmented_sample_path","unaugmented_sample_sha256"),
                    ("baseline_sample_path","baseline_sample_sha256"),("baseline_group_path","baseline_group_sha256")):
                    ref=write(str(ordinal)+pathkey,{"payload":pathkey})
                    row.update({pathkey:ref["path"],hashkey:ref["sha256"]})
                records.append(row)
            old=write("old_commit.json",dict(status="committed",task=task,plan_sha256=old_plan["sha256"],
                      records=records,audit_rows=[{"passed":True},{"passed":True}]))
            imported=dict(commit=old,old_plan=old_plan,old_pilot_complete=old_pilot)
            wrapper=root/"new_commit.json"
            result=admit_commit(imported,task,"new-plan-sha",wrapper)
            self.assertEqual(result["records"],records)
            self.assertEqual(result["imported_from"],imported)
            self.assertEqual(digest(old["path"]),old["sha256"])
            self.assertNotEqual(result["plan_sha256"],old_plan["sha256"])
            self.assertEqual(admit_commit(imported,task,"new-plan-sha",wrapper),result)
            with self.assertRaises(ValueError):admit_commit(imported,dict(task,slot=8),"new-plan-sha",wrapper)
            with self.assertRaises(ValueError):admit_commit(imported,task,"other-new-plan",wrapper)
            with self.assertRaises(ValueError):admit_commit({"commit":old},task,"new-plan-sha",wrapper)
            Path(records[0]["proof_path"]).write_text("changed pixels/proof")
            with self.assertRaises(ValueError):admit_commit(imported,task,"new-plan-sha",wrapper)

    def test_task_identity_does_not_depend_on_json_key_order(self):
        self.assertEqual(task_key({"a":1,"b":2}),task_key({"b":2,"a":1}))
        self.assertNotEqual(task_key({"slot":7}),task_key({"slot":8}))


if __name__=="__main__":unittest.main()

import copy
import json
import tempfile
import unittest
from pathlib import Path
from .heldout_run import expected_baseline_exhaustion, validate_tasks, validate_loaded_inventory


class HeldoutRunTests(unittest.TestCase):
    def test_only_exact_original_geometry_exhaustion_is_recoverable(self):
        task=dict(slot=7,recipe="partial",base_pair_ids=["p","n"])
        value=dict(slot=7,recipe="partial",partial=True,negative="n",failures={"too little seam":1024},
                   damage_attempts=1024,source_draws=1024,static_length_sources_excluded=0,
                   damage_attempt_budget=1024,source_pool_exhausted=False)
        self.assertTrue(expected_baseline_exhaustion(RuntimeError(json.dumps(value)),task))
        for error in (RuntimeError("CUDA out of memory"),RuntimeError("killed"),
                      RuntimeError(json.dumps(dict(value,slot=9))),RuntimeError(json.dumps(dict(value,extra=True)))):
            self.assertFalse(expected_baseline_exhaustion(error,task))

    def test_missing_cell_or_reserve_never_reports_complete(self):
        tasks=[]
        for role in ("cal","select"):
            for gen in ("Gen2","Gen3","Gen4","Gen5"):
                for stage in ("v17_filtered","v17.5","v18"):
                    for j in range(120):
                        for reserve in range(3):
                            tasks.append(dict(role=role,generator=gen,stage=stage,quota_slot=j,
                                reserve_index=reserve,slot=reserve*120+j,master_seed=1,
                                recipe="clean",size_class="smaller",mode="one",k=1,trim_target=.3))
        plan=dict(tasks=tasks,desired_pairs_per_role=3200,desired_pairs_per_curriculum_stage=960,
                  desired_strict_pairs_per_role=320)
        validate_tasks(plan)
        for changed in (dict(plan,tasks=tasks[:-1]),dict(plan,tasks=[]),dict(plan,desired_pairs_per_role=3198)):
            with self.assertRaises(ValueError):validate_tasks(changed)

    def test_loaded_source_must_match_preregistered_path_and_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            inventory=dict(root=directory,files={"native.py":"abc"})
            validate_loaded_inventory({str(root/"native.py"):"abc"},inventory)
            with self.assertRaises(ValueError):
                validate_loaded_inventory({str(root/"native.py"):"changed"},inventory)
            with self.assertRaises(ValueError):
                validate_loaded_inventory({str(root.parent/"outside.py"):"abc"},inventory)


if __name__=="__main__":unittest.main()

import copy
import unittest
from types import SimpleNamespace
import numpy as np

from .heldout_augment import rng_key, validate_task, tensor_identity


class HeldoutDomainTests(unittest.TestCase):
    def setUp(self):
        self.task = dict(role="select", stage="v18", generator="Gen5", slot=0,
                         master_seed=26100291, base_pair_ids=["positive", "negative"],
                         recipe="gaps", k=1, size_class="smaller", mode="both", trim_target=.30)

    def test_domain_is_reproducible(self):
        self.assertEqual(rng_key(self.task, 0), rng_key(copy.deepcopy(self.task), 0))

    def test_all_roles_and_provenance_separate_rng(self):
        reference = rng_key(self.task, 0)
        for key, value in (("role", "cal"), ("stage", "v17.5"), ("generator", "Gen4"),
                           ("slot", 1), ("master_seed", 26100292),
                           ("base_pair_ids", ["positive2", "negative"])):
            with self.subTest(key=key):
                task = dict(self.task, **{key: value})
                self.assertNotEqual(reference, rng_key(task, 0))
        self.assertNotEqual(reference, rng_key(self.task, 1))

    def test_forbids_training_test_and_out_of_recipe_values(self):
        for key, value in (("role", "train"), ("role", "test"), ("stage", "v14"),
                           ("generator", "Gen6"), ("size_class", "random"), ("mode", "middle"),
                           ("k", 5), ("trim_target", .5), ("slot", -1), ("master_seed", -1)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_task(dict(self.task, **{key:value}))

    def test_tensor_identity_ignores_names_but_not_pixels_or_labels(self):
        sample = SimpleNamespace(pair_id="first", label=np.bool_(True),
            translation_valid=np.bool_(True), translation_a_to_b_rc=np.zeros(2,np.float32))
        for side in "ab":
            for stem in ("mask_", "coarse_mask_", "points_rc_", "contour_valid_", "target_"):
                setattr(sample,stem+side,np.ones((2,2),np.float32))
        changed=copy.deepcopy(sample)
        changed.pair_id="different name"
        self.assertEqual(tensor_identity(sample),tensor_identity(changed))
        changed.label=np.bool_(False)
        self.assertEqual(tensor_identity(sample),tensor_identity(changed))
        self.assertNotEqual(tensor_identity(sample,include_supervision=True),
                            tensor_identity(changed,include_supervision=True))
        changed.mask_a[0,0]=0
        self.assertNotEqual(tensor_identity(sample),tensor_identity(changed))


if __name__ == "__main__":
    unittest.main()

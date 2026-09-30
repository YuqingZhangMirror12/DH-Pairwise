import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.rachel_n512_formal_30k.score_design_checkpoint_io import load_owned_epoch_checkpoint


class OwnedEpochTests(unittest.TestCase):
    def fixture(self, root):
        path = root / "epoch_013.pt"
        torch.save(dict(epoch=13, model_state_dict={"weight": torch.ones(2)},
                        optimizer_state_dict={}, rng_state={"numpy": np.random.get_state()}), path)
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_full_numpy_rng_snapshot_can_be_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, digest = self.fixture(root)
            value = load_owned_epoch_checkpoint(root, path, 13, digest)
            self.assertEqual(value["epoch"], 13)
            self.assertIsInstance(value["rng_state"]["numpy"][1], np.ndarray)
            self.assertTrue(torch.equal(value["model_state_dict"]["weight"], torch.ones(2)))

    def test_path_epoch_and_digest_fail_before_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, digest = self.fixture(root)
            with patch("torch.load") as loader:
                for arguments in ((root, path, 12, digest), (root, path, 13, "f" * 64)):
                    with self.assertRaises(ValueError):
                        load_owned_epoch_checkpoint(*arguments)
                other = root / "another_run"
                other.mkdir()
                with self.assertRaises(ValueError):
                    load_owned_epoch_checkpoint(other, path, 13, digest)
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()

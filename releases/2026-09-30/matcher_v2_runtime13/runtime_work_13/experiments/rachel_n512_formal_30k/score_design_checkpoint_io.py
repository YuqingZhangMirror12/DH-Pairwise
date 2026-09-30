"""Load known experiment-owned epoch snapshots that include optimizer/RNG.

These are trusted artifacts created by this task, not downloaded model files.
The generic sealed-test weights-only loader remains unchanged. Its restricted
unpickler cannot load the NumPy RNG state required for faithful resumption.
"""
from collections.abc import Mapping
import hashlib
from pathlib import Path

import torch


def load_owned_epoch_checkpoint(training_root, checkpoint_path, selected_epoch, expected_sha256):
    if type(selected_epoch) is not int or not 1 <= selected_epoch <= 50:
        raise ValueError("invalid experiment epoch")
    root = Path(training_root).resolve(strict=True)
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=True)
    if path.parent != root or path.name != "epoch_%03d.pt" % selected_epoch:
        raise ValueError("only the exact named epoch inside this training run can be loaded")
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ValueError("frozen epoch SHA256 required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("epoch snapshot differs from its frozen SHA256")
    # Only this task's SHA-bound epoch snapshot reaches the unrestricted loader.
    # Do not use this helper for third-party or user-uploaded pickle checkpoints.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("epoch") != selected_epoch:
        raise ValueError("epoch payload is not the selected training snapshot")
    return payload

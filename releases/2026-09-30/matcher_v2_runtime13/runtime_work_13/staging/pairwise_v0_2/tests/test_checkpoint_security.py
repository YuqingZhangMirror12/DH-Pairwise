import hashlib

import pytest
import torch

from staging.pairwise_v0_2.training import checkpoint
from staging.pairwise_v0_2.training.checkpoint import (
    canonical_tensor_tree_sha256,
    load_trusted_checkpoint,
    save_checkpoint,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model():
    torch.manual_seed(11)
    return torch.nn.Linear(3, 2)


def _save(tmp_path, **kwargs):
    path = tmp_path / "safe.pt"
    model = _model()
    receipt = save_checkpoint(
        path,
        model,
        config={"model": "fixture", "dropout": 0.25},
        epoch=1,
        **kwargs,
    )
    return path, model, receipt


def test_expected_file_sha_is_required_and_tampering_precedes_deserialization(
    tmp_path, monkeypatch
):
    path, _, receipt = _save(tmp_path)
    restored = _model()
    before = {key: value.clone() for key, value in restored.state_dict().items()}
    path.write_bytes(path.read_bytes() + b"tamper")
    called = {"load": False}
    original_load = checkpoint.torch.load

    def recording_load(file, map_location=None, weights_only=None):
        called["load"] = True
        return original_load(file, map_location=map_location, weights_only=weights_only)

    monkeypatch.setattr(checkpoint.torch, "load", recording_load)

    with pytest.raises(ValueError, match="SHA-256 does not match"):
        load_trusted_checkpoint(
            path,
            restored,
            expected_config={"model": "fixture", "dropout": 0.25},
            expected_file_sha256=receipt.file_sha256,
            trusted=True,
        )
    assert not called["load"]
    assert all(
        torch.equal(value, before[key]) for key, value in restored.state_dict().items()
    )

    with pytest.raises(TypeError, match="expected_file_sha256"):
        load_trusted_checkpoint(  # type: ignore[call-arg]
            path,
            restored,
            expected_config={"model": "fixture", "dropout": 0.25},
            trusted=True,
        )


def test_loader_always_requests_restricted_weights_only_mode(tmp_path, monkeypatch):
    path, _, receipt = _save(tmp_path, metrics={"loss": 0.5})
    restored = _model()
    observed = []
    original_load = checkpoint.torch.load

    def recording_load(file, map_location=None, weights_only=None):
        observed.append(weights_only)
        return original_load(file, map_location=map_location, weights_only=weights_only)

    monkeypatch.setattr(checkpoint.torch, "load", recording_load)
    metadata = load_trusted_checkpoint(
        path,
        restored,
        expected_config={"model": "fixture", "dropout": 0.25},
        expected_file_sha256=receipt.file_sha256,
        trusted=True,
    )

    assert observed == [True]
    assert metadata["file_sha256"] == receipt.file_sha256
    assert metadata["metrics"] == {"loss": 0.5}


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("config", {"api_key": "must-not-save"}, "sensitive key"),
        ("metrics", {"report": "/private/experiment.json"}, "absolute/file path"),
        (
            "provenance",
            {"source": "file:///Users/example/private.json"},
            "absolute/file path",
        ),
        (
            "provenance",
            {"source": "https://example.test/data?access_token=secret"},
            "sensitive URL query",
        ),
        (
            "metrics",
            {"source": "https://user:password@example.test/data"},
            "URL credentials",
        ),
    ],
)
def test_save_scans_config_metrics_and_provenance(tmp_path, field, value, error):
    kwargs = {
        "config": {"model": "fixture"},
        "metrics": {},
        "provenance": {},
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=error):
        save_checkpoint(
            tmp_path / "rejected.pt",
            _model(),
            config=kwargs["config"],
            epoch=0,
            metrics=kwargs["metrics"],
            provenance=kwargs["provenance"],
        )
    assert not (tmp_path / "rejected.pt").exists()


def test_loaded_metadata_is_rescanned_before_model_mutation(tmp_path):
    model = _model()
    config = {"model": "fixture"}
    payload = {
        "checkpoint_version": checkpoint.CHECKPOINT_VERSION,
        "config": config,
        "config_hash": checkpoint.canonical_config_hash(config),
        "epoch": 0,
        "metrics": {},
        "provenance": {
            "download": "https://example.test/archive?X-Amz-Signature=secret"
        },
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None,
    }
    path = tmp_path / "crafted.pt"
    torch.save(checkpoint._weights_only_safe_tree(payload), path, pickle_protocol=2)
    restored = _model()
    with torch.no_grad():
        for parameter in restored.parameters():
            parameter.fill_(7.0)
    before = {key: value.clone() for key, value in restored.state_dict().items()}

    with pytest.raises(ValueError, match="sensitive URL query"):
        load_trusted_checkpoint(
            path,
            restored,
            expected_config=config,
            expected_file_sha256=_sha256(path),
            trusted=True,
        )
    assert all(
        torch.equal(value, before[key]) for key, value in restored.state_dict().items()
    )


def test_invalid_expected_sha_shape_fails_closed(tmp_path):
    path, _, _ = _save(tmp_path)
    with pytest.raises(ValueError, match="64 lowercase"):
        load_trusted_checkpoint(
            path,
            _model(),
            expected_config={"model": "fixture", "dropout": 0.25},
            expected_file_sha256="ABC",
            trusted=True,
        )


def test_canonical_tree_is_order_stable_and_container_type_sensitive():
    tensor = torch.tensor([[1.0, -2.0], [3.5, 4.0]], dtype=torch.float32)
    left = {"z": (True, None), "a": [tensor, 7, "敦煌", 0.25]}
    right = {"a": [tensor.clone(), 7, "敦煌", 0.25], "z": (True, None)}
    assert canonical_tensor_tree_sha256(left) == canonical_tensor_tree_sha256(right)
    assert canonical_tensor_tree_sha256([1, 2]) != canonical_tensor_tree_sha256((1, 2))


def test_canonical_checkpoint_identity_is_independent_of_path_and_container_bytes(
    tmp_path,
):
    first_path = tmp_path / "first" / "checkpoint.pt"
    second_path = tmp_path / "second" / "renamed-container.pt"
    first = save_checkpoint(
        first_path,
        _model(),
        config={"model": "fixture", "dropout": 0.25},
        epoch=3,
        metrics={"loss": 0.5},
        provenance={"run": "reproduction"},
    )
    second = save_checkpoint(
        second_path,
        _model(),
        config={"dropout": 0.25, "model": "fixture"},
        epoch=3,
        metrics={"loss": 0.5},
        provenance={"run": "reproduction"},
    )
    assert first.model_state_sha256 == second.model_state_sha256
    assert first.canonical_content_sha256 == second.canonical_content_sha256
    # PyTorch's ZIP root includes the destination basename, so file identity
    # remains a separate transport/integrity property.
    assert first.file_sha256 != second.file_sha256


def test_canonical_checkpoint_binds_tensor_config_and_epoch(tmp_path):
    baseline = save_checkpoint(
        tmp_path / "baseline.pt",
        _model(),
        config={"model": "fixture"},
        epoch=1,
    )
    changed_model = _model()
    with torch.no_grad():
        changed_model.weight[0, 0].add_(1.0)
    tensor_change = save_checkpoint(
        tmp_path / "tensor.pt",
        changed_model,
        config={"model": "fixture"},
        epoch=1,
    )
    config_change = save_checkpoint(
        tmp_path / "config.pt",
        _model(),
        config={"model": "fixture-v2"},
        epoch=1,
    )
    epoch_change = save_checkpoint(
        tmp_path / "epoch.pt",
        _model(),
        config={"model": "fixture"},
        epoch=2,
    )
    assert tensor_change.model_state_sha256 != baseline.model_state_sha256
    assert tensor_change.canonical_content_sha256 != baseline.canonical_content_sha256
    assert config_change.canonical_content_sha256 != baseline.canonical_content_sha256
    assert epoch_change.canonical_content_sha256 != baseline.canonical_content_sha256


def test_optimizer_state_has_canonical_digest_and_round_trips(tmp_path):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones((2, 3))).sum().backward()
    optimizer.step()
    receipt = save_checkpoint(
        tmp_path / "optimizer.pt",
        model,
        config={"model": "fixture"},
        epoch=1,
        optimizer=optimizer,
    )
    assert receipt.optimizer_state_sha256 is not None
    restored_model = _model()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    metadata = load_trusted_checkpoint(
        tmp_path / "optimizer.pt",
        restored_model,
        expected_config={"model": "fixture"},
        expected_file_sha256=receipt.file_sha256,
        expected_canonical_content_sha256=receipt.canonical_content_sha256,
        optimizer=restored_optimizer,
        trusted=True,
    )
    assert metadata["optimizer_state_sha256"] == receipt.optimizer_state_sha256
    assert metadata["canonical_content_sha256"] == receipt.canonical_content_sha256


def test_repacked_tensor_tamper_fails_content_check_before_model_mutation(tmp_path):
    path, _, receipt = _save(tmp_path)
    with path.open("rb") as handle:
        encoded = torch.load(handle, map_location="cpu", weights_only=True)
    payload = checkpoint._restore_weights_only_safe_tree(encoded)
    payload["model_state_dict"]["weight"] = (
        payload["model_state_dict"]["weight"].clone() + 1.0
    )
    torch.save(checkpoint._weights_only_safe_tree(payload), path, pickle_protocol=2)

    restored = _model()
    with torch.no_grad():
        restored.weight.fill_(9.0)
    before = {key: value.clone() for key, value in restored.state_dict().items()}
    with pytest.raises(ValueError, match="model_state_sha256.*inconsistent"):
        load_trusted_checkpoint(
            path,
            restored,
            expected_config={"model": "fixture", "dropout": 0.25},
            expected_file_sha256=_sha256(path),
            expected_canonical_content_sha256=receipt.canonical_content_sha256,
            trusted=True,
        )
    assert all(
        torch.equal(value, before[key]) for key, value in restored.state_dict().items()
    )


@pytest.mark.parametrize(
    "value,error",
    [
        ({1: "not-a-string-key"}, "mapping keys"),
        ({"bad": object()}, "unsupported canonical tree type"),
        ({"bad": float("nan")}, "NaN or infinity"),
        ({"bad": torch.tensor([float("inf")])}, "NaN or infinity"),
        ({"bad": torch.empty((2, 2), device="meta")}, "meta device"),
        ({"bad": torch.eye(2).to_sparse()}, "dense strided"),
        (
            {"bad": torch.quantize_per_tensor(torch.ones(2), 0.1, 0, torch.quint8)},
            "quantized",
        ),
    ],
)
def test_canonical_tree_rejects_ambiguous_or_unsupported_values(value, error):
    with pytest.raises((TypeError, ValueError), match=error):
        canonical_tensor_tree_sha256(value)

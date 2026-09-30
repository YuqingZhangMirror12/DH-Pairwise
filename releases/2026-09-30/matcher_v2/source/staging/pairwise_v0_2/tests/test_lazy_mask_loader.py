import hashlib
import io
import tarfile
import zipfile

import numpy as np
import pytest
from PIL import Image

from staging.pairwise_v0_2.pairwise_data.lazy_mask_loader import (
    ArchiveSourceSpec,
    ArchiveVerificationReceipt,
    LazyMaskArchiveLoader,
    LazyMaskLoaderError,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ArchiveBinding,
    MaskMemberRef,
)


def _png(values):
    stream = io.BytesIO()
    Image.fromarray(np.asarray(values, dtype=np.uint8), mode="L").save(
        stream, format="PNG"
    )
    return stream.getvalue()


def _reference(binding, member, content_sha, threshold="grayscale_uint8_gt_127"):
    return MaskMemberRef(
        binding=binding,
        archive_member=member,
        fragment_id="fixture/group/fragment/0",
        dataset_id="fixture",
        canonical_group_id="fixture/group",
        component_id="fixture/component",
        split="train",
        threshold_rule=threshold,
        content_sha256=content_sha,
    )


def test_zip_loader_is_lazy_hash_checked_cached_and_read_only():
    payload = _png([[0, 255], [255, 0]])
    archive_stream = io.BytesIO()
    with zipfile.ZipFile(archive_stream, "w") as archive:
        archive.writestr("masks/0.png", payload)
    archive_bytes = archive_stream.getvalue()
    binding = ArchiveBinding(
        "local_asset://fixture_zip", "zip", hashlib.sha256(archive_bytes).hexdigest()
    )
    reference = _reference(binding, "masks/0.png", hashlib.sha256(payload).hexdigest())
    loader = LazyMaskArchiveLoader(
        {
            binding.logical_id: ArchiveSourceSpec(
                binding=binding,
                source=archive_bytes,
                sha256_verified=True,
            )
        }
    )

    assert loader.stats.archive_opens == 0
    first = loader(reference)
    second = loader(reference)

    assert loader.stats.archive_opens == 1
    assert loader.stats.archive_verifications == 1
    assert loader.stats.archive_verified_bytes == len(archive_bytes)
    assert loader.stats.decoded_masks == 1
    assert loader.stats.cache_hits == 1
    assert first is second
    assert first.dtype == np.bool_
    assert first.tolist() == [[False, True], [True, False]]
    assert not first.flags.writeable
    archive_provenance = loader.provenance()["registered_archives"][0]
    assert archive_provenance == {
        "logical_id": binding.logical_id,
        "format": "zip",
        "expected_sha256": binding.sha256,
        "observed_sha256": binding.sha256,
        "verification_mode": "observed_in_memory_bytes",
        "verified_byte_count": len(archive_bytes),
    }
    assert "sha256_verified" not in archive_provenance
    loader.close()
    with pytest.raises(LazyMaskLoaderError, match="closed"):
        loader(reference)


def test_tar_loader_reads_member_without_extraction():
    payload = _png([[0, 0], [255, 255]])
    archive_stream = io.BytesIO()
    with tarfile.open(fileobj=archive_stream, mode="w:gz") as archive:
        info = tarfile.TarInfo("masks/0.png")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    archive_bytes = archive_stream.getvalue()
    binding = ArchiveBinding(
        "local_asset://fixture_tar", "tar", hashlib.sha256(archive_bytes).hexdigest()
    )
    reference = _reference(binding, "masks/0.png", hashlib.sha256(payload).hexdigest())

    with LazyMaskArchiveLoader(
        {binding.logical_id: ArchiveSourceSpec(binding, archive_bytes, True)}
    ) as loader:
        mask = loader(reference)

    assert mask.tolist() == [[False, False], [True, True]]


def test_caller_verified_boolean_cannot_bypass_archive_hash_mismatch():
    payload = _png([[0, 255]])
    archive_stream = io.BytesIO()
    with zipfile.ZipFile(archive_stream, "w") as archive:
        archive.writestr("masks/0.png", payload)
    binding = ArchiveBinding("local_asset://fixture", "zip", "d" * 64)
    reference = _reference(binding, "masks/0.png", "e" * 64)
    loader = LazyMaskArchiveLoader(
        {
            binding.logical_id: ArchiveSourceSpec(
                binding, archive_stream.getvalue(), sha256_verified=True
            )
        }
    )

    with pytest.raises(LazyMaskLoaderError, match="archive SHA-256 mismatch"):
        loader(reference)
    assert loader.stats.archive_opens == 0
    assert loader.stats.decoded_masks == 0
    assert loader.stats.archive_hash_failures == 1


def test_historical_rule_rejects_multivalued_scalar_mask():
    payload = _png([[0, 1, 2]])
    archive_stream = io.BytesIO()
    with zipfile.ZipFile(archive_stream, "w") as archive:
        archive.writestr("masks/0.png", payload)
    archive_bytes = archive_stream.getvalue()
    binding = ArchiveBinding(
        "local_asset://historical",
        "zip",
        hashlib.sha256(archive_bytes).hexdigest(),
    )
    reference = _reference(
        binding,
        "masks/0.png",
        hashlib.sha256(payload).hexdigest(),
        threshold="binary_brighter_value",
    )
    loader = LazyMaskArchiveLoader(
        {binding.logical_id: ArchiveSourceSpec(binding, archive_bytes)}
    )

    with pytest.raises(LazyMaskLoaderError, match="more than two"):
        loader(reference)


def test_path_and_seekable_sources_are_observed_once(tmp_path):
    payload = _png([[0, 255]])
    archive_stream = io.BytesIO()
    with zipfile.ZipFile(archive_stream, "w") as archive:
        archive.writestr("masks/0.png", payload)
    archive_bytes = archive_stream.getvalue()
    archive_path = tmp_path / "fixture.zip"
    archive_path.write_bytes(archive_bytes)
    binding = ArchiveBinding(
        "local_asset://path_fixture",
        "zip",
        hashlib.sha256(archive_bytes).hexdigest(),
    )
    reference = _reference(binding, "masks/0.png", hashlib.sha256(payload).hexdigest())
    loader = LazyMaskArchiveLoader(
        {binding.logical_id: ArchiveSourceSpec(binding, archive_path)}
    )

    loader(reference)
    loader(reference)

    assert loader.stats.archive_verifications == 1
    provenance = loader.provenance()["registered_archives"][0]
    assert provenance["observed_sha256"] == binding.sha256
    assert provenance["verification_mode"] == "observed_open_file_stream"


def test_verify_all_sources_is_sorted_portable_and_does_not_decode_members():
    sources = {}
    expected = {}
    for suffix in ("z", "a"):
        payload = _png([[0, 255], [255, 0]])
        archive_stream = io.BytesIO()
        with zipfile.ZipFile(archive_stream, "w") as archive:
            archive.writestr("masks/{}.png".format(suffix), payload)
        archive_bytes = archive_stream.getvalue()
        binding = ArchiveBinding(
            "local_asset://preflight_{}".format(suffix),
            "zip",
            hashlib.sha256(archive_bytes).hexdigest(),
        )
        sources[binding.logical_id] = ArchiveSourceSpec(binding, archive_bytes)
        expected[binding.logical_id] = (binding, len(archive_bytes))

    loader = LazyMaskArchiveLoader(sources)
    receipts = loader.verify_all_sources()
    repeated = loader.verify_all_sources()

    assert receipts == repeated
    assert all(isinstance(item, ArchiveVerificationReceipt) for item in receipts)
    assert [item.logical_id for item in receipts] == sorted(expected)
    assert [item.to_dict() for item in receipts] == [
        {
            "logical_id": logical_id,
            "expected_sha256": expected[logical_id][0].sha256,
            "observed_sha256": expected[logical_id][0].sha256,
            "byte_count": expected[logical_id][1],
            "verification_mode": "observed_in_memory_bytes",
        }
        for logical_id in sorted(expected)
    ]
    assert loader.stats.archive_verifications == 2
    assert loader.stats.archive_opens == 0
    assert loader.stats.decoded_masks == 0
    assert loader.stats.requests == 0


def test_verify_all_sources_fails_before_any_archive_open_on_hash_mismatch():
    payload = _png([[0, 255]])
    archive_stream = io.BytesIO()
    with zipfile.ZipFile(archive_stream, "w") as archive:
        archive.writestr("masks/0.png", payload)
    binding = ArchiveBinding("local_asset://bad_preflight", "zip", "f" * 64)
    loader = LazyMaskArchiveLoader(
        {binding.logical_id: ArchiveSourceSpec(binding, archive_stream.getvalue())}
    )

    with pytest.raises(LazyMaskLoaderError, match="archive SHA-256 mismatch"):
        loader.verify_all_sources()
    assert loader.stats.archive_hash_failures == 1
    assert loader.stats.archive_opens == 0
    assert loader.stats.decoded_masks == 0


class _NonSeekableBinary:
    def __init__(self, payload):
        self._stream = io.BytesIO(payload)

    def readable(self):
        return True

    def seekable(self):
        return False

    def read(self, size=-1):
        return self._stream.read(size)


def test_nonseekable_archive_stream_fails_before_open_or_decode():
    payload = _png([[0, 255]])
    archive_stream = io.BytesIO()
    with zipfile.ZipFile(archive_stream, "w") as archive:
        archive.writestr("masks/0.png", payload)
    archive_bytes = archive_stream.getvalue()
    binding = ArchiveBinding(
        "local_asset://nonseekable_fixture",
        "zip",
        hashlib.sha256(archive_bytes).hexdigest(),
    )
    reference = _reference(binding, "masks/0.png", hashlib.sha256(payload).hexdigest())
    loader = LazyMaskArchiveLoader(
        {
            binding.logical_id: ArchiveSourceSpec(
                binding, _NonSeekableBinary(archive_bytes), True
            )
        }
    )

    with pytest.raises(LazyMaskLoaderError, match="seekable"):
        loader(reference)
    assert loader.stats.archive_opens == 0
    assert loader.stats.decoded_masks == 0


def test_cache_key_binds_member_hash_and_threshold_rule():
    payload = _png([[0, 1]])
    archive_stream = io.BytesIO()
    with zipfile.ZipFile(archive_stream, "w") as archive:
        archive.writestr("masks/0.png", payload)
    archive_bytes = archive_stream.getvalue()
    binding = ArchiveBinding(
        "local_asset://cache_semantics_fixture",
        "zip",
        hashlib.sha256(archive_bytes).hexdigest(),
    )
    content_sha = hashlib.sha256(payload).hexdigest()
    grayscale_ref = _reference(
        binding, "masks/0.png", content_sha, threshold="grayscale_uint8_gt_127"
    )
    brighter_ref = _reference(
        binding, "masks/0.png", content_sha, threshold="binary_brighter_value"
    )
    loader = LazyMaskArchiveLoader(
        {binding.logical_id: ArchiveSourceSpec(binding, archive_bytes)}
    )

    grayscale = loader(grayscale_ref)
    brighter = loader(brighter_ref)
    brighter_again = loader(brighter_ref)

    assert grayscale.tolist() == [[False, False]]
    assert brighter.tolist() == [[False, True]]
    assert brighter_again is brighter
    assert loader.stats.decoded_masks == 2
    assert loader.stats.cache_hits == 1
    assert loader.stats.archive_verifications == 1


def test_cache_hit_cannot_bypass_a_different_member_content_sha():
    payload = _png([[0, 255]])
    archive_stream = io.BytesIO()
    with zipfile.ZipFile(archive_stream, "w") as archive:
        archive.writestr("masks/0.png", payload)
    archive_bytes = archive_stream.getvalue()
    binding = ArchiveBinding(
        "local_asset://cache_member_hash_fixture",
        "zip",
        hashlib.sha256(archive_bytes).hexdigest(),
    )
    no_manifest_hash = _reference(binding, "masks/0.png", None)
    wrong_manifest_hash = _reference(binding, "masks/0.png", "a" * 64)
    loader = LazyMaskArchiveLoader(
        {binding.logical_id: ArchiveSourceSpec(binding, archive_bytes)}
    )

    loader(no_manifest_hash)
    with pytest.raises(LazyMaskLoaderError, match="content SHA-256 mismatch"):
        loader(wrong_manifest_hash)
    assert loader.stats.cache_hits == 0

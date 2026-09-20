"""Read-only formal memmap cache adapter; labels never enter model kwargs."""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import cache
from .model import CandidateSelection, DECODER_CONFIG

FULL_COUNTS = dict(cache.COUNTS)
VAL_SHA = "daa6ccdd7686e93ba91ddfb1452c987145c26898a1917d2ac7d3180e199a8af8"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CacheBatch:
    model_args: tuple
    model_kwargs: dict
    labels: torch.Tensor
    training_valid: torch.Tensor
    decision_valid: torch.Tensor
    pair_ids: tuple


class FormalCache:
    def __init__(self, root, split):
        if split not in FULL_COUNTS:
            raise ValueError("only TRAIN/VAL cache is reachable")
        self.root, self.split = Path(root).resolve(strict=True), split
        p = json.loads((self.root / "protocol.json").read_text())
        count = FULL_COUNTS[split]
        population = p.get("population", {})
        if (p.get("schema") != cache.SCHEMA or p.get("status") != "complete"
                or p.get("formal_training_eligible") is not True
                or p.get("split") != split or p.get("pair_count") != count
                or p.get("expected_full_count") != count or p.get("completed_pairs") != count
                or p.get("source_checkpoint_sha256") != cache.SOURCE_SHA
                or population.get("manifest_sha256") != (cache.TRAIN_SHA if split == "train" else VAL_SHA)
                or population.get("count") != count or population.get("split") != split
                or population.get("sampling") != "original512" or population.get("contour_cap") != 512
                or p.get("matcher_frozen") is not True or p.get("selector_gt_free") is not True
                or p.get("no_online_augmentation") is not True
                or p.get("precompute_device") != "cpu" or p.get("features_dtype") != "float32"
                or p.get("decoder") != asdict(DECODER_CONFIG)):
            raise ValueError("requires COMPLETE formal S7 M12 TRAIN24K/VAL3K cache; probes are forbidden")
        cfg = p.get("source_model_config", {})
        if (cfg.get("canvas_size") != 800 or cfg.get("contour_cap") != 512
                or cfg.get("feature_dim") != 96 or cfg.get("num_heads") != 4
                or cfg.get("window_sizes_px") != [7., 16., 32., 64.] or cfg.get("patch_size") != 16):
            raise ValueError("cache source configuration differs")
        if sha(self.root / "pairs.json") != p.get("pairs_sha256"):
            raise ValueError("pair metadata hash changed")
        self.records = json.loads((self.root / "pairs.json").read_text())
        if (not isinstance(self.records, list) or len(self.records) != count
                or any(not isinstance(r, dict) or r.get("ordinal") != i
                       or not isinstance(r.get("pair_id"), str) or not r["pair_id"]
                       or r.get("label") not in (0., 1.) for i, r in enumerate(self.records))
                or len({r["pair_id"] for r in self.records}) != count):
            raise ValueError("cache pair IDs/labels/order are malformed")
        self.arrays, files = {}, {}
        for name, (dtype, tail) in cache.ARRAYS.items():
            path = self.root / (name + ".npy")
            expected = dict(dtype=dtype, shape=[count, *tail])
            if p.get("arrays", {}).get(name) != expected:
                raise ValueError("cache array manifest mismatch: " + name)
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            if value.shape != (count, *tail) or value.dtype != np.dtype(dtype):
                raise ValueError("cache array header mismatch: " + name)
            info = path.stat()
            files[name] = dict(bytes=info.st_size, mtime_ns=info.st_mtime_ns,
                               shape=list(value.shape), dtype=str(value.dtype))
            self.arrays[name] = value
        labels = np.asarray(self.arrays["label"])
        if (not self.arrays["ready"].all() or not np.isin(labels, [0., 1.]).all()
                or not np.array_equal(labels, np.asarray([r["label"] for r in self.records]))
                or int(labels.sum()) != p.get("positive_count") or not 0 < labels.sum() < count):
            raise ValueError("cache readiness/labels disagree with committed protocol")
        self.binding = dict(root=str(self.root), split=split, pair_count=count,
            protocol_sha256=sha(self.root / "protocol.json"), pairs_sha256=p["pairs_sha256"],
            source_checkpoint_sha256=p["source_checkpoint_sha256"], population=population,
            cache_implementation_sha256=p.get("implementation_sha256"), arrays=files,
            precompute_device="cpu", features_dtype="float32",
            array_integrity="readonly memmap; header/size/mtime bound; writer supplied no full-array SHA")

    def __len__(self):
        return len(self.records)

    def batch(self, indices, device="cpu"):
        ids = np.asarray(indices, dtype=np.int64)
        if ids.ndim != 1 or not len(ids) or (ids < 0).any() or (ids >= len(self)).any():
            raise ValueError("invalid cache batch indices")
        def value(name):
            # Fancy indexing returns writable detached memory, never mutates mmap.
            return torch.from_numpy(np.array(self.arrays[name][ids], copy=True)).to(device)
        selection = CandidateSelection(
            value("mask_a"), value("mask_b"), value("layout_valid"), value("translation_a_to_b_rc"),
            value("candidate_indices"), value("candidate_valid"), value("candidate_inliers"),
            tuple("cached_decoder_reason_not_recorded" for _ in ids))
        return CacheBatch(
            model_args=(value("features_a"), value("features_b"), value("valid_a"), value("valid_b"), selection),
            model_kwargs=dict(candidate_weights=value("candidate_weights"), points_a_rc=value("points_a"),
                              points_b_rc=value("points_b")),
            labels=value("label"), training_valid=value("training_valid"), decision_valid=value("decision_valid"),
            pair_ids=tuple(self.records[int(i)]["pair_id"] for i in ids))


class CandidateStageCache:
    """Adds precomputed stage geometry only. Labels stay in CacheBatch.targets."""
    def __init__(self,source,root,stage):
        from .stage_cache import StageCache, STAGES
        if stage not in STAGES:
            raise ValueError("unknown stage edge arm")
        self.source,self.stage=source,stage
        self.groups=StageCache(root,source)
        self.root,self.split,self.records=source.root,source.split,source.records
        self.binding=dict(source=source.binding,stage=stage,stage_cache=self.groups.binding)

    def __len__(self): return len(self.source)

    def batch(self,indices,device="cpu"):
        original=self.source.batch(indices,device)
        return replace(original,model_kwargs=dict(original.model_kwargs,
            groups=self.groups.batch(indices,self.stage,device)))

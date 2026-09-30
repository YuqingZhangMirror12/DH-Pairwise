"""CPU-only frozen CNN features for an InfoNCE projection adaptation.

This is not a PairingNet reproduction: the encoder was trained by the Rachel
BCE objectives and is never trained here. A downstream experiment may train a
new InfoNCE projection using TRAIN only. Features are the final coarse.encoder
feature map's spatial mean, BEFORE the original BCE projection (64 dimensions
for the fixed candidate). No local matcher, Sinkhorn, or pose decoder runs.

Output schema ``rachel-coarse-retrieval-features/v1``:
  manifest.json: status, split, matcher_checkpoint_id, feature_dim, feature_file,
    pair_file, fragment_count, sample_count, precision, provenance.
  features.npz: embeddings float32[F,D], fragment_id str[F], source_id str[F].
  pairs.npz: pair_id str[P], index_a/index_b int64[P], label bool[P].
    REAL additionally includes strict_member bool[P] and case_id str[P].
Fragment indices use first endpoint occurrence; pair order is unchanged.

Each invocation exports exactly one split. TRAIN/VAL never open TEST/REAL.
TEST/REAL require a completed VAL selection receipt before their loader opens:
``status=complete, selected_on=val, matcher_checkpoint_id=<source SHA256>``.
Use a fresh output directory; existing output is never overwritten.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


SCHEMA = "rachel-coarse-retrieval-features/v1"
FIXED_CHECKPOINT_SHA = "350bf95413f828698f3396294a317569b013891d0906e2d4212d046fba059233"
DEFAULT_DATASET = "/root/autodl-tmp/dataset_rachel_pairwise_n512_v1"
DEFAULT_PREPARED = "/root/autodl-tmp/rachel_layout_v2_20260906_001/real_preparation/prepared"


@dataclass
class FragmentPopulation:
    """Model-facing mask accessor plus lightweight, ordered pair identities."""

    fragment_ids: list
    source_ids: list
    pairs: dict
    load_mask: Callable[[int], np.ndarray]
    provenance: dict


def _source_identity(fragment):
    """Use original image lineage, not one generator/group view of that image."""
    identities = [fragment[key] for key in ("split_unit_id", "lineage_id", "image_name")
                  if isinstance(fragment.get(key), str) and fragment[key]]
    if not identities or len(set(identities)) != 1:
        raise ValueError("fragment requires unambiguous source-image lineage")
    return identities[0]


def index_pair_records(records):
    """Index normalized endpoint records without examining any image/pose GT.

    Each record contains pair_id, label, fragment_a/b (ID), source_a/b. REAL
    records additionally contain strict_member and case_id. All positive pairs
    must share source identity; negative pairs may be same- or cross-source.
    """
    ids, sources, index, seen_pairs = [], [], {}, set()
    pair_ids, aa, bb, labels = [], [], [], []
    extras = None
    strict, cases = [], []
    for row in records:
        pair_id = row["pair_id"]
        if not isinstance(pair_id, str) or not pair_id or pair_id in seen_pairs:
            raise ValueError("pair IDs must be nonempty and unique")
        if not isinstance(row["label"], (bool, np.bool_)):
            raise ValueError("pair labels must be Boolean")
        seen_pairs.add(pair_id)
        endpoints = []
        for side in ("a", "b"):
            token, source = row["fragment_" + side], row["source_" + side]
            if not isinstance(token, str) or not token or not isinstance(source, str) or not source:
                raise ValueError("fragment and source IDs must be nonempty strings")
            if token not in index:
                index[token] = len(ids)
                ids.append(token)
                sources.append(source)
            elif sources[index[token]] != source:
                raise ValueError("one fragment has inconsistent source identity")
            endpoints.append(index[token])
        if endpoints[0] == endpoints[1]:
            raise ValueError("pair endpoints must differ")
        if row["label"] and sources[endpoints[0]] != sources[endpoints[1]]:
            raise ValueError("positive pair must share source identity")
        pair_ids.append(pair_id)
        aa.append(endpoints[0])
        bb.append(endpoints[1])
        labels.append(row["label"])
        real_extras = "strict_member" in row or "case_id" in row
        if extras is not None and extras != real_extras:
            raise ValueError("REAL metadata must be present for every pair or none")
        extras = real_extras
        if real_extras:
            if not isinstance(row.get("strict_member"), (bool, np.bool_)) or not row.get("case_id"):
                raise ValueError("REAL pair requires strict_member and case_id")
            strict.append(row["strict_member"])
            cases.append(row["case_id"])
    if not pair_ids:
        raise ValueError("empty pair population")
    pairs = dict(pair_id=np.asarray(pair_ids, dtype="U"), index_a=np.asarray(aa, np.int64),
                 index_b=np.asarray(bb, np.int64), label=np.asarray(labels, bool))
    if extras:
        pairs.update(strict_member=np.asarray(strict, bool), case_id=np.asarray(cases, dtype="U"))
    return ids, sources, pairs


def load_rachel_population(root, split):
    """Reuse selected-split loader and mask decoder without loading pose targets.

    RachelPairDataset initialization validates referenced release paths but
    never opens the target archives. Its model-mask loader is reused directly
    per unique fragment; __getitem__ (which reads positive pose GT) is unused.
    """
    from staging.pairwise_v0_2.pairwise_data import rachel_training_dataset as runtime
    dataset = runtime.RachelPairDataset(root, split)
    manifest = dataset.root / "pairs" / (split + ".jsonl")
    with manifest.open(encoding="utf-8") as stream:
        raw_rows = [json.loads(line) for line in stream if line.strip()]
    if len(raw_rows) != len(dataset._rows):
        raise ValueError("selected source rows differ from runtime loader")
    fragments, records = {}, []
    for raw, selected in zip(raw_rows, dataset._rows):
        if raw["pair_id"] != selected.pair_id:
            raise ValueError("runtime/source pair order differs")
        record = dict(pair_id=selected.pair_id, label=selected.label)
        for side in ("a", "b"):
            endpoint = getattr(selected, "fragment_" + side)
            raw_endpoint = raw["fragment_" + side]
            if raw_endpoint["fragment_token"] != endpoint.token:
                raise ValueError("runtime/source ordered endpoints differ")
            previous = fragments.setdefault(endpoint.token, endpoint)
            if previous != endpoint:
                raise ValueError("one fragment ID refers to different model inputs")
            record["fragment_" + side] = endpoint.token
            record["source_" + side] = _source_identity(raw_endpoint)
        records.append(record)
    ids, sources, pairs = index_pair_records(records)

    def load_mask(index):
        # Use the full mask then torch nearest resize, exactly as Rachel.forward;
        # the legacy loader's PIL-resized coarse_mask is deliberately not used.
        return runtime._load_mask(fragments[ids[index]].mask_path, dataset.config)[0]

    return FragmentPopulation(ids, sources, pairs, load_mask,
                              dict(dataset_root=str(dataset.root), pair_manifest=str(manifest),
                                   pair_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                                   source_identity="endpoint split_unit_id/source-image lineage",
                                   pose_target_archives_opened=False))


def load_real_population(prepared):
    """Reuse the existing prepared REAL model masks, never any translation GT."""
    from experiments.rachel_n512_formal_30k.run_real_contiguous_seam_ablation import load_prepared_cache
    metadata, arrays = load_prepared_cache(prepared)
    prepared_index = {token: i for i, token in enumerate(metadata["fragment_ids"])}

    def source_id(token):
        if "/fragment/" not in token:
            raise ValueError("REAL endpoint lacks existing case/fragment identity")
        return token.split("/fragment/", 1)[0]

    records = [dict(pair_id=row["pair_id"], label=row["label"],
                    fragment_a=row["fragment_a_id"], fragment_b=row["fragment_b_id"],
                    source_a=source_id(row["fragment_a_id"]), source_b=source_id(row["fragment_b_id"]),
                    strict_member=row["strict"], case_id=row["case_cluster"])
               for row in metadata["pairs"]]
    ids, sources, pairs = index_pair_records(records)
    if any(token not in prepared_index for token in ids):
        raise ValueError("REAL pair endpoint absent from prepared inputs")

    def load_mask(index):
        packed = arrays["packed_masks"][prepared_index[ids[index]]]
        return np.unpackbits(packed, axis=1)[None].astype(np.float32)

    return FragmentPopulation(ids, sources, pairs, load_mask,
                              dict(prepared_cache=str(Path(prepared).resolve()),
                                   prepared_manifest_sha256=metadata["manifest_sha256"],
                                   source_identity="existing REAL endpoint case prefix",
                                   real_translation_gt_opened=False))


def check_evaluation_gate(split, freeze_path, checkpoint_id):
    if split not in ("train", "val", "test", "real"):
        raise ValueError("unknown split")
    if split in ("train", "val"):
        return None  # Do not open external selection/model files for fitting.
    if not freeze_path:
        raise ValueError("TEST/REAL require --evaluation-freeze after VAL selection")
    path = Path(freeze_path)
    with path.open(encoding="utf-8") as stream:
        freeze = json.load(stream)
    if (freeze.get("status") != "complete" or freeze.get("selected_on") != "val"
            or freeze.get("matcher_checkpoint_id") != checkpoint_id):
        raise ValueError("external export requires completed VAL freeze for this encoder")
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def encode_pooled_features(model, masks):
    """Exact full-model coarse resize; never call projection/head/local forward."""
    if next(model.coarse.parameters()).device.type != "cpu":
        raise ValueError("retrieval feature export is CPU-only")
    value = torch.as_tensor(np.stack(masks), dtype=torch.float32, device="cpu")
    if value.ndim != 4 or tuple(value.shape[1:]) != (1, model.config.canvas_size, model.config.canvas_size):
        raise ValueError("model masks must be [1,canvas,canvas]")
    if not torch.isfinite(value).all() or not torch.all((value == 0) | (value == 1)):
        raise ValueError("model masks must be finite binary values")
    coarse = F.interpolate(value, size=(model.config.coarse_size,) * 2, mode="nearest")
    if not bool(model.coarse._validate(coarse, coarse).all()):
        raise ValueError("coarse resize produced an invalid/empty fragment")
    with torch.inference_mode():
        features = model.coarse.encoder(coarse).mean(dim=(-2, -1))
    result = features.detach().cpu().numpy().astype(np.float32)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("encoder produced nonfinite features")
    return result


def export_population(model, population, destination, *, split, checkpoint_id,
                      checkpoint_path="", batch_size=32, threads=2, evaluation_gate=None):
    """Fixture-friendly export API; production callers use run()'s split gate."""
    if type(threads) is not int or not 1 <= threads <= 2 or batch_size <= 0:
        raise ValueError("use 1–2 CPU threads and a positive batch size")
    if split in ("test", "real") and evaluation_gate is None:
        raise ValueError("external export must carry the checked VAL freeze")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    model.cpu().eval().requires_grad_(False)
    started, chunks = time.perf_counter(), []
    for start in range(0, len(population.fragment_ids), batch_size):
        stop = min(start + batch_size, len(population.fragment_ids))
        masks = [population.load_mask(i) for i in range(start, stop)]
        chunks.append(encode_pooled_features(model, masks))
        if stop % 512 < batch_size or stop == len(population.fragment_ids):
            print(json.dumps(dict(event="coarse_feature_export", split=split,
                                  fragments=stop, total_fragments=len(population.fragment_ids),
                                  elapsed_seconds=time.perf_counter() - started)), flush=True)
    features = np.concatenate(chunks)
    with (destination / "features.npz").open("xb") as stream:
        np.savez(stream, embeddings=features, fragment_id=np.asarray(population.fragment_ids, dtype="U"),
                 source_id=np.asarray(population.source_ids, dtype="U"))
    with (destination / "pairs.npz").open("xb") as stream:
        np.savez(stream, **population.pairs)
    manifest = dict(schema_version=SCHEMA, status="complete", split=split,
                    matcher_checkpoint_id=checkpoint_id, matcher_checkpoint_path=str(checkpoint_path),
                    feature_dim=int(features.shape[1]), feature_file="features.npz", pair_file="pairs.npz",
                    fragment_count=len(population.fragment_ids), sample_count=len(population.pairs["pair_id"]),
                    positive_count=int(population.pairs["label"].sum()), precision="fp32", device="cpu",
                    cpu_threads=threads, feature_layer="coarse.encoder final map spatial mean before BCE projection",
                    encoder_frozen=True, encoder_training_objective="previous Rachel BCE-trained encoder",
                    adaptation="frozen BCE encoder + downstream trainable InfoNCE projection; NOT full PairingNet",
                    local_matcher_executed=False, pose_gt_opened=False, evaluation_gate=evaluation_gate,
                    fragment_order="first occurrence in ordered pair endpoints",
                    provenance=population.provenance, elapsed_seconds=time.perf_counter() - started)
    with (destination / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return manifest


def run(args):
    if args.threads not in (1, 2) or args.batch_size <= 0:
        raise ValueError("use 1–2 CPU threads and a positive batch size")
    torch.set_num_threads(args.threads)
    if args.split in ("test", "real") and not args.evaluation_freeze:
        raise ValueError("TEST/REAL require completed --evaluation-freeze")
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_id = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if checkpoint_id != FIXED_CHECKPOINT_SHA:
        raise ValueError("this adaptation must use the fixed four-scale source checkpoint")
    gate = check_evaluation_gate(args.split, args.evaluation_freeze, checkpoint_id)
    if Path(args.output).exists():
        raise FileExistsError("use a new output directory: " + str(args.output))
    from staging.pairwise_v0_2.models.rachel_model_factory import load_rachel_checkpoint
    from staging.pairwise_v0_2.training.rachel_n512_sealed_test import _torch_load_checkpoint
    from experiments.rachel_n512_formal_30k.train_realism_data_ablation import check_fixed_architecture
    checkpoint = _torch_load_checkpoint(checkpoint_path)
    model = load_rachel_checkpoint(checkpoint)
    check_fixed_architecture(model, checkpoint)
    population = (load_real_population(args.prepared_cache) if args.split == "real"
                  else load_rachel_population(args.dataset, args.split))
    manifest = export_population(model, population, args.output, split=args.split,
                                 checkpoint_id=checkpoint_id, checkpoint_path=checkpoint_path,
                                 batch_size=args.batch_size, threads=args.threads, evaluation_gate=gate)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--prepared-cache", default=DEFAULT_PREPARED)
    parser.add_argument("--split", choices=("train", "val", "test", "real"), default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation-freeze")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu-threads", "--threads", dest="threads", type=int, choices=(1, 2), default=2)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

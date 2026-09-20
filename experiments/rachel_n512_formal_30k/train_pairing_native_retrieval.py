"""Frozen mask-only PairingNet stage1 -> official-structure stage2 retrieval.

TRAIN is the fixed materialized E1 table; clean VAL is a fixed input gallery.
There is deliberately no TEST/REAL loader or path argument. This is not the
official RGB experiment: valid-token attention/pooling and N512 are explicit
Rachel adaptations. All-gallery REAL evaluation belongs to a separate driver.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time
from types import ModuleType, SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = "rachel-pairing-native-retrieval-adapted/1"
OFFICIAL_COMMIT = "e878b781b2b2065a4b7da09d2f639e8f0a35e97a"
CLEAN_VAL_SHA256 = "daa6ccdd7686e93ba91ddfb1452c987145c26898a1917d2ac7d3180e199a8af8"
KS = (1, 5, 10, 20, 50)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_torch(path):
    # Outputs are trusted local experiment artifacts, never downloaded pickles.
    return torch.load(path, map_location="cpu", weights_only=False)


def fragment_input_digest(mask, points, valid):
    """Hash only exact model-facing arrays, with canonical dtypes and shapes."""
    value = hashlib.sha256(b"pairing-fragment-input/1\0")
    for name, array, dtype in (("mask", mask, np.float32),
                               ("points", points, np.float32),
                               ("valid", valid, np.bool_)):
        array = np.ascontiguousarray(array, dtype=dtype)
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise ValueError("non-finite model input: " + name)
        value.update(name.encode() + b"\0")
        value.update(str(array.shape).encode() + b"\0" + array.tobytes())
    return value.hexdigest()


def export_features(dataset, model, device, batch_size=8, positive_only=False, progress=None):
    """Deduplicate actual inputs, not tokens; supervision never enters _encode.

    TRAIN can select positive records only. VAL exports endpoints of *all*
    fixed rows, including negatives, to avoid a positives-only candidate pool.
    Positive pair rows remain records here; ranking later deduplicates edges.
    """
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    from staging.pairwise_v0_2.baselines.rachel_materialized_training import compact_benchmark_sample
    device = torch.device(device)
    cap = int(getattr(getattr(model, "config", None), "contour_cap", 512))
    model.eval().requires_grad_(False)
    key_to_index, keys, features, valids, pending = {}, [], [], [], []
    pair_ids, positive_pairs, tokens_by_key = [], [], defaultdict(set)
    input_keys_by_token = defaultdict(set)
    seen_pair_ids, selected = set(), 0

    def flush():
        if not pending:
            return
        masks = torch.from_numpy(np.stack([entry[0] for entry in pending])).to(device)
        points = torch.from_numpy(np.stack([entry[1] for entry in pending])).to(device)
        valid = torch.from_numpy(np.stack([entry[2] for entry in pending])).to(device)
        with torch.no_grad():
            _, contour, context = model._encode(masks, points, valid)
            combined = torch.cat((contour, context), dim=-1).float()
            if combined.shape != (len(pending), cap, 128):
                raise ValueError("stage1 must export unmerged contour64 + context64")
            if not torch.isfinite(combined).all():
                raise ValueError("non-finite stage1 features")
            combined = combined.masked_fill(~valid[:, :, None], 0).cpu()
        features.extend(combined.unbind(0))
        valids.extend(valid.cpu().unbind(0))
        pending.clear()

    for index in range(len(dataset)):
        # Metadata can skip negative TRAIN archives, but never supplies features.
        if positive_only and hasattr(dataset, "entries") and not dataset.entries[index]["label"]:
            continue
        sample = dataset[index]
        # E1 removes interior slots. Use exactly the same stable compaction as
        # stage1 training before hashing/encoding; no geometric relabeling.
        try:
            sample = compact_benchmark_sample(sample)
        except (AttributeError, TypeError) as error:
            raise ValueError("invalid sample for shared contour compaction") from error
        label = float(sample.label)
        if label not in (0., 1.):
            raise ValueError("pair label must be binary")
        if positive_only and not label:
            continue
        if sample.pair_id in seen_pair_ids:
            raise ValueError("duplicate source pair ID")
        seen_pair_ids.add(sample.pair_id)
        selected += 1
        endpoints = []
        for side in ("a", "b"):
            mask = np.asarray(getattr(sample, "mask_" + side), dtype=np.float32)
            points = np.asarray(getattr(sample, "points_rc_" + side), dtype=np.float32)
            valid = np.asarray(getattr(sample, "contour_valid_" + side), dtype=np.bool_)
            if points.ndim != 2 or points.shape != (len(valid), 2) or not 0 < len(valid) <= cap:
                raise ValueError("invalid contour input shape")
            length = int(valid.sum())
            if length < 1 or not np.array_equal(valid, np.arange(len(valid)) < length):
                raise ValueError("contour validity must be a nonempty prefix")
            if mask.ndim == 2:
                mask = mask[None]
            if mask.ndim != 3 or mask.shape[0] != 1:
                raise ValueError("requires one-channel mask input")
            key = fragment_input_digest(mask, points, valid)
            token = str(getattr(sample, "fragment_" + side + "_token", ""))
            tokens_by_key[key].add(token)
            input_keys_by_token[token].add(key)
            if key not in key_to_index:
                key_to_index[key] = len(keys)
                keys.append(key)
                padded_points = np.zeros((cap, 2), np.float32)
                padded_valid = np.zeros(cap, np.bool_)
                padded_points[:len(points)], padded_valid[:len(valid)] = points, valid
                pending.append((mask, padded_points, padded_valid))
                if len(pending) == batch_size:
                    flush()
            endpoints.append(key_to_index[key])
        if label:
            if endpoints[0] == endpoints[1]:
                raise ValueError("positive edge collapses to one identical model input")
            positive_pairs.append(endpoints)
            pair_ids.append(str(sample.pair_id))
        if progress is not None and selected % 250 == 0:
            progress(dict(selected_pair_records=selected, unique_inputs=len(keys)))
    flush()
    if not positive_pairs or not keys:
        raise ValueError("export requires positive pairs and fragment inputs")
    return dict(features=torch.stack(features), valid=torch.stack(valids),
                positive_pairs=torch.tensor(positive_pairs, dtype=torch.long),
                pair_ids=pair_ids, fragment_keys=keys,
                fragment_tokens=[sorted(tokens_by_key[key]) for key in keys],
                stats=dict(source_pairs=len(dataset), selected_pair_records=selected,
                           positive_pair_records=len(positive_pairs), unique_inputs=len(keys),
                           tokens_with_multiple_inputs=sum(len(x) > 1 for x in input_keys_by_token.values()),
                           inputs_with_multiple_tokens=sum(len(x) > 1 for x in tokens_by_key.values()),
                           feature_dtype="float32", contour_cap=cap,
                           gallery="all supplied pair endpoints, actual-input digest deduplicated",
                           unique_inputs_are_not_asserted_physical_entities=True))


class NativeRetrievalHead(nn.Module):
    """Official submodules, with valid-token masking added for Rachel padding."""
    def __init__(self, native_model):
        super().__init__()
        self.native = native_model

    def forward(self, features, valid):
        if features.ndim != 3 or features.shape[-1] != 128 or valid.shape != features.shape[:2]:
            raise ValueError("requires BxLx128 unmerged features and BxL validity")
        if valid.dtype != torch.bool or not bool(valid.any(dim=1).all()):
            raise ValueError("each fragment needs valid tokens")
        if features.shape[1] > self.native.tranct_length:
            raise ValueError("sequence exceeds configured native stage2 length")
        features = features.float().masked_fill(~valid[:, :, None], 0)
        contour = self.native.transformer_encoder_c(features[:, :, :64], input_mask=valid)
        context = self.native.transformer_encoder_t(features[:, :, 64:], input_mask=valid)
        fused, _ = self.native.selfgate(contour, context)
        pooled = fused.masked_fill(~valid[:, :, None], 0).sum(dim=1)
        pooled = pooled / valid.sum(dim=1, keepdim=True).to(pooled.dtype)
        return self.native.FC_layer(pooled)


def load_official_components(root):
    """Load exact official class ASTs without importing unused GCN dependencies.

    Class bodies are unchanged; NativeRetrievalHead replaces only forward's
    padding behavior. The complete InfoNCE module is imported unchanged.
    The caller first verifies this checkout with the existing official audit.
    """
    root = Path(root)
    pipeline = root / "PairingNet Code/utils/pipeline.py"
    wanted = {"SelfGateV2", "TransformerEncoderModel"}
    tree = ast.parse(pipeline.read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in wanted]
    if {node.name for node in classes} != wanted:
        raise ValueError("official stage2 classes absent")
    namespace = dict(torch=torch, nn=nn, __name__=__name__)
    code = ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[]))
    exec(compile(code, str(pipeline), "exec"), namespace)
    loss_path = root / "PairingNet Code/utils/infornce_loss.py"
    # Compile exact source in memory: importing with SourceFileLoader would
    # write __pycache__ into the immutable audited official checkout.
    loss_module = ModuleType("rachel_official_pairing_infonce")
    loss_module.__file__ = str(loss_path)
    exec(compile(loss_path.read_text(), str(loss_path), "exec"), loss_module.__dict__)
    return namespace["TransformerEncoderModel"], loss_module.InfoNCE


def verify_dependencies():
    versions = {}
    for package, required in (("linear-attention-transformer", "0.19.1"),
                              ("local-attention", "1.8.6")):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError("stage2 requires " + package + "==" + required) from error
        if versions[package] != required:
            raise RuntimeError("stage2 dependency differs: " + package + "==" + versions[package]
                               + "; requires " + required)
    return versions


def epoch_pair_order(n, seed, epoch):
    generator = torch.Generator().manual_seed(int(seed) + 1_000_003 * int(epoch))
    return torch.randperm(n, generator=generator)


def retrieval_metrics(embeddings, positive_pairs, ks=KS):
    """Full fixed gallery, excluding self; known positive edges, not full qrels.

    Deduplicate undirected GT edges and expand both directions. Unknown pairs
    participate in ranking but are never labeled negative for precision/AP.
    Ties are resolved by frozen gallery order, using stable descending sort.
    """
    vectors = torch.as_tensor(embeddings, dtype=torch.float32).detach().cpu()
    pairs = torch.as_tensor(positive_pairs, dtype=torch.long).cpu()
    if vectors.ndim != 2 or len(vectors) < 2 or not torch.isfinite(vectors).all():
        raise ValueError("requires finite gallery embeddings")
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0:
        raise ValueError("requires nonempty positive edge records")
    if bool((pairs < 0).any()) or bool((pairs >= len(vectors)).any()):
        raise ValueError("positive endpoint outside gallery")
    if bool((pairs[:, 0] == pairs[:, 1]).any()):
        raise ValueError("positive self-edge cannot be retrieved")
    if not ks or any(int(k) != k or k <= 0 for k in ks):
        raise ValueError("K values must be positive integers")
    undirected = sorted({tuple(sorted(row)) for row in pairs.tolist()})
    targets = defaultdict(set)
    for first, second in undirected:
        targets[first].add(second)
        targets[second].add(first)
    normalized = F.normalize(vectors, dim=-1)
    hits = {int(k): 0 for k in ks}
    query_ids = sorted(targets)
    max_k = min(max(ks), len(vectors) - 1)
    for offset in range(0, len(query_ids), 256):
        queries = query_ids[offset:offset + 256]
        similarity = normalized[queries] @ normalized.T
        similarity[torch.arange(len(queries)), torch.tensor(queries)] = -torch.inf
        ranked = torch.argsort(similarity, dim=1, descending=True, stable=True)[:, :max_k]
        for query, row in zip(queries, ranked.tolist()):
            for k in hits:
                hits[k] += len(targets[query].intersection(row[:min(k, len(vectors) - 1)]))
    denominator = 2 * len(undirected)
    return dict(**{"recall_at_%d" % k: hits[k] / denominator for k in hits},
                hits={str(k): hits[k] for k in hits}, positive_directed_edges=denominator,
                unique_undirected_positive_edges=len(undirected), positive_pair_records=len(pairs),
                gallery_size=len(vectors), positive_queries=len(query_ids),
                denominator="unique known positive undirected edges expanded in both directions",
                protocol="known-positive actual-input-edge diagnostic; unknown qrels not negative",
                tie_break="frozen gallery input order", self_excluded=True)


def encode_gallery(head, cache, device, batch_size):
    head.eval()
    result = []
    with torch.no_grad():
        for offset in range(0, len(cache["features"]), batch_size):
            features = cache["features"][offset:offset + batch_size].to(device)
            valid = cache["valid"][offset:offset + batch_size].to(device)
            result.append(head(features, valid).cpu())
    return torch.cat(result)


def pair_objective(head, cache, indices, device, criterion):
    edges = cache["positive_pairs"][indices]
    first, second = edges[:, 0], edges[:, 1]
    source = head(cache["features"][first].to(device), cache["valid"][first].to(device))
    target = head(cache["features"][second].to(device), cache["valid"][second].to(device))
    first, second = first.to(device), second.to(device)
    cross = criterion(source, target, gt_pairs=(first, second))
    same_source = criterion(source, source, gt_pairs=(first, first))
    same_target = criterion(target, target, gt_pairs=(second, second))
    return cross + .25 * same_source + .25 * same_target


def validate(head, cache, device, batch_size, criterion):
    vectors = encode_gallery(head, cache, device, batch_size)
    metrics = retrieval_metrics(vectors, cache["positive_pairs"])
    total, count = 0., 0
    with torch.no_grad():
        # Deterministic fixed VAL batches; in-batch negatives are batch-dependent.
        order = epoch_pair_order(len(cache["positive_pairs"]), 260911, 0)
        for offset in range(0, len(order), batch_size):
            indices = order[offset:offset + batch_size]
            loss = pair_objective(head, cache, indices, device, criterion)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite cleanVAL InfoNCE objective")
            total += float(loss) * len(indices)
            count += len(indices)
    metrics["infonce_objective"] = total / count
    return metrics


def _cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def run(args):
    from staging.pairwise_v0_2.baselines import rachel_pairingnet_benchmark as adapter
    from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset
    from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import MaterializedRachelDataset

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("formal training requires the authorized CUDA server")
    if args.epochs < 1 or args.batch_size < 2 or args.export_batch_size < 1:
        raise ValueError("invalid epoch/batch budget")
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device, root = torch.device(args.device), Path(args.output).resolve()
    dataset_root = Path(args.dataset).resolve(strict=True)
    manifest = Path(args.train_materialized_manifest).resolve(strict=True)
    stage1_path = Path(args.stage1_checkpoint).resolve(strict=True)
    official = Path(args.official_source).resolve(strict=True)
    official_audit = adapter.audit_official_source(official)
    dependencies = verify_dependencies()
    population = adapter.audit_train_val_population(dataset_root, require_formal_counts=False,
                                                     train_materialized_manifest=manifest)
    if population["manifests"]["val_sha256"] != CLEAN_VAL_SHA256:
        raise ValueError("requires unchanged original clean VAL manifest")
    stage1_record = load_torch(stage1_path)
    if (stage1_record.get("manifest_sha256") != population["manifests"]
            or stage1_record.get("run_config", {}).get("experimental_data") is not True
            or stage1_record.get("sealed_synthetic_accessed") is not False
            or stage1_record.get("real_data_accessed") is not False):
        raise ValueError("stage1 must be the new TRAIN/VAL-only winner on this exact materialized table")
    del stage1_record
    training, validation = MaterializedRachelDataset(manifest), RachelPairDataset(dataset_root, "val")
    if len(training) != args.expected_train_pairs or len(validation) != 3000:
        raise ValueError("unexpected TRAIN/cleanVAL population")
    identity = dict(schema_version=SCHEMA, driver_sha256=sha256(__file__), seed=args.seed,
                    train_manifest=str(manifest), train_manifest_sha256=sha256(manifest),
                    validation_manifest_sha256=CLEAN_VAL_SHA256,
                    stage1_checkpoint=str(stage1_path), stage1_checkpoint_sha256=sha256(stage1_path),
                    official_commit=OFFICIAL_COMMIT,
                    official_pipeline_sha256=sha256(official / "PairingNet Code/utils/pipeline.py"),
                    official_loss_sha256=sha256(official / "PairingNet Code/utils/infornce_loss.py"),
                    materialized_training_hook=population["manifests"].get("materialized_training_hook"),
                    epochs=args.epochs, batch_size=args.batch_size, export_batch_size=args.export_batch_size,
                    learning_rate=args.lr, weight_decay=args.weight_decay, temperature=.12,
                    expected_train_pairs=args.expected_train_pairs,
                    selection="cleanVAL known-positive full-gallery Recall@10, then @5, then lower InfoNCE; earliest tie")
    if args.resume:
        previous = json.loads((root / "protocol.json").read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("resume identity differs")
        last = load_torch(root / "last.pt") if (root / "last.pt").exists() else None
        if last is not None and last["identity"] != identity:
            raise ValueError("resume checkpoint identity differs")
        if last is not None:
            expected = last.get("feature_cache_sha256")
            if not isinstance(expected, dict) or set(expected) != {"train", "val"}:
                raise ValueError("resume checkpoint lacks frozen feature-cache hashes")
            for name in ("train", "val"):
                if sha256(root / (name + "_features.pt")) != expected[name]:
                    raise ValueError("resume feature cache content changed: " + name)
    else:
        root.mkdir(parents=True, exist_ok=False)
        last = None
    protocol = dict(**identity, official_source_audit=official_audit, population_audit=population,
                    architecture="official TransformerEncoderModel submodules, two64D branches, gate-concat128, mean, Linear128",
                    adaptations=["binary-mask context instead of RGB", "N512", "unmerged stage1 features",
                                 "shared stage1 stable active-contour compaction before input hashing",
                                 "valid-token attention mask and masked mean", "single-device effective in-batch negatives"],
                    pretrained="frozen newly trained PairingNet-adapted stage1; randomly initialized stage2",
                    train_positive_records=args.expected_train_pairs // 2, validation_positive_records=1500,
                    planned_positive_exposures=args.epochs * args.expected_train_pairs // 2,
                    planned_optimizer_updates=args.epochs * ((args.expected_train_pairs // 2 + args.batch_size - 1) // args.batch_size),
                    loss="official InfoNCE(s,t) + .25 InfoNCE(s,s) + .25 InfoNCE(t,t); T=.12",
                    optimizer="Adam", scheduler="CosineAnnealingLR over epoch budget", precision="FP32",
                    dependencies=dependencies,
                    gallery_qrels_complete=False, reported_precision_f1_ap_ndcg=False,
                    test_or_real_used_for_training_or_selection=False,
                    limitations=["VAL gallery is actual-input endpoint union of original clean3000, not author dataset",
                                 "known-positive edge Recall is not a fully annotated native retrieval benchmark",
                                 "actual-input digest identity is not asserted to be physical-fragment identity",
                                 "official in-batch loss only recognizes relationships present in that batch"])
    save_json(root / "protocol.json", protocol)
    started = time.monotonic()
    caches = {}
    stage1 = None
    try:
        for name, dataset, positive_only in (("train", training, True), ("val", validation, False)):
            path = root / (name + "_features.pt")
            if args.resume and path.exists():
                caches[name] = load_torch(path)
                if caches[name]["identity"] != identity:
                    raise ValueError("feature cache identity differs")
                continue
            if stage1 is None:
                stage1 = adapter.load_frozen_pairingnet_checkpoint(stage1_path, device)
                stage1.eval().requires_grad_(False)

            def progress(value):
                save_json(root / "status.json", dict(status="running", phase="export_" + name,
                          pid=os.getpid(), elapsed_s=time.monotonic() - started, **value))

            progress(dict(selected_pair_records=0, unique_inputs=0))
            cache = export_features(dataset, stage1, device, batch_size=args.export_batch_size,
                                    positive_only=positive_only, progress=progress)
            cache["identity"] = identity
            save_torch(path, cache)
            caches[name] = cache
    except Exception as error:
        save_json(root / "status.json", dict(status="failed", phase="export", pid=os.getpid(), error=repr(error)))
        raise
    del stage1
    torch.cuda.empty_cache()
    train, val = caches["train"], caches["val"]
    if len(train["positive_pairs"]) != args.expected_train_pairs // 2 or len(val["positive_pairs"]) != 1500:
        raise ValueError("positive training/validation denominator differs")
    cache_hashes = (dict(last["feature_cache_sha256"]) if last is not None else
                    {name: sha256(root / (name + "_features.pt")) for name in ("train", "val")})
    cache_summary = dict(train=train["stats"], val=val["stats"], feature_cache_sha256=cache_hashes,
                         actual_input_digest_overlap=len(set(train["fragment_keys"]) & set(val["fragment_keys"])),
                         train_gallery_sha256=hashlib.sha256("\n".join(train["fragment_keys"]).encode()).hexdigest(),
                         val_gallery_sha256=hashlib.sha256("\n".join(val["fragment_keys"]).encode()).hexdigest())
    save_json(root / "feature_export.json", cache_summary)
    NativeModel, InfoNCE = load_official_components(official)
    config = dict(tranct_length=512, max_length=512, in_channels_stage2=128, global_out_channels=128)
    # Initial stage2 weights must not depend on whether export was resumed or
    # whether stage1's constructor consumed random numbers in this invocation.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    head = NativeRetrievalHead(NativeModel(SimpleNamespace(**config))).to(device)
    criterion = InfoNCE(temperature=.12)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history, best_key, best_epoch, completed = [], None, None, 0
    if last:
        head.load_state_dict(last["model_state_dict"], strict=True)
        optimizer.load_state_dict(last["optimizer_state_dict"])
        scheduler.load_state_dict(last["scheduler_state_dict"])
        torch.set_rng_state(last["torch_rng_state"])
        torch.cuda.set_rng_state_all(last["cuda_rng_state"])
        history, best_key, best_epoch, completed = last["history"], tuple(last["best_key"]), last["best_epoch"], last["epoch"]

    try:
        for epoch in range(completed + 1, args.epochs + 1):
            head.train()
            order = epoch_pair_order(len(train["positive_pairs"]), args.seed, epoch)
            total, count, updates = 0., 0, 0
            for offset in range(0, len(order), args.batch_size):
                indices = order[offset:offset + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = pair_objective(head, train, indices, device, criterion)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite native stage2 training objective")
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(indices)
                count += len(indices)
                updates += 1
                if updates % 50 == 0:
                    save_json(root / "status.json", dict(status="running", phase="train", pid=os.getpid(),
                              epoch=epoch, positive_records=count, optimizer_updates=updates,
                              elapsed_s=time.monotonic() - started))
            validation_metrics = validate(head, val, device, args.batch_size, criterion)
            key = (validation_metrics["recall_at_10"], validation_metrics["recall_at_5"],
                   -validation_metrics["infonce_objective"])
            row = dict(epoch=epoch, positive_records=count, optimizer_updates=updates,
                       train_objective=total / count, validation=validation_metrics,
                       learning_rate=optimizer.param_groups[0]["lr"], elapsed_s=time.monotonic() - started)
            scheduler.step()
            history.append(row)
            winner = best_key is None or key > best_key
            if winner:
                best_key, best_epoch = key, epoch
            payload = dict(schema_version=SCHEMA, identity=identity, model_config=config,
                           model_state_dict=_cpu_state(head), epoch=epoch, best_key=list(best_key),
                           best_epoch=best_epoch, validation=validation_metrics, feature_export=cache_summary,
                           feature_cache_sha256=cache_hashes)
            if winner:
                save_torch(root / "best.pt", payload)
            save_torch(root / "last.pt", dict(**payload, optimizer_state_dict=optimizer.state_dict(),
                       scheduler_state_dict=scheduler.state_dict(), history=history,
                       torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state_all()))
            save_json(root / "epoch_history.json", history)
            save_json(root / "status.json", dict(status="running", phase="epoch_complete", pid=os.getpid(),
                      epoch=epoch, best_epoch=best_epoch, validation=validation_metrics,
                      elapsed_s=time.monotonic() - started))
            print(json.dumps(row, allow_nan=False), flush=True)
    except Exception as error:
        save_json(root / "status.json", dict(status="failed", pid=os.getpid(), error=repr(error)))
        raise
    completion = dict(schema_version=SCHEMA, status="complete", identity=identity, epochs=args.epochs,
                      best_epoch=best_epoch, best_checkpoint=str(root / "best.pt"),
                      best_checkpoint_sha256=sha256(root / "best.pt"),
                      final_validation=history[-1]["validation"],
                      best_validation=history[best_epoch - 1]["validation"],
                      positive_exposures=sum(row["positive_records"] for row in history),
                      optimizer_updates=sum(row["optimizer_updates"] for row in history),
                      feature_export=cache_summary, test_or_real_used=False)
    save_json(root / "completion.json", completion)
    save_json(root / "status.json", completion)
    return completion


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset", required=True, help="original clean Rachel root, VAL only")
    value.add_argument("--train-materialized-manifest", required=True)
    value.add_argument("--stage1-checkpoint", required=True, help="new PairingNet-adapted winner")
    value.add_argument("--official-source", required=True, help="clean pinned official PairingNet checkout")
    value.add_argument("--output", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--epochs", type=int, default=24)
    value.add_argument("--seed", type=int, default=260911)
    value.add_argument("--batch-size", type=int, default=32)
    value.add_argument("--export-batch-size", type=int, default=4)
    value.add_argument("--expected-train-pairs", type=int, choices=(6000, 12000, 24000), default=24000)
    value.add_argument("--lr", type=float, default=1e-3)
    value.add_argument("--weight-decay", type=float, default=1e-3)
    value.add_argument("--resume", action="store_true")
    return value


if __name__ == "__main__":
    run(parser().parse_args())

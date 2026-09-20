"""Independent paired-source N512/N1024 joint trainer, remote Linux CUDA only.

Both arms rebuild TRAIN and clean VAL correspondences from the same original
source cells. Old512 checkpoints/results are not this experiment's control.
TRAIN must be the complete fixed24000 paired derivative, VAL the prepared
complete3000 clean split at the selected cap. Twelve-pair preparation probes
cannot be substituted, even for this trainer's discard-only32-sample smoke.
V2/v3/v4 local-unknown ownership data requires the matching explicit
--source-density-version; cache versions are never inferred or silently
converted. Existing v1/v2/v3 identities stay compatible without added fields.

Example:
  python -m experiments.rachel_n512_formal_30k.train_score_density
    --checkpoint FULL24_METADATA.pt --dataset ORIGINAL_RELEASE
    --train-density-manifest PAIRED/train_n1024.json
    --clean-density-cache-root CLEAN_CACHE --contour-cap 1024
    --architecture candidate_pair --output NEW_RUN --stop-after-epoch 5

The original scoring loss/optimizer/6000-pair segment and50-epoch LR schedule
are reused unchanged. Cap is the sole architecture/input change. No rotation,
layout decoder modification, real/OOD calibration, or hidden staged training.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from experiments.rachel_n512_formal_30k.score_design_input_variants import full24_reference_config
from experiments.rachel_n512_formal_30k.train_score_design import (
    SEED, TRAIN_COUNT, VAL_COUNT, SEGMENT_SIZE, MAX_EPOCHS, MICROBATCH, EFFECTIVE_BATCH,
    MIN_SELECTION_EPOCH, BUDGETS, SELECTIONS, learning_rate, segment_plan, canonical_digest,
    run_lock, event, update_winners, validate_resume as validate_joint_progress,
    train_score_segment as train_density_segment)
from experiments.rachel_n512_formal_30k.train_joint_damage import build_random_model, state_digest
from experiments.rachel_n512_formal_30k.train_edge_weathering import (
    _cpu_model_state, _sha256, capture_rng_state, restore_rng_state)
from experiments.rachel_n512_formal_30k.train_realism_data_ablation import evaluate_pair_validation, save_json
from experiments.rachel_n512_formal_30k.resampled_input_support import make_ablation_loader
from staging.pairwise_v0_2.models.rachel_candidate_score import RachelCandidateScore, CandidateScoreConfig
from staging.pairwise_v0_2.pairwise_data.rachel_paired_density_dataset import PairedSourceDensityWeatheredDataset
from staging.pairwise_v0_2.pairwise_data.rachel_clean_density import CleanSourceDensityDataset
from staging.pairwise_v0_2.training import rachel_n512_runner as runner
from staging.pairwise_v0_2.training.rachel_weathering_training import make_weathering_loader

SCHEMA = "rachel-score-density-training/1"
CHECKPOINT_SCHEMA = "rachel-score-density-checkpoint/1"
SOURCE_DENSITY_VERSIONS = ("v1", "v2", "v3", "v4")
OWNERSHIP_V2 = "density-local-unknown-ownership/2"
OWNERSHIP_V3 = "density-local-unknown-ownership/3"
OWNERSHIP_V4 = "density-local-unknown-ownership/4"
OWNERSHIP_PROTOCOLS = {"v2": OWNERSHIP_V2, "v3": OWNERSHIP_V3, "v4": OWNERSHIP_V4}


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def common_validation_pipeline(protocol):
    return {k: v for k, v in protocol.items() if k not in ("contour_cap", "identity_sha256")}


def _source_density_consumer(version):
    """Registered descriptors: existing version strings remain byte-for-byte."""
    return dict(module="staging.pairwise_v0_2.pairwise_data.rachel_density_ownership_" + version,
        clean_reader="CleanSourceDensity" + version.upper() + "Dataset",
        ownership_protocol=OWNERSHIP_PROTOCOLS[version])


def validate_source_density_protocols(source_density_version, *, train_pipeline=None,
                                     clean_protocol=None, preparation=None):
    """Explicit producer/consumer agreement; never reinterpret a cache version."""
    if source_density_version not in SOURCE_DENSITY_VERSIONS:
        raise ValueError("source-density version must be explicitly v1, v2, v3 or v4")
    number = int(source_density_version[1:])
    ownership = OWNERSHIP_PROTOCOLS.get(source_density_version)
    if train_pipeline is not None:
        if (train_pipeline.get("version") != "paired-source-cell-density-materialization/%d" % number
                or train_pipeline.get("source_density_version", source_density_version) != source_density_version):
            raise ValueError("TRAIN producer version differs from explicit source-density consumer")
        if source_density_version != "v1" and (train_pipeline.get("source_density_version") != source_density_version
                or train_pipeline.get("ownership_protocol") != ownership):
            raise ValueError(source_density_version + " TRAIN producer lacks explicit local-unknown ownership provenance")
        if source_density_version == "v1" and train_pipeline.get("ownership_protocol") in OWNERSHIP_PROTOCOLS.values():
            raise ValueError("v1 TRAIN producer cannot claim newer ownership data")
    if clean_protocol is not None:
        if (clean_protocol.get("schema_version") != "rachel-clean-source-density-eval/%d" % number
                or clean_protocol.get("source_density_version", "v1") != source_density_version):
            raise ValueError("clean-cache producer version differs from explicit source-density consumer")
        if source_density_version != "v1" and clean_protocol.get("ownership_protocol") != ownership:
            raise ValueError(source_density_version + " clean cache lacks the registered local-unknown ownership protocol")
        if source_density_version == "v1" and clean_protocol.get("ownership_protocol") in OWNERSHIP_PROTOCOLS.values():
            raise ValueError("v1 consumer cannot reinterpret newer ownership data")
    if preparation is not None:
        if preparation.get("source_density_version", "v1") != source_density_version:
            raise ValueError("clean preparation receipt version differs from the selected consumer")
        if source_density_version != "v1" and preparation.get("ownership_protocol") != ownership:
            raise ValueError(source_density_version + " clean preparation lacks explicit ownership provenance")


def get_clean_density_reader(source_density_version="v1"):
    validate_source_density_protocols(source_density_version)
    if source_density_version == "v1":
        return CleanSourceDensityDataset
    # Newer modules are required only when explicitly selected. A missing reader
    # never falls back to an older version, even if unaffected samples agree.
    if source_density_version == "v2":
        from staging.pairwise_v0_2.pairwise_data.rachel_density_ownership_v2 import CleanSourceDensityV2Dataset
        return CleanSourceDensityV2Dataset
    if source_density_version == "v3":
        from staging.pairwise_v0_2.pairwise_data.rachel_density_ownership_v3 import CleanSourceDensityV3Dataset
        return CleanSourceDensityV3Dataset
    from staging.pairwise_v0_2.pairwise_data.rachel_density_ownership_v4 import CleanSourceDensityV4Dataset
    return CleanSourceDensityV4Dataset


def make_clean_density_dataset(root, split, contour_cap, cache_dir, *, source_density_version="v1"):
    reader = get_clean_density_reader(source_density_version)
    dataset = reader(root, split, contour_cap, cache_dir)
    validate_source_density_protocols(source_density_version, clean_protocol=dataset.protocol)
    return dataset


def identity_source_density_version(identity):
    """Missing version means historical v1, never inferred newer/fallback loading."""
    version = identity.get("source_density_version", "v1")
    validate_source_density_protocols(version)
    train = identity.get("train_source_pipeline", {})
    clean = identity.get("validation_density_protocol", {})
    newer_marked = any(train.get("version") == "paired-source-cell-density-materialization/" + v[1:]
        or train.get("source_density_version") == v or train.get("ownership_protocol") == OWNERSHIP_PROTOCOLS[v]
        or clean.get("schema_version") == "rachel-clean-source-density-eval/" + v[1:]
        or clean.get("source_density_version") == v for v in OWNERSHIP_PROTOCOLS)
    if version != "v1" or newer_marked:
        validate_source_density_protocols(version, train_pipeline=train, clean_protocol=clean)
        expected = _source_density_consumer(version)
        if identity.get("source_density_consumer") != expected:
            raise ValueError(version + " training identity lacks its explicit registered consumer")
    return version


def build_training_model(source, contour_cap, architecture):
    from experiments.rachel_n512_formal_30k.score_design_density_model import build_score_density_model
    reference_config = asdict(full24_reference_config())
    if (canonical(source.get("model_config")) != canonical(reference_config)
            or source.get("seam_loss_enabled", False)):
        raise ValueError("source must contain unchanged Full24 reference metadata; no weights are loaded")
    built = build_score_density_model(contour_cap, architecture=architecture, seed=SEED)
    # Factory preserves caller RNG; advance to the same post-construction point
    # as the existing Full24 joint trajectory, independent of the selected cap.
    metadata = dict(model_kind="full", model_options={}, model_config=reference_config,
        loss_config=source["loss_config"], seam_loss_enabled=False)
    reference, _, config, shared_digest = build_random_model(metadata, seed=SEED)
    if architecture != "original":
        reference = RachelCandidateScore(reference, CandidateScoreConfig(), architecture)
    expected_full = state_digest(reference)
    actual_full = state_digest(built.model)
    del reference
    if expected_full != actual_full:
        raise ValueError("density cap must not change any common initial parameter/buffer")
    return built, config, dict(initial_weights_sha256=actual_full,
        shared_base_initial_weights_sha256=shared_digest,
        initial_torch_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())


def validate_populations(training, validation, contour_cap, cache_root, source_density_version="v1"):
    """Bind complete original membership and source-target pipelines, no NPZ scan."""
    if contour_cap not in (512, 1024) or training.contour_cap != contour_cap or validation.contour_cap != contour_cap:
        raise ValueError("both derivative datasets must match the registered model cap")
    if len(training) != TRAIN_COUNT or len(validation) != VAL_COUNT:
        raise ValueError("requires full paired TRAIN24000 and clean VAL3000, never a subset probe")
    validate_source_density_protocols(source_density_version, train_pipeline=training.protocol.get("pipeline", {}),
        clean_protocol=validation.protocol)
    if (training.protocol.get("selection_mode") != "full_source_manifest"
            or training.protocol.get("paired512_control_required") is not True
            or training.protocol.get("live_dataset_modified") is not False):
        raise ValueError("TRAIN must use the complete paired source-density derivative")
    selection_path = training.root / "source_selection.json"
    selection = json.loads(selection_path.read_text())
    declared_identity = selection.pop("identity_sha256")
    if (declared_identity != training.identity or canonical_digest(selection) != training.identity
            or selection.get("schema_version") != training.protocol["pipeline"]["version"]
            or selection.get("selection_mode") != "full_source_manifest"
            or selection.get("source_indices") != list(range(TRAIN_COUNT))
            or selection.get("selected_pair_ids") != [e["pair_id"] for e in training.entries]
            or len(set(selection["selected_pair_ids"])) != TRAIN_COUNT
            or selection.get("source_manifest_sha256") != training.protocol.get("source_manifest_sha256")
            or selection.get("pipeline") != training.protocol.get("pipeline")):
        raise ValueError("TRAIN source-selection identity/order/pipeline differs")
    original = Path(training.protocol["source_manifest"]).resolve(strict=True)
    if _sha256(original) != training.protocol["source_manifest_sha256"]:
        raise ValueError("original fixed Full24 source manifest changed")
    companion = training.root / ("train_n%d.json" % (1024 if contour_cap == 512 else 512))
    paired = json.loads(companion.read_text())
    if (paired.get("status") != "complete" or paired.get("failed_pair_count") != 0
            or paired.get("schema_version") != "rachel-paired-source-density-train/1"
            or paired.get("contour_cap") != (1024 if contour_cap == 512 else 512)
            or paired.get("identity_sha256") != training.identity
            or paired.get("completed_pair_count") != TRAIN_COUNT
            or [e["pair_id"] for e in paired.get("entries", [])] != selection["selected_pair_ids"]):
        raise ValueError("companion cap lacks the same complete paired24000 population")
    validate_source_density_protocols(source_density_version, train_pipeline=paired.get("protocol", {}).get("pipeline", {}))
    cache_root = Path(cache_root).resolve(strict=True)
    expected_cache = cache_root / ("val_n%d" % contour_cap)
    if validation.split != "val" or validation.cache_dir != expected_cache:
        raise ValueError("VAL must use its independent cap-specific clean cache")
    receipt_path = cache_root / "clean_val_preparation.json"
    receipt = json.loads(receipt_path.read_text())
    validate_source_density_protocols(source_density_version, preparation=receipt)
    ids = [row["pair_id"] for row in validation.rows]
    records = receipt.get("records", [])
    if (receipt.get("schema_version") != "clean-source-density-preparation/1"
            or receipt.get("status") != "complete" or receipt.get("split") != "val"
            or receipt.get("full_split") is not True or receipt.get("failures") != []
            or any(receipt.get(k) != VAL_COUNT for k in ("original_split_count", "selected_count", "cached_pair_count"))
            or receipt.get("pair_ids") != ids or len(set(ids)) != VAL_COUNT
            or [r["pair_id"] for r in records] != ids
            or any(set(r.get("caps", {})) != {"512", "1024"} for r in records)
            or receipt.get("model_inference") is not False or receipt.get("model_or_threshold_selected") is not False):
        raise ValueError("clean_val_preparation must certify complete3000, both caps and zero failures")
    if any(bool(record["label"]) != bool(row["label"]) for record, row in zip(records, validation.rows)):
        raise ValueError("prepared VAL labels differ from original clean labels")
    if any((detail.get("positive_matches", 0) <= 0 if r["label"] else detail.get("positive_matches") != 0)
           for r in records for detail in r["caps"].values()):
        raise ValueError("clean VAL must retain genuine positive source correspondences, not all-ignore targets")
    if validation.stats != dict(positive=1500, negative=1500):
        raise ValueError("clean VAL must keep the complete original balanced labels")
    current = validation.protocol
    if (current.get("source_manifest_sha256") != _sha256(validation.manifest_path)
            or current.get("identity_sha256") != validation.identity or current.get("contour_cap") != contour_cap
            or current.get("source_pair_count") != VAL_COUNT or current.get("weathering_applied") is not False
            or current.get("checkpoint_threshold_selection_permitted") is not True
            or current.get("test_used_for_selection") is not False):
        raise ValueError("VAL source/selection pipeline changed")
    for cap in (512, 1024):
        cached = json.loads((cache_root / ("val_n%d" % cap) / "cache_identity.json").read_text())
        validate_source_density_protocols(source_density_version, clean_protocol=cached)
        no_hash = {k: v for k, v in cached.items() if k != "identity_sha256"}
        if (cached.get("contour_cap") != cap or cached.get("identity_sha256") != canonical_digest(no_hash)
                or common_validation_pipeline(cached) != common_validation_pipeline(current)
                or cap == contour_cap and cached != current):
            raise ValueError("paired clean VAL caches do not share the same original pipeline")
    populations = dict(train_density_identity_sha256=training.identity,
        train_source_selection_sha256=training.identity, train_source_selection_file_sha256=_sha256(selection_path),
        train_source_pipeline=training.protocol["pipeline"], train_density_protocol=training.protocol,
        train_original_manifest=str(original), train_original_manifest_sha256=_sha256(original),
        train_manifest=str(training.manifest_path), train_manifest_sha256=_sha256(training.manifest_path),
        validation_manifest=str(validation.manifest_path), validation_manifest_sha256=_sha256(validation.manifest_path),
        validation_density_protocol=current, validation_density_identity_sha256=validation.identity,
        validation_common_pipeline_sha256=canonical_digest(common_validation_pipeline(current)),
        validation_preparation=str(receipt_path), validation_preparation_sha256=_sha256(receipt_path),
        clean_density_cache_root=str(cache_root))
    # Preserve historical v1 resume identities exactly, rather than inventing
    # defaults which would make a still-valid v1 recovery fail identity equality.
    if source_density_version != "v1":
        populations.update(source_density_version=source_density_version,
            source_density_consumer=_source_density_consumer(source_density_version))
    return populations


def create_optimizer(model, architecture):
    model.train()
    if architecture == "original":
        model.requires_grad_(True)
    else:
        model.set_training_mode("joint")
    return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
        lr=learning_rate(1), weight_decay=1e-4)


def experiment_identity(args, *, source_path, model_metadata, config, digests, populations):
    if identity_source_density_version(populations) != getattr(args, "source_density_version", "v1"):
        raise ValueError("CLI source-density version differs from validated populations")
    return dict(schema_version=SCHEMA, seed=SEED, score_design=args.architecture, training_mode="joint",
        contour_cap=args.contour_cap, score_density_metadata=model_metadata, loss_config=asdict(config),
        candidate_config=asdict(CandidateScoreConfig()) if args.architecture != "original" else {},
        **digests, **populations, metadata_source_checkpoint_sha256=_sha256(source_path), source_weights_loaded=False,
        train_count=TRAIN_COUNT, train_split="train", validation_count=VAL_COUNT, validation_split="val",
        max_epochs=MAX_EPOCHS, segment_pairs=SEGMENT_SIZE, microbatch=MICROBATCH, effective_batch=EFFECTIVE_BATCH,
        optimizer="AdamW", weight_decay=1e-4, lr_by_epoch=[learning_rate(e) for e in range(1, MAX_EPOCHS + 1)],
        grad_clip_norm=5., precision="fp32", workers=args.workers,
        candidate_correctness_weight=.5 if args.architecture == "candidate_dual" else 0.,
        candidate_correctness_tolerance_px=20.,
        candidate_correctness_reduction="mean valid candidates per pair, then mean supervised pairs",
        validation_every_epochs=1, min_selection_epoch=MIN_SELECTION_EPOCH, budgets=list(BUDGETS),
        selection_rules={"max_f1": "VAL fused F1, then AP, then earliest epoch",
                         "recall95": "VAL precision at empirical95% recall, then AP, then earliest epoch"},
        optimizer_reset_at_epoch_or_budget=False, staged_training=False,
        paired_source_rebuilt512_control_required=True, legacy512_baseline_permitted=False,
        held_out_used_for_training_or_selection=False)


def checkpoint_metadata(metadata):
    return dict(score_density_checkpoint_schema=CHECKPOINT_SCHEMA, score_density_training_schema=SCHEMA,
        score_density_metadata=metadata, contour_cap=metadata["contour_cap"],
        model_kind="score_density_" + metadata["architecture"],
        model_config=metadata["base_model_metadata"]["model_config"], base_model_kind="full", base_model_options={})


def validate_density_checkpoint(checkpoint, identity):
    if (checkpoint.get("score_density_checkpoint_schema") != CHECKPOINT_SCHEMA
            or checkpoint.get("score_density_training_schema") != SCHEMA
            or identity.get("schema_version") != SCHEMA
            or any(k in checkpoint for k in ("score_design_schema", "s3_checkpoint_schema", "score_input_checkpoint_schema"))):
        raise ValueError("not an independent paired-source-density checkpoint")
    validate_joint_progress(checkpoint, identity)
    identity_source_density_version(identity)
    if (checkpoint.get("score_density_metadata") != identity.get("score_density_metadata")
            or checkpoint.get("contour_cap") != identity.get("contour_cap")
            or checkpoint.get("resample_contour_cap") != identity.get("contour_cap")
            or checkpoint.get("density_targets_rebuilt_from_source") is not True
            or checkpoint.get("epoch") != (checkpoint["completed_segments"] + 3) // 4):
        raise ValueError("density metadata/cap/epoch differs from committed identity")
    aliases = checkpoint_metadata(checkpoint["score_density_metadata"])
    if any(canonical(checkpoint.get(k)) != canonical(v) for k, v in aliases.items()):
        raise ValueError("checkpoint base/config/options differ from typed density metadata")


def load_score_density_checkpoint(checkpoint):
    from experiments.rachel_n512_formal_30k.score_design_density_model import restore_score_density_model
    validate_density_checkpoint(checkpoint, checkpoint.get("resume_identity", {}))
    return restore_score_density_model(checkpoint["score_density_metadata"], checkpoint["model_state_dict"]).model


def restore_training_state(model, optimizer, checkpoint, identity):
    validate_density_checkpoint(checkpoint, identity)
    restored = load_score_density_checkpoint(checkpoint)
    model.load_state_dict(restored.state_dict(), strict=True)
    del restored
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint["completed_segments"], checkpoint["winners"]


def payload(model, optimizer, *, metadata, identity, config, training_data, completed, winners, role):
    return dict(**checkpoint_metadata(metadata), model_state_dict=_cpu_model_state(model),
        optimizer_state_dict=optimizer.state_dict(), rng_state=capture_rng_state(),
        resume_identity=identity, loss_config=asdict(config), completed_segments=completed,
        epoch=(completed + 3) // 4, global_exposure=completed * SEGMENT_SIZE,
        optimizer_updates=completed * SEGMENT_SIZE // EFFECTIVE_BATCH, winners=winners,
        checkpoint_role=role, training_data=training_data, seed=SEED, initialization="random", source_weights_loaded=False,
        initial_weights_sha256=identity["initial_weights_sha256"],
        shared_base_initial_weights_sha256=identity["shared_base_initial_weights_sha256"],
        formal_training_counted=True, precision="fp32", seam_loss_enabled=False,
        resample_contour_cap=identity["contour_cap"], density_targets_rebuilt_from_source=True)


def publish_freezes(root, *, epoch, winners, identity):
    if epoch < MIN_SELECTION_EPOCH:
        return
    resolved = {}
    for name, record in winners.items():
        record = dict(record)
        record["checkpoint_sha256"] = _sha256(record["checkpoint"])
        record["validation_predictions_sha256"] = _sha256(record["validation_predictions"])
        resolved[name] = record
    if set(resolved) != set(SELECTIONS):
        raise RuntimeError("no fully covered eligible cap-specific clean VAL checkpoint")
    freeze = dict(schema_version=SCHEMA, status="frozen_at_budget" if epoch in BUDGETS else "provisional",
        budget_epochs=epoch, budget_exposures=epoch * TRAIN_COUNT, eligible_epoch_range=[5, epoch],
        contour_cap=identity["contour_cap"], score_density_metadata=identity["score_density_metadata"],
        winners=resolved, resume_identity=identity, resume_identity_sha256=canonical_digest(identity),
        selection_population="source-rebuilt cleanVAL3000 only", held_out_used_for_fit=False,
        paired_source_rebuilt512_control_required=True, legacy512_baseline_permitted=False)
    save_json(root / "current_winners.json", freeze)
    if epoch in BUDGETS:
        save_json(root / "budget_freezes" / ("%03d" % epoch) / "freeze.json", freeze)


def run(args):
    if args.contour_cap not in (512, 1024) or not 1 <= args.stop_after_epoch <= MAX_EPOCHS or args.workers < 0 or args.log_every <= 0:
        raise ValueError("invalid cap/stop/workers/log interval")
    if args.resume and args.smoke:
        raise ValueError("discard-only smoke cannot resume a formal trajectory")
    if platform.system() != "Linux" or not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("paired-density training requires the remote Linux CUDA server")
    torch.set_num_threads(1)
    source_path = Path(args.checkpoint).resolve(strict=True)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    built, config, digests = build_training_model(source, args.contour_cap, args.architecture)
    del source
    training = PairedSourceDensityWeatheredDataset(args.train_density_manifest)
    cache_root = Path(args.clean_density_cache_root).resolve(strict=True)
    # Cache/preparation must already exist; training must not begin by silently
    # replacing an unprepared clean VAL with an all-ignore inference wrapper.
    if not (cache_root / ("val_n%d" % args.contour_cap) / "cache_identity.json").is_file():
        raise ValueError("prepare the complete clean VAL density cache before training")
    version = args.source_density_version
    validate_source_density_protocols(version, train_pipeline=training.protocol.get("pipeline", {}))
    validation = make_clean_density_dataset(args.dataset, "val", args.contour_cap,
        cache_root / ("val_n%d" % args.contour_cap), source_density_version=version)
    populations = validate_populations(training, validation, args.contour_cap, cache_root, version)
    identity = experiment_identity(args, source_path=source_path, model_metadata=built.metadata,
        config=config, digests=digests, populations=populations)
    data_record = dict(kind="paired_source_density_materialized", unique_count=TRAIN_COUNT,
        contour_cap=args.contour_cap, manifest=str(training.manifest_path), stats=training.stats,
        protocol=training.protocol, paired_identity_sha256=training.identity)
    root = Path(args.output).resolve()
    if args.resume:
        if not root.is_dir():
            raise ValueError("resume directory does not exist")
    else:
        root.mkdir(parents=True, exist_ok=False)
    args.output = str(root)
    with run_lock(root):
        return _run_locked(args, root, built, training, validation, config, identity, data_record)


def _run_locked(args, root, built, training, validation, config, identity, data_record):
    device, cap = torch.device(args.device), args.contour_cap
    model, metadata = built.model.to(device), built.metadata
    optimizer = create_optimizer(model, args.architecture)
    completed, winners = 0, {}
    if args.resume:
        saved = torch.load(root / "last.pt", map_location="cpu", weights_only=False)
        completed, winners = restore_training_state(model, optimizer, saved, identity)
        del saved
        if completed > args.stop_after_epoch * 4:
            raise ValueError("requested stop precedes committed progress")
        if completed and completed % 4 == 0:
            publish_freezes(root, epoch=completed // 4, winners=winners, identity=identity)
    protocol = dict(**identity, status="running", arguments=vars(args), training_data=data_record,
        plan=segment_plan(), requested_stop_after_epoch=args.stop_after_epoch,
        smoke=bool(args.smoke), formal_training_counted=not bool(args.smoke),
        implementation_sha256=_sha256(__file__),
        runtime=dict(torch=torch.__version__, python=platform.python_version(), cuda=torch.version.cuda))
    save_json(root / "protocol.json", protocol)
    if not args.resume and not args.smoke:
        runner._atomic_torch_save(root / "last.pt", payload(model, optimizer, metadata=metadata,
            identity=identity, config=config, training_data=data_record, completed=0, winners={}, role="initial_recovery"))
    event(root, event="resume" if args.resume else "start", completed_segments=completed,
        stop_after_epoch=args.stop_after_epoch, contour_cap=cap, schema=SCHEMA)
    started = time.monotonic()
    try:
        for segment in segment_plan():
            number, epoch = segment["number"], segment["epoch"]
            if number <= completed:
                continue
            if epoch > args.stop_after_epoch:
                break
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(epoch)
            count = args.smoke or SEGMENT_SIZE
            order = runner.epoch_indices(TRAIN_COUNT, seed=SEED, epoch=epoch, limit=None)
            order = order[segment["offset"]:segment["offset"] + count]
            loader = make_weathering_loader(training, order, batch_size=MICROBATCH,
                num_workers=args.workers, seed=SEED + number, contour_cap=cap)
            save_json(root / "status.json", dict(status="running", phase="train", pid=os.getpid(),
                epoch=epoch, segment=number, global_exposure=completed * SEGMENT_SIZE,
                optimizer_updates=completed * SEGMENT_SIZE // EFFECTIVE_BATCH))
            torch.cuda.reset_peak_memory_stats(device)
            report = train_density_segment(model, loader, optimizer, config, device, args, epoch)
            if report["samples"] != count or report["optimizer_updates"] != count // EFFECTIVE_BATCH:
                raise RuntimeError("density segment exposure/update count differs")
            save_json(root / ("segment_%03d.json" % number), dict(segment=segment, training=report))
            if args.smoke:
                result = dict(status="smoke_complete", smoke=True, formal_training_counted=False,
                    weights_discarded=True, training=report, contour_cap=cap)
                save_json(root / "smoke.json", result)
                save_json(root / "status.json", result)
                protocol.update(result); save_json(root / "protocol.json", protocol)
                return result
            if segment["epoch_complete"]:
                from experiments.rachel_n512_formal_30k.recall_operating_points import fit_operating_points
                save_json(root / "status.json", dict(status="running", phase="validation", epoch=epoch,
                    global_exposure=number * SEGMENT_SIZE, optimizer_updates=number * SEGMENT_SIZE // EFFECTIVE_BATCH))
                val_loader = make_ablation_loader(validation, list(range(VAL_COUNT)), batch_size=8,
                    num_workers=args.workers, seed=SEED, contour_cap=cap)
                val_report, rows = evaluate_pair_validation(model, val_loader, device)
                points = fit_operating_points([r["label"] for r in rows], [r["classification"]["fused"] for r in rows])
                save_json(root / ("validation_%03d_rows.json" % epoch), rows)
                save_json(root / ("validation_%03d.json" % epoch), dict(epoch=epoch,
                    global_exposure=number * SEGMENT_SIZE, validation=val_report, operating_points=points,
                    selection_eligible=epoch >= MIN_SELECTION_EPOCH,
                    validation_density_identity_sha256=validation.identity))
                winners = update_winners(winners, root=root, epoch=epoch, report=val_report, points=points)
                event(root, event="validation_complete", epoch=epoch, validation=val_report, operating_points=points)
            recovery = payload(model, optimizer, metadata=metadata, identity=identity, config=config,
                training_data=data_record, completed=number, winners=winners,
                role="epoch_anchor" if segment["epoch_complete"] else "recovery")
            if segment["epoch_complete"]:
                runner._atomic_torch_save(root / ("epoch_%03d.pt" % epoch), recovery)
            runner._atomic_torch_save(root / "last.pt", recovery)
            completed = number
            del recovery
            if segment["epoch_complete"]:
                publish_freezes(root, epoch=epoch, winners=winners, identity=identity)
            event(root, event="segment_committed", segment=number, epoch=epoch, global_exposure=completed * SEGMENT_SIZE)
        if completed != args.stop_after_epoch * 4:
            raise RuntimeError("requested complete-epoch budget was not reached")
        status = "complete" if args.stop_after_epoch == MAX_EPOCHS else "budget_complete"
        result = dict(status=status, phase="train_val_complete", pid=os.getpid(), epoch=args.stop_after_epoch,
            completed_segments=completed, global_exposure=completed * SEGMENT_SIZE,
            optimizer_updates=completed * SEGMENT_SIZE // EFFECTIVE_BATCH,
            can_resume_to_epoch=MAX_EPOCHS if args.stop_after_epoch < MAX_EPOCHS else None,
            elapsed_s=time.monotonic() - started)
        protocol.update(result)
        save_json(root / "protocol.json", protocol); save_json(root / "status.json", result)
        event(root, event=status, **{k: v for k, v in result.items() if k != "status"})
        return result
    except BaseException as error:
        save_json(root / "status.json", dict(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=repr(error), completed_segments=completed, resumable_exposure=completed * SEGMENT_SIZE))
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--train-density-manifest", required=True)
    p.add_argument("--clean-density-cache-root", required=True)
    p.add_argument("--source-density-version", choices=SOURCE_DENSITY_VERSIONS, default="v1")
    p.add_argument("--contour-cap", required=True, type=int, choices=(512, 1024))
    p.add_argument("--architecture", required=True, choices=("original", "candidate_pair", "candidate_dual"))
    p.add_argument("--output", required=True)
    p.add_argument("--stop-after-epoch", type=int, default=MAX_EPOCHS)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--smoke", type=int, choices=(32, 64, 128), nargs="?", const=32)
    return p


if __name__ == "__main__":
    run(parser().parse_args())

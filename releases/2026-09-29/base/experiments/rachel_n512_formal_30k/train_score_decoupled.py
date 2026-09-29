"""Isolated M12+C8 training for thresholded-matrix / cross-attention heads.

No live score-design checkpoint is resumable here. --checkpoint supplies only
the Full24 architecture/loss metadata. --matcher-checkpoint optionally imports
an exact, complete M12 produced by this trainer, never a joint/P-R checkpoint.
The inherited 288k matcher exposures remain part of the 480k total budget.
Only clean SIM VAL in C13..20 selects auxiliary winners; epoch20 is primary.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.nn import functional as F

from experiments.rachel_n512_formal_30k.train_joint_damage import build_random_model, state_digest
from experiments.rachel_n512_formal_30k.train_edge_weathering import (
    _cpu_model_state, _sha256, capture_rng_state, restore_rng_state)
from experiments.rachel_n512_formal_30k.train_realism_data_ablation import (
    evaluate_pair_validation, save_json)
from experiments.rachel_n512_formal_30k.resampled_input_support import make_ablation_loader
from experiments.rachel_n512_formal_30k.train_score_design import canonical_digest, run_lock, event
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Pairwise
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import MaterializedWeatheredDataset
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset
from staging.pairwise_v0_2.training import rachel_n512_runner as runner
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig
from staging.pairwise_v0_2.training.rachel_weathering_training import (
    make_weathering_loader, compute_weathering_loss, WeatheringStatistics)

SCHEMA = "rachel-score-decoupled-training/1"
MODEL_SCHEMA = "rachel-decoupled-score-model/1"
RECEIPT_SCHEMA = "rachel-decoupled-matcher-pretraining/1"
SEED, HEAD_SEED, CLASSIFIER_SEED = 260913, 260914, 260915
TRAIN_COUNT, VAL_COUNT, SEGMENT_SIZE = 24000, 3000, 6000
MATCHER_EPOCHS, TOTAL_EPOCHS = 12, 20
LOSS_NAMES = ("fused_pair_bce", "coarse_pair_bce", "local_pair_bce",
              "assignment_nll", "translation_smooth_l1", "sinkhorn_residual")
PROBE_SCHEMA = "rachel-decoupled-classifier-input-probe/1"
RUNTIME_BATCHING_SCHEMA = "rachel-decoupled-runtime-batching/1"
PHYSICAL_BATCH_ENV = "RACHEL_SCORE_PHYSICAL_MICROBATCH_N512"
FORMAL_PHYSICAL_MICROBATCHES = (1, 2, 4, 8, 16)
SAMPLEWISE_LOSS_SCHEMA = "rachel-decoupled-samplewise-loss/1"


class FailedClassifierInputProbe(RuntimeError):
    pass


def phase_for_epoch(epoch):
    if type(epoch) is not int or not 1 <= epoch <= TOTAL_EPOCHS:
        raise ValueError("epoch must be1..20")
    return "matcher" if epoch <= MATCHER_EPOCHS else "classifier"


def learning_rate(epoch):
    phase_for_epoch(epoch)
    phase_epoch = epoch if epoch <= MATCHER_EPOCHS else epoch - MATCHER_EPOCHS
    return 1e-4 if phase_epoch <= 3 else 2e-5


def segment_plan():
    return [dict(number=(e - 1) * 4 + i + 1, epoch=e, phase=phase_for_epoch(e),
        offset=i * SEGMENT_SIZE, count=SEGMENT_SIZE,
        global_start=(e - 1) * TRAIN_COUNT + i * SEGMENT_SIZE,
        global_stop=(e - 1) * TRAIN_COUNT + (i + 1) * SEGMENT_SIZE,
        epoch_complete=i == 3, learning_rate=learning_rate(e))
        for e in range(1, TOTAL_EPOCHS + 1) for i in range(4)]


def matcher_loss_config(base):
    return replace(base, fused_pair_weight=0., coarse_pair_weight=0., local_pair_weight=0.,
                   assignment_weight=.5, translation_weight=.5, sinkhorn_residual_weight=.05)


def phase_loss(output, targets, pose, base, phase, *, samplewise=False):
    """C computes only PairBCE, without reading correspondence or pose labels."""
    if samplewise:
        from experiments.rachel_n512_formal_30k.decoupled_samplewise_loss import compute_samplewise_phase_loss
        return compute_samplewise_phase_loss(output, targets, pose, base, phase)
    if phase == "matcher":
        result = compute_weathering_loss(output, *targets, config=matcher_loss_config(base),
                                         pose_supervision_enabled=pose)
        return result.total, {key: getattr(result, key) for key in LOSS_NAMES}
    if phase != "classifier":
        raise ValueError("unregistered training phase")
    labels = targets[0].to(output.fused_logit.dtype)
    valid = output.training_valid
    values = F.binary_cross_entropy_with_logits(output.fused_logit, labels, reduction="none")
    total = (values * valid.to(values.dtype)).sum() / valid.sum().clamp_min(1)
    if not torch.isfinite(total).item():
        raise FloatingPointError("nonfinite classifier PairBCE")
    components = {key: total.detach().new_zeros(()) for key in LOSS_NAMES}
    components["fused_pair_bce"] = total
    return total, components


def physical_batch_size(args):
    value = getattr(args, "physical_microbatch", None)
    return args.microbatch if value is None else value


def effective_batch_size(args):
    # The optional runtime value is currently useful for discard benchmarks.
    # Formal entry rejects values other than16 until update-history migration
    # has its own registered contract; never misreport exposure//16 as actual.
    value = getattr(args, "runtime_effective_batch", None)
    return args.effective_batch if value is None else value


def resolve_runtime_arguments(args):
    """Explicit CLI wins; N512 environment defaults never affect step3/2048."""
    if getattr(args, "physical_microbatch", None) is not None:
        source = "explicit_cli"
    elif args.sampling in ("original512", "paired512") and PHYSICAL_BATCH_ENV in os.environ:
        try:
            args.physical_microbatch = int(os.environ[PHYSICAL_BATCH_ENV])
        except ValueError as error:
            raise ValueError("invalid " + PHYSICAL_BATCH_ENV) from error
        source = PHYSICAL_BATCH_ENV
    else:
        source = "legacy_default"
    args.physical_microbatch_source = source


def validate_runtime_batching(runtime, identity, completed):
    """No relaxation of the original identity or committed update cadence."""
    if runtime is None:
        return
    if (runtime.get("schema_version") != RUNTIME_BATCHING_SCHEMA or
            runtime.get("origin_resume_identity") != identity or
            runtime.get("origin_resume_identity_sha256") != canonical_digest(identity) or
            identity["microbatch"] != 1 or identity["effective_batch"] != 16 or
            runtime.get("logical_microbatch") != 1 or runtime.get("effective_batch") != 16):
        raise ValueError("runtime batching identity or logical loss contract differs")
    history = runtime.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("runtime batching requires an explicit migration history")
    previous, previous_segment = 1, -1
    for record in history:
        at = record.get("committed_segments")
        if (type(at) is not int or not 0 <= at <= completed or at < previous_segment or
                record.get("resumable_exposure") != at * SEGMENT_SIZE or
                record.get("optimizer_updates") != at * SEGMENT_SIZE // 16 or
                record.get("previous_physical_microbatch") != previous or
                record.get("physical_microbatch") not in FORMAL_PHYSICAL_MICROBATCHES or
                record.get("samplewise_loss_schema") != SAMPLEWISE_LOSS_SCHEMA or
                record.get("reason") != "user_authorized_implementation_migration" or
                record.get("bitwise_trajectory_equivalence_claimed") is not False):
            raise ValueError("invalid or nonmonotonic runtime batching history")
        source_sha = record.get("source_checkpoint_sha256")
        if at and (not isinstance(source_sha, str) or len(source_sha) != 64):
            raise ValueError("runtime migration requires its exact source checkpoint SHA")
        previous, previous_segment = record["physical_microbatch"], at
    if runtime.get("physical_microbatch") != previous:
        raise ValueError("runtime physical batch differs from its history")


def prepare_runtime_batching(args, identity, completed, *, checkpoint=None,
                             source_checkpoint_sha256=None, origin_protocol=None):
    """Append a bounded runtime event after strict weight/optimizer/RNG restore.

    The first origin protocol is retained, and history travels with checkpoints.
    A later resume cannot silently revert a migrated run when its override is
    missing. Importing M12 for S4 remains a separate base-only import operation.
    """
    previous = (checkpoint or {}).get("runtime_batching")
    validate_runtime_batching(previous, identity, completed)
    requested = getattr(args, "physical_microbatch", None)
    if requested is None:
        if previous is not None:
            raise ValueError("migrated resume requires an explicit physical batch CLI or N512 environment")
        return None
    if identity["microbatch"] != 1 or identity["effective_batch"] != 16 or requested not in FORMAL_PHYSICAL_MICROBATCHES:
        raise ValueError("physical migration requires logical micro1/effective16")
    runtime = deepcopy(previous) if previous is not None else dict(
        schema_version=RUNTIME_BATCHING_SCHEMA, origin_resume_identity=deepcopy(identity),
        origin_resume_identity_sha256=canonical_digest(identity),
        origin_protocol=deepcopy(origin_protocol), logical_microbatch=1, effective_batch=16, history=[])
    old = runtime.get("physical_microbatch", 1)
    if previous is None or old != requested:
        from experiments.rachel_n512_formal_30k import decoupled_samplewise_loss
        runtime["history"].append(dict(committed_segments=completed,
            resumable_exposure=completed * SEGMENT_SIZE, optimizer_updates=completed * SEGMENT_SIZE // 16,
            previous_physical_microbatch=old, physical_microbatch=requested,
            source_checkpoint_sha256=source_checkpoint_sha256,
            override_source=getattr(args, "physical_microbatch_source", "explicit_api"),
            implementation_sha256=_sha256(__file__),
            samplewise_loss_sha256=_sha256(decoupled_samplewise_loss.__file__),
            samplewise_loss_schema=SAMPLEWISE_LOSS_SCHEMA,
            reason="user_authorized_implementation_migration",
            preserved=["model_state", "optimizer_state", "RNG_state", "sample_order", "learning_rate", "effective_batch16", "logical_samplewise_loss"],
            bitwise_trajectory_equivalence_claimed=False))
    runtime["physical_microbatch"] = requested
    validate_runtime_batching(runtime, identity, completed)
    return runtime


def build_random_decoupled(source, head_kind, matrix_threshold, cap, matrix_head_revision=None,
                           cross_attention_depth=1):
    """Canonical random Full24 base; head RNG cannot change matcher RNG.

    Changing the accepted cap creates no cap-sized parameters. The2048 base
    receives only the freshly initialized512 reference tensors, not checkpoint
    weights. The complete initial base digest is identical at both caps.
    """
    from staging.pairwise_v0_2.models.rachel_decoupled_score import DecoupledScoreModel
    if type(cross_attention_depth) is not int or cross_attention_depth not in (1, 2, 4):
        raise ValueError("cross-attention depth must be1,2,or4")
    if head_kind != "cross_attention" and cross_attention_depth != 1:
        raise ValueError("cross-attention depth requires the cross_attention head")
    if cap not in (512, 2048):
        raise ValueError("this new protocol accepts only cap512 or2048")
    base, architecture, loss, base_digest = build_random_model(source, seed=SEED)
    after_base = capture_rng_state()
    if cap != base.config.contour_cap:
        reference = base
        base = RachelN512Pairwise(replace(reference.config, contour_cap=cap))
        base.load_state_dict(reference.state_dict(), strict=True)
        del reference
    # Head construction is isolated even across different head parameter counts.
    torch.manual_seed(HEAD_SEED)
    model = DecoupledScoreModel(base, head_kind=head_kind, matrix_threshold=matrix_threshold,
                               matrix_head_revision=matrix_head_revision,
                               model_options=({"cross_attention_depth": cross_attention_depth}
                                              if cross_attention_depth != 1 else None))
    restore_rng_state(after_base)
    model.set_phase("matcher")
    if state_digest(model.base_model) != base_digest:
        raise RuntimeError("cap/head construction changed random base tensors")
    return model, architecture, loss, dict(initial_weights_sha256=state_digest(model),
        shared_base_initial_weights_sha256=base_digest,
        classifier_initial_weights_sha256=state_digest(model.score_head))


class WeatheredView:
    """Train-only view; clean/S5 readers retain their sample-only public API."""
    def __init__(self, source):
        self.source = source
    def __len__(self):
        return len(self.source)
    def __getitem__(self, index):
        return self.source.weathered(index)
    def __getattr__(self, name):
        source = self.__dict__.get("source")
        if source is None:
            raise AttributeError(name)
        return getattr(source, name)


def population_record(dataset, path, *, sampling):
    return dict(count=len(dataset), split=dataset.split,
        contour_cap=dataset.contour_cap, sampling=sampling,
        manifest=str(Path(path).resolve()), manifest_sha256=_sha256(path),
        producer_identity=getattr(dataset, "identity", None),
        protocol=getattr(dataset, "protocol", {}))


def make_populations(args):
    if args.sampling == "original512":
        training = MaterializedWeatheredDataset(args.train_materialized_manifest)
        validation = RachelPairDataset(args.dataset, "val")
        paths = (args.train_materialized_manifest, Path(args.dataset) / "pairs" / "val.jsonl")
        cap = 512
    else:
        from staging.pairwise_v0_2.pairwise_data.rachel_step_dataset import StepSourceDataset
        if not args.density_train_manifest or not args.clean_val_manifest:
            raise ValueError("paired512/step3 require both completed TRAIN and VAL step manifests")
        raw_train = StepSourceDataset(args.density_train_manifest, sampling=args.sampling)
        validation = StepSourceDataset(args.clean_val_manifest, sampling=args.sampling)
        training = WeatheredView(raw_train)
        paths = (args.density_train_manifest, args.clean_val_manifest)
        cap = 512 if args.sampling == "paired512" else 2048
    if (len(training), len(validation), training.split, validation.split) != (TRAIN_COUNT, VAL_COUNT, "train", "val"):
        raise ValueError("requires complete fixed TRAIN24000 and clean SIM VAL3000, never TEST/REAL/OOD")
    # The original clean reader is intrinsically512, without a cap attribute.
    if getattr(validation, "contour_cap", 512) != cap or training.contour_cap != cap:
        raise ValueError("reader sampling cap differs from the registered arm")
    if not hasattr(validation, "contour_cap"):
        validation.contour_cap = 512
    records = dict(train=population_record(training, paths[0], sampling=args.sampling),
                   val=population_record(validation, paths[1], sampling=args.sampling))
    return training, validation, cap, records


def experiment_identity(args, source_path, model, loss_config, records, digests):
    identity = dict(schema_version=SCHEMA, schedule="M12_C8_pair_decoupled",
        seed=SEED, head_seed=HEAD_SEED, classifier_phase_seed=CLASSIFIER_SEED,
        head_kind=args.head_kind, matrix_threshold=args.matrix_threshold,
        matrix_head_revision=model.metadata()["matrix_head_revision"],
        sampling=args.sampling, contour_cap=model.config.contour_cap,
        base_model_config=asdict(model.config), **digests,
        metadata_source_checkpoint_sha256=_sha256(source_path), source_weights_loaded=False,
        matcher_checkpoint_sha256=_sha256(args.matcher_checkpoint) if args.matcher_checkpoint else None,
        populations=records, loss_config=asdict(loss_config), matcher_loss_config=asdict(matcher_loss_config(loss_config)),
        classifier_loss=dict(fused_pair_weight=1., coarse_pair_weight=0., local_pair_weight=0.,
                             assignment_weight=0., translation_weight=0., sinkhorn_residual_weight=0.,
                             candidate_correctness_weight=0.),
        max_epochs=TOTAL_EPOCHS, matcher_epochs=MATCHER_EPOCHS, classifier_epochs=8,
        train_count=TRAIN_COUNT, validation_count=VAL_COUNT, segment_pairs=SEGMENT_SIZE,
        microbatch=args.microbatch, effective_batch=args.effective_batch,
        optimizer="AdamW", weight_decay=1e-4, lr_by_epoch=[learning_rate(e) for e in range(1, 21)],
        optimizer_lifetime_groups=["base", "new_head"], optimizer_reset_at_transition=False,
        classifier_phase_rng_reset_once=True, frozen_base_eval_in_classifier=True,
        grad_clip_norm=5., precision="fp32", workers=args.workers,
        validation_every_classifier_epochs=1, matcher_pair_selection=False,
        classifier_input_probe=dict(schema_version=PROBE_SCHEMA, population="fixed first32 TRAIN positives and first32 negatives",
            count=64, once_before_C13=True, automatic_threshold_change=False,
            reject_all_empty_positive_binary_matrices=True, held_out_used=False),
        primary_selection="fixed_epoch", auxiliary_selection_epochs=list(range(13, 21)),
        held_out_used_for_training_or_selection=False)
    # Keep legacy depth-one identities byte-for-byte structurally compatible.
    # Classifier depth belongs to the run identity, never to the shared M12 contract.
    options = model.metadata().get("model_options", {})
    if options:
        identity["model_options"] = dict(options)
    return identity


def matcher_contract(identity):
    """Head-independent binding: only the same data/cap/random-base M12 is reusable."""
    fields = ("seed", "base_model_config", "shared_base_initial_weights_sha256",
              "populations", "matcher_loss_config", "matcher_epochs", "train_count",
              "segment_pairs", "microbatch", "effective_batch", "optimizer", "weight_decay",
              "grad_clip_norm", "precision", "workers")
    result = {key: identity[key] for key in fields}
    result["lr_by_epoch"] = identity["lr_by_epoch"][:MATCHER_EPOCHS]
    return result


def matcher_receipt(model, identity):
    return dict(schema_version=RECEIPT_SCHEMA, completed_epochs=12, pair_exposures=288000,
        optimizer_updates=288000 // identity["effective_batch"],
        base_state_sha256=state_digest(model.base_model),
        initial_base_state_sha256=identity["shared_base_initial_weights_sha256"],
        matcher_contract_sha256=canonical_digest(matcher_contract(identity)),
        loss=identity["matcher_loss_config"], pair_BCE_used=False, GT_layout_classification_labels_used=False,
        source_run_identity_sha256=canonical_digest(identity))


def verify_receipt(model, receipt, identity):
    if not receipt or receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError("a typed complete M12 receipt is required")
    expected = dict(completed_epochs=12, pair_exposures=288000,
        optimizer_updates=288000 // identity["effective_batch"],
        matcher_contract_sha256=canonical_digest(matcher_contract(identity)), pair_BCE_used=False)
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("matcher receipt population/configuration/budget differs")
    if state_digest(model.base_model) != receipt.get("base_state_sha256"):
        raise ValueError("frozen matcher tensor/buffer digest differs from M12")


def create_optimizer(model):
    # Frozen parameters have grad=None and no AdamW state/decay/update. The new
    # head has no optimizer history at C13; the base state is retained for resume.
    return torch.optim.AdamW([dict(params=list(model.base_model.parameters()), phase_family="base"),
        dict(params=list(model.score_head.parameters()), phase_family="new_head")], lr=1e-4, weight_decay=1e-4)


def validate_head_revision(metadata, identity):
    """Absent old matrix revision means legacy, never the repaired default.

    The M12 contract intentionally excludes the classifier architecture. Full
    checkpoint loading/resume does not: a parameter-free activation change is
    still a different classifier even if tensor shapes happen to agree.
    """
    from staging.pairwise_v0_2.models.rachel_decoupled_score import matrix_revision_from_metadata
    if (metadata.get("head_kind") != identity.get("head_kind") or
            matrix_revision_from_metadata(metadata) != matrix_revision_from_metadata(identity)):
        raise ValueError("matrix head revision differs from the frozen training identity")
    if metadata.get("model_options", {}) != identity.get("model_options", {}):
        raise ValueError("classifier model options differ from the frozen training identity")


def validate_checkpoint_progress(checkpoint, identity):
    if checkpoint.get("decoupled_training_schema") != SCHEMA or checkpoint.get("decoupled_score_schema") != MODEL_SCHEMA:
        raise ValueError("not a new decoupled checkpoint; old joint/staged checkpoints are prohibited")
    if checkpoint.get("resume_identity") != identity:
        raise ValueError("decoupled resume identity differs")
    validate_head_revision(checkpoint.get("decoupled_score", {}), identity)
    completed = checkpoint.get("completed_segments")
    if type(completed) is not int or not 0 <= completed <= 80:
        raise ValueError("invalid committed segment count")
    validate_runtime_batching(checkpoint.get("runtime_batching"), identity, completed)
    epoch = (completed + 3) // 4
    expected = dict(epoch=epoch, global_exposure=completed * SEGMENT_SIZE,
        optimizer_updates=completed * SEGMENT_SIZE // identity["effective_batch"],
        phase=phase_for_epoch(max(1, epoch)))
    if any(checkpoint.get(k) != v for k, v in expected.items()):
        raise ValueError("checkpoint phase/exposure/updates differ from committed segments")
    if (completed >= 48) != bool(checkpoint.get("matcher_pretraining_receipt")):
        raise ValueError("checkpoint M12 receipt presence differs")
    if checkpoint.get("decoupled_score", {}).get("phase") != expected["phase"]:
        raise ValueError("model and training phase differ")
    return expected["phase"]


def load_decoupled_checkpoint(checkpoint):
    """Typed inference loader; evaluator must not use the old joint factory."""
    from staging.pairwise_v0_2.models.rachel_decoupled_score import load_decoupled_score_checkpoint
    identity = checkpoint.get("resume_identity", {})
    validate_checkpoint_progress(checkpoint, identity)
    metadata = checkpoint.get("decoupled_score", {})
    if (metadata.get("base_model_config") != identity["base_model_config"] or
            metadata.get("head_kind") != identity["head_kind"] or
            metadata.get("matrix_threshold") != identity["matrix_threshold"]):
        raise ValueError("model metadata differs from the frozen training identity")
    model = load_decoupled_score_checkpoint(checkpoint)
    if checkpoint.get("matcher_pretraining_receipt"):
        verify_receipt(model, checkpoint["matcher_pretraining_receipt"], identity)
    return model


def import_matcher(model, checkpoint, identity):
    """Import only an actual M12 base, never a classifier or optimizer history."""
    validate_head_revision(model.metadata(), identity)
    old_identity = checkpoint.get("resume_identity", {})
    phase = validate_checkpoint_progress(checkpoint, old_identity)
    if checkpoint["completed_segments"] != 48 or phase != "matcher":
        raise ValueError("--matcher-checkpoint must be the exact complete epoch12 checkpoint")
    if matcher_contract(old_identity) != matcher_contract(identity):
        raise ValueError("M12 source uses different sampling/data/loss/random initialization")
    old = load_decoupled_checkpoint(checkpoint)
    model.base_model.load_state_dict(old.base_model.state_dict(), strict=True)
    receipt = checkpoint["matcher_pretraining_receipt"]
    verify_receipt(model, receipt, identity)
    return receipt


def checkpoint_payload(model, optimizer, *, identity, loss_config, completed, receipt, winners, role,
                       runtime_batching=None):
    validate_head_revision(model.metadata(), identity)
    epoch = (completed + 3) // 4
    inherited = 288000 if identity.get("matcher_checkpoint_sha256") else 0
    payload = dict(decoupled_training_schema=SCHEMA, decoupled_score_schema=MODEL_SCHEMA,
        decoupled_score=model.metadata(), model_state_dict=_cpu_model_state(model),
        optimizer_state_dict=optimizer.state_dict(), rng_state=capture_rng_state(),
        model_kind="score_decoupled_" + model.head_kind, model_config=asdict(model.config),
        resume_identity=identity, loss_config=asdict(loss_config),
        completed_segments=completed, epoch=epoch, phase=phase_for_epoch(max(1, epoch)),
        global_exposure=completed * SEGMENT_SIZE,
        optimizer_updates=completed * SEGMENT_SIZE // identity["effective_batch"],
        inherited_matcher_exposures=inherited, executed_pair_exposures=completed * SEGMENT_SIZE - inherited,
        matcher_pretraining_receipt=receipt, winners=winners, checkpoint_role=role,
        initialization="random matcher plus random independent classifier",
        metadata_source_weights_loaded=False, matcher_weights_reused=bool(inherited),
        formal_training_counted=True, seed=SEED, precision="fp32")
    if runtime_batching is not None:
        validate_runtime_batching(runtime_batching, identity, completed)
        payload["runtime_batching"] = deepcopy(runtime_batching)
    return payload


def restore_training_state(model, optimizer, checkpoint, identity):
    if checkpoint.get("inference_only"):
        raise ValueError("inference-only converted checkpoint cannot resume training")
    phase = validate_checkpoint_progress(checkpoint, identity)
    validate_head_revision(model.metadata(), identity)
    if "rng_state" not in checkpoint or "optimizer_state_dict" not in checkpoint:
        raise ValueError("resume requires optimizer and RNG states")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.set_phase(phase)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if [g.get("phase_family") for g in optimizer.param_groups] != ["base", "new_head"]:
        raise ValueError("optimizer lifetime groups differ")
    receipt = checkpoint["matcher_pretraining_receipt"]
    if receipt:
        verify_receipt(model, receipt, identity)
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint["completed_segments"], receipt, checkpoint["winners"]


def configure_phase(model, epoch, identity, receipt, *, entering_classifier=False):
    phase = phase_for_epoch(epoch)
    if phase == "classifier":
        verify_receipt(model, receipt, identity)
        if entering_classifier:
            runner._set_determinism(CLASSIFIER_SEED)
    model.set_phase(phase)
    model.train()
    if phase == "classifier" and (model.base_model.training or any(p.requires_grad for p in model.base_model.parameters())):
        raise RuntimeError("classifier stage failed to freeze base parameters and BatchNorm buffers")


def classifier_probe_indices(training):
    """Fixed32+32 TRAIN rows, selected by labels only, never by model scores."""
    if training.split != "train":
        raise ValueError("classifier input probe is TRAIN-only")
    by_label = {0: [], 1: []}
    for index, row in enumerate(training.rows):
        label = int(row["label"])
        if label not in by_label:
            raise ValueError("probe requires binary TRAIN labels")
        if len(by_label[label]) < 32:
            by_label[label].append(index)
        if all(len(v) == 32 for v in by_label.values()):
            return by_label[1] + by_label[0]
    raise ValueError("TRAIN probe requires at least32 positives and32 negatives")


def summarize_classifier_probe(rows, head_kind):
    groups = {}
    for label, name in ((1, "positive"), (0, "negative")):
        selected = [r for r in rows if r["label"] == label]
        if len(selected) != 32:
            raise ValueError("classifier probe must cover exactly32+32 TRAIN rows")
        group = dict(pair_count=len(selected), valid_token_count_a=sum(r["valid_token_count_a"] for r in selected),
            valid_token_count_b=sum(r["valid_token_count_b"] for r in selected),
            all_features_finite=all(r["features_finite"] for r in selected))
        if head_kind == "matrix_cnn":
            group.update(raw_q_max=max(r["raw_q_max"] for r in selected),
                raw_positive_cell_count=sum(r["raw_positive_cell_count"] for r in selected),
                raw_above_threshold_cells=sum(r["raw_above_threshold_cells"] for r in selected),
                after_morph_above_threshold_cells=sum(r["after_morph_above_threshold_cells"] for r in selected),
                empty_binary_pair_count=sum(r["after_morph_above_threshold_cells"] == 0 for r in selected),
                nonempty_binary_pair_count=sum(r["after_morph_above_threshold_cells"] > 0 for r in selected))
        groups[name] = group
    failed = not all(g["all_features_finite"] for g in groups.values())
    reason = "nonfinite_valid_token_features" if failed else None
    if any(min(r["valid_token_count_a"], r["valid_token_count_b"]) < 4 for r in rows):
        failed, reason = True, "insufficient_valid_token_features"
    if head_kind == "matrix_cnn" and groups["positive"]["nonempty_binary_pair_count"] == 0:
        failed, reason = True, "all_32_positive_thresholded_matrices_empty"
    return groups, reason


def ensure_classifier_input_probe(model, training, identity, receipt, device, args, root):
    """Bounded no-grad TRAIN evidence check, persisted once before formal C13.

    No threshold search or held-out data. A failed probe stops classification;
    it does not change tau or discard training pairs. A new tau requires a new
    explicitly registered run identity. RNG and training modes are restored.
    """
    from staging.pairwise_v0_2.models.rachel_decoupled_score import antidiagonal_opening
    verify_receipt(model, receipt, identity)
    path = root / "classifier_input_probe.json"
    binding = dict(schema_version=PROBE_SCHEMA, resume_identity_sha256=canonical_digest(identity),
        matcher_state_sha256=receipt["base_state_sha256"], head_kind=model.head_kind,
        matrix_threshold=model.matrix_threshold)
    if path.exists():
        existing = json.loads(path.read_text())
        if any(existing.get(k) != v for k, v in binding.items()):
            raise FailedClassifierInputProbe("existing input probe is bound to a different model/data/tau")
        if existing.get("status") != "complete":
            raise FailedClassifierInputProbe("registered TRAIN input probe failed; no automatic threshold change")
        return existing
    indices = classifier_probe_indices(training)
    rng, previous_training = capture_rng_state(), model.training
    rows = []
    result = dict(**binding, population="TRAIN only: first32 positive and first32 negative rows",
        source_indices=indices, sample_count=64, no_grad=True, automatic_threshold_change=False,
        held_out_used=False, probe_used_for_model_selection=False,
        threshold_note="The configured Sinkhorn tau (default .006) is not numerically equivalent to original dual-softmax .006")
    try:
        model.eval()
        loader = make_weathering_loader(training, indices, batch_size=args.microbatch,
            num_workers=args.workers, seed=SEED, contour_cap=model.config.contour_cap)
        with torch.no_grad():
            for wrapped in loader:
                inputs, targets = runner._full_batch(wrapped.batch, device)
                output = model.base_model(*inputs)
                for i, pair_id in enumerate(wrapped.batch.pair_ids):
                    va, vb = inputs[4][i], inputs[5][i]
                    fa, fb = output.token_features_a[i][va], output.token_features_b[i][vb]
                    row = dict(pair_id=pair_id, label=int(targets[0][i].item()),
                        valid_token_count_a=int(va.sum().item()), valid_token_count_b=int(vb.sum().item()),
                        features_finite=bool(torch.isfinite(fa).all().item() and torch.isfinite(fb).all().item()))
                    if model.head_kind == "matrix_cnn":
                        raw = output.assignment[i][va][:, vb]
                        if not raw.numel() or not torch.isfinite(raw).all().item():
                            raise FailedClassifierInputProbe("empty/nonfinite valid Q in TRAIN probe")
                        morphed = antidiagonal_opening(raw)
                        binary = model.score_head.binary_matrix(output.assignment[i], va, vb)
                        after = int(torch.count_nonzero(binary).item())
                        if after != int((morphed > model.matrix_threshold).sum().item()):
                            raise FailedClassifierInputProbe("probe morphology differs from classifier actual input")
                        row.update(raw_q_max=float(raw.max().item()),
                            raw_positive_cell_count=int((raw > 0).sum().item()),
                            raw_above_threshold_cells=int((raw > model.matrix_threshold).sum().item()),
                            after_morph_above_threshold_cells=after, matrix_shape=list(raw.shape))
                    rows.append(row)
        if len(rows) != 64 or len({r["pair_id"] for r in rows}) != 64:
            raise FailedClassifierInputProbe("probe did not produce exactly64 unique TRAIN predictions")
        groups, reason = summarize_classifier_probe(rows, model.head_kind)
        result.update(groups=groups, rows=rows, failure_reason=reason,
                      status="failed_input_probe" if reason else "complete")
        save_json(path, result)
        if reason:
            raise FailedClassifierInputProbe(reason)
        return result
    except BaseException as error:
        if not path.exists():
            save_json(path, dict(result, status="failed_input_probe", failure_reason=repr(error), rows=rows))
        raise
    finally:
        model.train(previous_training)
        restore_rng_state(rng)


def train_segment(model, loader, optimizer, loss_config, device, args, epoch):
    phase = phase_for_epoch(epoch)
    optimizer.zero_grad(set_to_none=True)
    physical, effective = physical_batch_size(args), effective_batch_size(args)
    samplewise = getattr(args, "physical_microbatch", None) is not None
    if physical <= 0 or effective < physical or effective % physical:
        raise ValueError("physical batch must be positive and divide the runtime effective batch")
    if samplewise and args.microbatch != 1:
        raise ValueError("samplewise physical batching requires logical microbatch1")
    if loader.batch_size != physical:
        raise ValueError("loader physical batch differs from the registered runtime batch")
    accumulation = effective // physical
    samples, updates, valid_count = 0, 0, 0
    sums = {k: 0. for k in ("total",) + LOSS_NAMES}
    # Opt-in path transfers detached diagnostics only at log intervals/segment
    # end. It never changes loss/gradient scaling, clipping, or optimizer order.
    device_sums = torch.zeros(len(sums) + 1, dtype=torch.float64, device=device) if samplewise else None

    def read_diagnostics():
        nonlocal valid_count
        if device_sums is not None:
            values = device_sums.cpu().tolist()
            sums.update(zip(sums, values[:-1]))
            valid_count = int(values[-1])

    statistics, started = WeatheringStatistics(), time.perf_counter()
    for step, wrapped in enumerate(loader):
        group_start = step // accumulation * accumulation
        group_samples = min(effective, len(loader.dataset) - group_start * physical)
        inputs, targets = runner._full_batch(wrapped.batch, device)
        output = model(*inputs)
        pose = torch.as_tensor(wrapped.pose_supervision_enabled, dtype=torch.bool, device=device)
        total, components = phase_loss(output, targets, pose, loss_config, phase, samplewise=samplewise)
        count = len(wrapped.batch.pair_ids)
        (total * (count / group_samples)).backward()
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                          5., error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        samples += count
        values = torch.stack([total.detach()] + [components[k].detach() for k in LOSS_NAMES])
        if device_sums is not None:
            device_sums[:-1].add_(values.to(torch.float64), alpha=count)
            device_sums[-1].add_(output.training_valid.sum())
        else:
            valid_count += int(output.training_valid.sum().item())
            for key, value in zip(sums, values.cpu().tolist()):
                sums[key] += float(value) * count
        statistics.add(wrapped)
        if (step + 1) % args.log_every == 0:
            read_diagnostics()
            event(Path(args.output), event="train_progress", phase=phase, epoch=epoch,
                  samples=samples, optimizer_updates=updates, loss_components={k: v / samples for k, v in sums.items()})
    read_diagnostics()
    result = dict(samples=samples, optimizer_updates=updates, phase=phase, valid_pair_count=valid_count,
        loss_components={k: v / max(1, samples) for k, v in sums.items()},
        # Raw matcher BCE diagnostics may be nonzero but ALL have zero weight.
        pair_bce_weight=0. if phase == "matcher" else 1.,
        weathering=statistics.report(), elapsed_s=time.perf_counter() - started,
        peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
    if samplewise:
        result["runtime_batching"] = dict(physical_microbatch=physical, effective_batch=effective,
            logical_microbatch=1, samplewise_loss_schema=SAMPLEWISE_LOSS_SCHEMA,
            bitwise_trajectory_equivalence_claimed=False)
    return result


def winner_record(root, epoch, report, points, selection):
    key = "recall_95" if selection == "recall95" else "max_f1"
    return dict(selection=selection, selected_epoch=epoch, selected_global_exposure=epoch * TRAIN_COUNT,
        checkpoint=str(root / ("epoch_%03d.pt" % epoch)), checkpoint_sha256=None,
        validation_predictions=str(root / ("validation_%03d_rows.json" % epoch)),
        classifier_thresholds=report["thresholds"], operating_points=points,
        primary_pair_threshold=points["thresholds"][key], validation=report,
        selection_key=points["selection_key"] if selection == "recall95" else report["selection_key"],
        test_or_real_or_ood_used_for_fit=False)


def update_winners(winners, *, root, epoch, report, points):
    result = dict(winners)
    if epoch < 13 or report["decision_coverage"] != 1.:
        return result
    for name in ("max_f1", "recall95"):
        current = winner_record(root, epoch, report, points, name)
        if name not in result or tuple(current["selection_key"]) > tuple(result[name]["selection_key"]):
            result[name] = current
    if epoch == 20:
        result["fixed_epoch"] = winner_record(root, epoch, report, points, "fixed_epoch")
    return result


def publish_freezes(root, epoch, winners, identity):
    if epoch < 13:
        return
    required = {"max_f1", "recall95"} | ({"fixed_epoch"} if epoch == 20 else set())
    if set(winners) != required:
        raise ValueError("no complete fully-covered VAL selection")
    records = {name: dict(value, checkpoint_sha256=_sha256(value["checkpoint"])) for name, value in winners.items()}
    freeze = dict(schema_version=SCHEMA, status="complete" if epoch == 20 else "provisional",
        budget_epochs=epoch, budget_exposures=epoch * TRAIN_COUNT,
        matcher_epochs=12, classifier_epochs=epoch - 12, primary_selection="fixed_epoch",
        eligible_epoch_range=[13, epoch], selections=records,
        resume_identity=identity, resume_identity_sha256=canonical_digest(identity), selection_population="clean SIM VAL3000 only",
        held_out_used_for_fit=False, no_GT_layout_used_for_selection=True)
    save_json(root / "classifier_freezes" / "freeze.json", freeze)


def validate_arguments(args):
    resolve_runtime_arguments(args)
    depth = getattr(args, "cross_attention_depth", 1)
    if type(depth) is not int or depth not in (1, 2, 4):
        raise ValueError("cross-attention depth must be1,2,or4")
    if args.head_kind != "cross_attention" and depth != 1:
        raise ValueError("cross-attention depth requires the cross_attention head")
    revision = getattr(args, "matrix_head_revision", None)
    if revision is not None:
        from staging.pairwise_v0_2.models.rachel_decoupled_score import MATRIX_HEAD_REVISIONS
        if args.head_kind != "matrix_cnn" or revision not in MATRIX_HEAD_REVISIONS:
            raise ValueError("matrix-head revision requires matrix_cnn and a registered revision")
    if not 1 <= args.stop_after_epoch <= 20 or args.workers < 0 or args.log_every <= 0:
        raise ValueError("invalid stop/workers/log interval")
    if args.microbatch not in (1, 2, 4, 8, 16) or args.effective_batch != 16 or 16 % args.microbatch:
        raise ValueError("effective batch must remain16; microbatch must divide16")
    if effective_batch_size(args) != 16:
        raise ValueError("formal runtime-effective-batch migration is not registered; effective batch must remain16")
    if getattr(args, "physical_microbatch", None) is not None and (
            args.microbatch != 1 or physical_batch_size(args) not in FORMAL_PHYSICAL_MICROBATCHES):
        raise ValueError("physical migration accepts1/2/4/8/16 only, with logical microbatch1")
    if args.resume and args.smoke:
        raise ValueError("discard smoke cannot resume")
    if args.matcher_checkpoint and args.stop_after_epoch < 12:
        raise ValueError("reusing M12 cannot stop before epoch12")
    if args.matcher_checkpoint and args.smoke and args.smoke_phase != "classifier":
        raise ValueError("an imported M12 discard smoke must be classifier-only")
    if args.smoke_phase != "both" and not args.smoke:
        raise ValueError("smoke-phase only applies to discard smoke")


def run(args):
    validate_arguments(args)
    if platform.system() != "Linux" or not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("formal/discard-smoke entry requires the remote Linux CUDA server")
    torch.set_num_threads(1)
    training, validation, cap, records = make_populations(args)
    source_path = Path(args.checkpoint).resolve(strict=True)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    model, _, loss_config, digests = build_random_decoupled(source, args.head_kind, args.matrix_threshold, cap,
        matrix_head_revision=getattr(args, "matrix_head_revision", None),
        cross_attention_depth=getattr(args, "cross_attention_depth", 1))
    del source
    identity = experiment_identity(args, source_path, model, loss_config, records, digests)
    root = Path(args.output).resolve()
    if args.resume:
        if not root.is_dir():
            raise ValueError("resume directory does not exist")
    else:
        root.mkdir(parents=True, exist_ok=False)
    args.output = str(root)
    with run_lock(root):
        return _run_locked(args, root, model, training, validation, loss_config, identity)


def _run_locked(args, root, model, training, validation, loss_config, identity):
    device = torch.device(args.device)
    model = model.to(device)
    optimizer = create_optimizer(model)
    completed, receipt, winners = 0, None, {}
    restored_checkpoint, source_sha, origin_protocol = None, None, None
    if (root / "protocol.json").exists():
        origin_protocol = json.loads((root / "protocol.json").read_text())
    if args.resume:
        checkpoint = torch.load(root / "last.pt", map_location="cpu", weights_only=False)
        completed, receipt, winners = restore_training_state(model, optimizer, checkpoint, identity)
        restored_checkpoint = {"runtime_batching": checkpoint.get("runtime_batching")}
        if getattr(args, "physical_microbatch", None) is not None:
            source_sha = _sha256(root / "last.pt")
        del checkpoint
        if completed > args.stop_after_epoch * 4:
            raise ValueError("requested stop precedes the committed checkpoint")
        if completed % 4 == 0:
            publish_freezes(root, completed // 4, winners, identity)
    elif args.matcher_checkpoint:
        checkpoint = torch.load(args.matcher_checkpoint, map_location="cpu", weights_only=False)
        receipt = import_matcher(model, checkpoint, identity)
        completed = 48
        source_sha = _sha256(args.matcher_checkpoint)
        del checkpoint
    runtime_batching = prepare_runtime_batching(args, identity, completed,
        checkpoint=restored_checkpoint, source_checkpoint_sha256=source_sha, origin_protocol=origin_protocol)
    protocol = dict(**identity, status="running", arguments=vars(args), plan=segment_plan(),
        requested_stop_after_epoch=args.stop_after_epoch, implementation_sha256=_sha256(__file__),
        smoke=bool(args.smoke), formal_training_counted=not bool(args.smoke),
        runtime=dict(torch=torch.__version__, python=platform.python_version(), cuda=torch.version.cuda))
    if runtime_batching is not None:
        protocol["runtime_batching"] = runtime_batching
        save_json(root / "runtime_batching_history.json", runtime_batching)
    save_json(root / "protocol.json", protocol)
    if not args.smoke and (not args.resume or runtime_batching is not None):
        initial = checkpoint_payload(model, optimizer, identity=identity,
            loss_config=loss_config, completed=completed, receipt=receipt, winners=winners,
            role="runtime_migration_recovery" if args.resume else "initial_recovery",
            runtime_batching=runtime_batching)
        if args.matcher_checkpoint and not args.resume:
            # Preserve the existing downstream S4 M12 path without retraining
            # M. This is a typed new-run anchor: exact source base/receipt, but
            # fresh current-revision head, empty optimizer, and no C winners.
            runner._atomic_torch_save(root / "epoch_012.pt",
                dict(initial, checkpoint_role="imported_matcher_anchor"))
        runner._atomic_torch_save(root / "last.pt", initial)
        del initial
    if receipt:
        save_json(root / "matcher_pretraining_receipt.json", receipt)
    event(root, event="resume" if args.resume else "start", completed_segments=completed,
          total_budget="12M+8C", inherited_M12=bool(args.matcher_checkpoint))
    started = time.monotonic()
    try:
        plan = segment_plan()
        smoke_reports = {}
        if args.smoke:
            # Bounded M then C wiring: N exposures in each phase; no formal M12
            # receipt or claim of pretraining. All resulting weights discarded.
            phases = ("matcher", "classifier") if args.smoke_phase == "both" else (args.smoke_phase,)
            plan = [dict(plan[(0 if phase == "matcher" else 12) * 4], number=completed + i + 1)
                    for i, phase in enumerate(phases)]
        for segment in plan:
            number, epoch = segment["number"], segment["epoch"]
            if not args.smoke and (number <= completed or epoch > args.stop_after_epoch):
                continue
            if args.smoke:
                if args.matcher_checkpoint:
                    # Same frozen M12 and fresh C RNG as the repaired formal
                    # run. No source head/optimizer/winners/RNG are restored.
                    configure_phase(model, epoch, identity, receipt, entering_classifier=True)
                else:
                    model.set_phase(segment["phase"])
                    model.train()
                smoke_base_digest = state_digest(model.base_model)
            else:
                if number == 49:
                    ensure_classifier_input_probe(model, training, identity, receipt, device, args, root)
                configure_phase(model, epoch, identity, receipt, entering_classifier=number == 49)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(epoch)
            order = runner.epoch_indices(TRAIN_COUNT, seed=SEED, epoch=epoch, limit=None)
            count = args.smoke or SEGMENT_SIZE
            indices = order[segment["offset"]:segment["offset"] + count]
            loader = make_weathering_loader(training, indices, batch_size=physical_batch_size(args),
                num_workers=args.workers, seed=SEED + number, contour_cap=model.config.contour_cap)
            save_json(root / "status.json", dict(status="running", pid=os.getpid(), phase=segment["phase"],
                epoch=epoch, segment=number, global_exposure=completed * SEGMENT_SIZE))
            torch.cuda.reset_peak_memory_stats(device)
            report = train_segment(model, loader, optimizer, loss_config, device, args, epoch)
            if (report["samples"], report["optimizer_updates"]) != (count, count // args.effective_batch):
                raise RuntimeError("actual segment exposure/update count differs")
            save_json(root / ("segment_%03d.json" % number), dict(segment=segment, training=report))
            if args.smoke:
                if segment["phase"] == "classifier" and state_digest(model.base_model) != smoke_base_digest:
                    raise RuntimeError("discard classifier smoke modified frozen base tensors/buffers")
                smoke_reports[segment["phase"]] = report
                continue
            if segment["phase"] == "classifier":
                verify_receipt(model, receipt, identity)
            if segment["epoch_complete"]:
                if epoch == 12:
                    receipt = matcher_receipt(model, identity)
                if epoch >= 13:
                    from experiments.rachel_n512_formal_30k.recall_operating_points import fit_operating_points
                    save_json(root / "status.json", dict(status="running", pid=os.getpid(), phase="validation",
                        epoch=epoch, global_exposure=number * SEGMENT_SIZE))
                    val_loader = make_ablation_loader(validation, list(range(VAL_COUNT)),
                        batch_size=args.microbatch, num_workers=args.workers, seed=SEED,
                        contour_cap=model.config.contour_cap)
                    val_report, rows = evaluate_pair_validation(model, val_loader, device)
                    points = fit_operating_points([r["label"] for r in rows], [r["classification"]["fused"] for r in rows])
                    save_json(root / ("validation_%03d_rows.json" % epoch), rows)
                    save_json(root / ("validation_%03d.json" % epoch), dict(epoch=epoch,
                        validation=val_report, operating_points=points, selection_eligible=True,
                        global_exposure=number * SEGMENT_SIZE))
                    winners = update_winners(winners, root=root, epoch=epoch, report=val_report, points=points)
                    event(root, event="validation_complete", epoch=epoch, validation=val_report)
            payload = checkpoint_payload(model, optimizer, identity=identity, loss_config=loss_config,
                completed=number, receipt=receipt, winners=winners,
                role="epoch_anchor" if segment["epoch_complete"] else "recovery",
                runtime_batching=runtime_batching)
            if segment["epoch_complete"]:
                runner._atomic_torch_save(root / ("epoch_%03d.pt" % epoch), payload)
            runner._atomic_torch_save(root / "last.pt", payload)
            completed = number
            del payload
            if receipt:
                save_json(root / "matcher_pretraining_receipt.json", receipt)
            if segment["epoch_complete"]:
                publish_freezes(root, epoch, winners, identity)
                event(root, event="epoch_committed", epoch=epoch, phase=segment["phase"], global_exposure=completed * SEGMENT_SIZE)
        if args.smoke:
            result = dict(status="smoke_complete", weights_discarded=True, formal_training_counted=False,
                smoke_phase=args.smoke_phase, training=smoke_reports,
                total_discard_pair_exposures=sum(r["samples"] for r in smoke_reports.values()),
                total_discard_optimizer_updates=sum(r["optimizer_updates"] for r in smoke_reports.values()),
                no_pretraining_receipt_created=True)
            save_json(root / "smoke.json", result)
            save_json(root / "status.json", result)
            protocol.update(result)
            save_json(root / "protocol.json", protocol)
            return result
        if completed != args.stop_after_epoch * 4:
            raise RuntimeError("requested complete-epoch budget was not reached")
        result = dict(status="complete" if completed == 80 else "budget_complete", pid=os.getpid(),
            epoch=completed // 4, completed_segments=completed, global_exposure=completed * SEGMENT_SIZE,
            optimizer_updates=completed * SEGMENT_SIZE // args.effective_batch,
            matcher_pretraining_complete=receipt is not None, phase="train_val_complete",
            can_resume_to_epoch=20 if completed < 80 else None, elapsed_s=time.monotonic() - started)
        protocol.update(result)
        save_json(root / "protocol.json", protocol)
        save_json(root / "status.json", result)
        return result
    except BaseException as error:
        failure_status = ("failed_input_probe" if isinstance(error, FailedClassifierInputProbe) else
                          "interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
        save_json(root / "status.json", dict(status=failure_status,
            error=repr(error), completed_segments=completed, resumable_exposure=completed * SEGMENT_SIZE))
        raise


def parser():
    from staging.pairwise_v0_2.models.rachel_decoupled_score import MATRIX_HEAD_REVISIONS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Full24 architecture/loss metadata only")
    p.add_argument("--dataset", required=True)
    p.add_argument("--train-materialized-manifest", required=True)
    p.add_argument("--head-kind", choices=("matrix_cnn", "cross_attention"), required=True)
    p.add_argument("--cross-attention-depth", type=int, choices=(1, 2, 4), default=1,
        help="Classifier-only bidirectional attention depth;1 preserves the historical S4 architecture")
    p.add_argument("--matrix-threshold", type=float, default=.006)
    p.add_argument("--matrix-head-revision", choices=MATRIX_HEAD_REVISIONS,
        help="Explicit matrix-head architecture; omission preserves the registered default")
    p.add_argument("--sampling", choices=("original512", "paired512", "step3"), default="original512")
    p.add_argument("--density-train-manifest")
    p.add_argument("--clean-val-manifest")
    p.add_argument("--matcher-checkpoint", help="exact complete new-protocol M12, never old S2")
    p.add_argument("--output", required=True)
    p.add_argument("--stop-after-epoch", type=int, default=20)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--microbatch", type=int, default=1)
    p.add_argument("--effective-batch", type=int, default=16)
    p.add_argument("--physical-microbatch", type=int,
        help="Opt-in logical-micro1 batching migration; CLI overrides N512-only " + PHYSICAL_BATCH_ENV)
    p.add_argument("--runtime-effective-batch", type=int,
        help="Formal migration currently requires16; larger values are discard-benchmark-only")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--smoke", type=int, nargs="?", const=32, choices=(16, 32, 64))
    p.add_argument("--smoke-phase", choices=("both", "matcher", "classifier"), default="both")
    return p


if __name__ == "__main__":
    run(parser().parse_args())

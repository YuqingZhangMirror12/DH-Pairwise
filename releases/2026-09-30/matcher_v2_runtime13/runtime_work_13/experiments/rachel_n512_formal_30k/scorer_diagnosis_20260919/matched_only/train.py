"""Finite fresh Scorer C1..C16 from COMPLETE frozen S7 M12 caches only.

No Matcher/checkpoint inference, online augmentation, REAL/OOD or queue editing.
Import is inert. Formal CLI requires exclusive remote CUDA; CPU unit tests call
the numerical helpers only. Each 6000-pair commit saves Adam and full RNG.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.nn import functional as F

from . import cache, data, model as architecture, stage_cache
from experiments.rachel_n512_formal_30k import run_layout_decoder_experiment as metrics
from experiments.rachel_n512_formal_30k.train_joint_damage import state_digest
from experiments.rachel_n512_formal_30k.train_edge_weathering import capture_rng_state, restore_rng_state
from experiments.rachel_n512_formal_30k.train_score_design import run_lock
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.candidate_local import train as lock_owner
from staging.pairwise_v0_2.models import rachel_decoupled_score as ca_source
from staging.pairwise_v0_2.training import rachel_n512_runner as runner

SCHEMA = "s7-m12-fresh-matched-scorer-training/1"
HEAD_SEED, DATA_SEED = 260914, 260913
BATCH, SEGMENT_SIZE, HEAD_EPOCHS = 16, 6000, 16
TRAIN_COUNT, VAL_COUNT = 24000, 3000
SEGMENTS_PER_EPOCH, TOTAL_SEGMENTS = 4, 64
ARMS = architecture.ARMS + tuple(stage_cache.STAGES)


class CandidateStageScorer(torch.nn.Module):
    """Shared matched_edges head per proposal; pair BCE through amax only.

    Input features retain frozen Matcher global context. This wrapper performs
    no geometric search: all proposal masks/translations came from side-cache.
    No-group fallback is the SAME learned scalar as the final-edge head.
    """
    def __init__(self,arm,seed=HEAD_SEED,feature_dim=96,num_heads=4):
        super().__init__()
        if arm not in stage_cache.STAGES: raise ValueError("unknown stage arm")
        self.arm=arm
        self.edge_head=architecture.make_fresh_scorer("matched_edges",seed=seed,
            feature_dim=feature_dim,num_heads=num_heads)

    def metadata(self):
        result=self.edge_head.metadata()
        result.update(schema_version="stage-group-shared-edge-ca/1",arm=self.arm,
            underlying_head="matched_edges",stage_config=stage_cache.CONFIG,
            groups=1 if self.arm=="edge_seed" else 5,head_shared_across_groups=True,
            pair_readout="amax(group logits); PairBCE through same amax; ties share gradient",
            eligibility="present >=3 edges, including ambiguous/proposal; never a target",
            no_group="learnable matched_edges.head.no_evidence_logit; no global rescue",
            labels="original pair labels unchanged even when proposed groups are wrong",
            selector="one-time target-blind side-cache; never final-inlier union across modes",
            capacity_control="same matched_edges parameters/initialization for seed, multi and final; only evidence/readout differs",
            multiple_testing_caveat="max over more candidate groups can also raise negative-pair scores",
            stage_meaning=("best-support raw seed BEFORE refinement" if self.arm=="edge_seed"
                           else "up to5 separate refined modes; not a union"))
        return result

    def forward(self,tokens_a,tokens_b,valid_a,valid_b,selection=None,*,groups,
                candidate_weights=None,points_a_rc=None,points_b_rc=None):
        architecture._features(tokens_a,tokens_b,valid_a,valid_b,self.edge_head.feature_dim)
        architecture.validate_selection(selection,valid_a,valid_b)
        if not isinstance(groups,stage_cache.StageGroups) or groups.stage!=self.arm:
            raise ValueError("exact stage group object required, not a target mask")
        batch,slots=len(tokens_a),1 if self.arm=="edge_seed" else 5
        c=selection.candidate_indices.shape[1]
        bool_fields=(groups.present,groups.eligible)
        count_fields=(groups.ranks,groups.seed_candidate_id,groups.inlier_count,groups.status_code)
        all_tensors=(groups.candidate_inliers,groups.translation_rc,*bool_fields,*count_fields)
        if (any(not isinstance(v,torch.Tensor) or v.device!=tokens_a.device for v in all_tensors)
                or groups.candidate_inliers.dtype!=torch.bool
                or groups.candidate_inliers.shape!=(batch,slots,c)
                or groups.translation_rc.shape!=(batch,slots,2)
                or not groups.translation_rc.is_floating_point()
                or any(v.shape!=(batch,slots) or v.dtype!=torch.bool for v in bool_fields)
                or any(v.shape!=(batch,slots) or v.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64)
                       for v in count_fields)):
            raise ValueError("stage group tensor schema/dtype/device differs")
        counts=groups.candidate_inliers.sum(-1)
        if (not torch.equal(groups.inlier_count.long(),counts)
                or not torch.equal(groups.eligible,groups.present & (counts>=3))
                or (groups.candidate_inliers & ~selection.candidate_valid[:,None,:]).any()
                or (groups.candidate_inliers & ~groups.present[:,:,None]).any()
                or not torch.isfinite(groups.translation_rc[groups.present]).all()
                or ((groups.ranks<=0) & groups.present).any()):
            raise ValueError("stage membership/eligibility/translation/rank inconsistent")
        # Only gathered rows for real groups enter the shared edge head. Absent
        # slots do not create fake evidence or waste CA on padded groups.
        logits=tokens_a.new_full((batch,slots),float("nan"))
        for slot in range(slots):
            ids=torch.nonzero(groups.eligible[:,slot],as_tuple=False).flatten()
            if not len(ids): continue
            active=groups.candidate_inliers[ids,slot]
            indices=selection.candidate_indices[ids]
            valid=selection.candidate_valid[ids]
            ma=torch.zeros_like(valid_a[ids]); mb=torch.zeros_like(valid_b[ids])
            bi=torch.arange(len(ids),device=ids.device)[:,None].expand_as(active)[active]
            edges=indices[active]
            ma[bi,edges[:,0]]=True; mb[bi,edges[:,1]]=True
            group_selection=architecture.CandidateSelection(ma,mb,
                torch.ones(len(ids),device=ids.device,dtype=torch.bool),groups.translation_rc[ids,slot],
                indices,valid,active,tuple("stage_proposal_not_gt" for _ in range(len(ids))))
            output=self.edge_head(tokens_a[ids],tokens_b[ids],valid_a[ids],valid_b[ids],group_selection,
                candidate_weights=None if candidate_weights is None else candidate_weights[ids],
                points_a_rc=None if points_a_rc is None else points_a_rc[ids],
                points_b_rc=None if points_b_rc is None else points_b_rc[ids])
            logits[ids,slot]=output.logit
        has_group=groups.eligible.any(1)
        safe=torch.where(groups.eligible,logits,torch.full_like(logits,-torch.finfo(logits.dtype).max))
        top=safe.amax(1)
        probability_logit=torch.where(has_group,top,self.edge_head.head.no_evidence_logit.expand(batch))
        chosen=safe.argmax(1)
        ranks=groups.ranks.gather(1,chosen[:,None]).squeeze(1)
        return SimpleNamespace(logit=probability_logit,used_fallback=~has_group,
            has_raw_candidates=selection.candidate_valid.any(1),
            has_decoded_candidate=selection.layout_valid,has_selected_candidate_group=has_group,
            group_logits=logits,group_ranks=groups.ranks,group_eligible=groups.eligible,
            group_present=groups.present,group_status_code=groups.status_code,
            group_inlier_count=groups.inlier_count,group_seed_candidate_id=groups.seed_candidate_id,
            group_translation_rc=groups.translation_rc,
            selected_group_rank=torch.where(has_group,ranks,torch.zeros_like(ranks)))


def make_scorer(arm,seed=HEAD_SEED):
    if arm in architecture.ARMS:
        return architecture.make_fresh_scorer(arm,seed=seed)
    return CandidateStageScorer(arm,seed=seed)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def learning_rate(head_epoch):
    if type(head_epoch) is not int or not 1 <= head_epoch <= HEAD_EPOCHS:
        raise ValueError("head epoch must be1..16")
    return 1e-4 if head_epoch <= 3 else 2e-5


def plan():
    return [dict(number=(e-1)*SEGMENTS_PER_EPOCH+i+1, head_epoch=e, absolute_epoch=12+e,
                 offset=i*SEGMENT_SIZE, count=SEGMENT_SIZE, epoch_complete=i==SEGMENTS_PER_EPOCH-1,
                 learning_rate=learning_rate(e))
            for e in range(1, HEAD_EPOCHS+1) for i in range(SEGMENTS_PER_EPOCH)]


def pair_loss(logits, labels, training_valid):
    if (logits.ndim != 1 or labels.shape != logits.shape or training_valid.shape != logits.shape
            or not len(logits) or training_valid.dtype != torch.bool
            or not torch.isfinite(logits).all() or not torch.isfinite(labels).all()
            or not ((labels == 0) | (labels == 1)).all()):
        raise ValueError("finite batch logits, binary labels and bool training_valid required")
    values = F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype), reduction="none")
    # Deliberately mean over ALL pair rows, not renormalized by valid count.
    return (values * training_valid.to(values.dtype)).mean()


def create_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)


def implementation_binding():
    files = dict(train=__file__, data=data.__file__, model=architecture.__file__,
                 ca=ca_source.__file__, runner=runner.__file__,
                 metrics=metrics.__file__,
                 operating_points=Path(metrics.__file__).with_name("recall_operating_points.py"),
                 gpu_lock=lock_owner.__file__,
                 selector=sys.modules[architecture.select_predicted_inliers.__module__].__file__,
                 rng=sys.modules[capture_rng_state.__module__].__file__,
                 state_digest=sys.modules[state_digest.__module__].__file__,
                 run_lock=sys.modules[run_lock.__module__].__file__,
                 stage_cache=stage_cache.__file__,candidate_groups=stage_cache.candidate_groups.__file__)
    return {name: data.sha(path) for name, path in files.items()}


def identity(arm, model, training, validation):
    if len(training) != TRAIN_COUNT or len(validation) != VAL_COUNT:
        raise ValueError("formal training requires full24K/3K caches")
    if {r["pair_id"] for r in training.records} & {r["pair_id"] for r in validation.records}:
        raise ValueError("TRAIN/VAL cache pair IDs overlap")
    return dict(schema=SCHEMA, arm=arm, model=model.metadata(), initial_state_sha256=state_digest(model),
        cache_bindings=dict(train=training.binding, val=validation.binding),
        source_checkpoint_sha256=cache.SOURCE_SHA, source_matcher_epochs=12,
        matcher_frozen=True, source_matcher_pair_exposures=288000,
        training_count=TRAIN_COUNT, validation_count=VAL_COUNT, classifier_epochs=HEAD_EPOCHS,
        retained_endpoints=[8, 16], primary_endpoint=16, absolute_epochs=[13, 28],
        physical_microbatch=BATCH, effective_batch=BATCH, accumulation_steps=1,
        head_seed=HEAD_SEED, data_seed=DATA_SEED, segment_size=SEGMENT_SIZE,
        optimizer="AdamW", lr_by_head_epoch=[learning_rate(e) for e in range(1, HEAD_EPOCHS+1)],
        weight_decay=1e-4, grad_clip_norm=5., precision="fp32", loss="mean(PairBCE * training_valid) over ALL batch rows",
        decision_valid_role="reported, never used as positive label or loss denominator",
        selection_population="clean SIMVAL3000 only", real_ood_used=False,
        primary_selection="fixed C16; C8 retained learning-curve endpoint",
        auxiliary_selection="within each budget: max-F1/AP; P@Recall95/AP; earliest exact tie",
        numerical_contract="identical CPU-precomputed FP32 Matcher features across arms; old per-pair CA implementation retained",
        runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda, python=platform.python_version()),
        implementation_sha256=implementation_binding())


def train_segment(model, dataset, indices, optimizer, device):
    if not len(indices) or len(indices) % BATCH:
        raise ValueError("training segment must contain full batches16")
    model.train()
    started = time.perf_counter()
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    valid_count = torch.zeros((), dtype=torch.long, device=device)
    fallback_count = torch.zeros((), dtype=torch.long, device=device)
    for start in range(0, len(indices), BATCH):
        batch = dataset.batch(indices[start:start+BATCH], device)
        optimizer.zero_grad(set_to_none=True)
        output = model(*batch.model_args, **batch.model_kwargs)
        loss = pair_loss(output.logit, batch.labels, batch.training_valid)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite Scorer loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        loss_sum += loss.detach().double() * len(batch.labels)
        valid_count += batch.training_valid.sum()
        fallback_count += output.used_fallback.sum()
    optimizer.zero_grad(set_to_none=True)
    return dict(samples=len(indices), optimizer_updates=len(indices)//BATCH,
        mean_loss=float(loss_sum.cpu())/len(indices), training_valid_count=int(valid_count.cpu()),
        fallback_count=int(fallback_count.cpu()), elapsed_s=time.perf_counter()-started,
        physical_microbatch=BATCH, effective_batch=BATCH)


@torch.no_grad()
def evaluate(model, validation, device):
    # Imported only for evaluation; numerical/restore helpers do not need the
    # historical sklearn dependency. Formal runs use its original implementation.
    from experiments.rachel_n512_formal_30k.recall_operating_points import fit_operating_points
    model.eval()
    started, rows, loss_sum = time.perf_counter(), [], 0.
    for start in range(0, len(validation), BATCH):
        batch = validation.batch(range(start, min(start+BATCH, len(validation))), device)
        output = model(*batch.model_args, **batch.model_kwargs)
        loss = pair_loss(output.logit, batch.labels, batch.training_valid)
        loss_sum += float(loss) * len(batch.labels)
        # Historical FrozenMatcherScoreModel masks !training_valid logits to
        # zero before sigmoid. Keep this convention (score=.5, NOT zero), while
        # exposing raw head logits separately instead of hiding the fallback.
        deployed_logit = torch.where(batch.training_valid, output.logit, torch.zeros_like(output.logit))
        probabilities = deployed_logit.sigmoid().cpu().tolist()
        logits, raw_logits = deployed_logit.cpu().tolist(), output.logit.cpu().tolist()
        labels, training_valid, decision_valid, fallback, candidates = [v.cpu().tolist() for v in
            (batch.labels, batch.training_valid, batch.decision_valid, output.used_fallback, output.has_decoded_candidate)]
        for i, pair_id in enumerate(batch.pair_ids):
            row=dict(pair_id=pair_id, label=int(labels[i]), logit=logits[i], raw_head_logit=raw_logits[i], score=probabilities[i],
                training_valid=training_valid[i], decision_valid=decision_valid[i],
                used_fallback=fallback[i], has_decoded_candidate=candidates[i])
            if hasattr(output,"group_logits"):
                row["selected_group_rank"]=int(output.selected_group_rank[i])
                row["has_selected_candidate_group"]=bool(output.has_selected_candidate_group[i])
                row["groups"]=[]
                for slot in range(output.group_logits.shape[1]):
                    eligible=bool(output.group_eligible[i,slot])
                    present=bool(output.group_present[i,slot])
                    group_logit=float(output.group_logits[i,slot]) if eligible else None
                    row["groups"].append(dict(slot=slot,present=present,eligible=eligible,
                        rank=int(output.group_ranks[i,slot]),logit=group_logit,
                        score=float(torch.sigmoid(output.group_logits[i,slot])) if eligible else None,
                        seed_candidate_id=int(output.group_seed_candidate_id[i,slot]),
                        inlier_count=int(output.group_inlier_count[i,slot]),
                        status=stage_cache.STATUS_NAMES[int(output.group_status_code[i,slot])],
                        translation_a_to_b_rc=output.group_translation_rc[i,slot].cpu().tolist() if present else None))
            rows.append(row)
    labels, scores = [r["label"] for r in rows], [r["score"] for r in rows]
    op = fit_operating_points(labels, scores)
    best = op["validation"]["max_f1"]
    return dict(sample_count=len(rows), positive_count=sum(labels), mean_loss=loss_sum/len(rows),
        training_valid_count=sum(r["training_valid"] for r in rows),
        decision_valid_count=sum(r["decision_valid"] for r in rows),
        fallback_count=sum(r["used_fallback"] for r in rows), operating_points=op,
        metrics_at_0_5=metrics.classification(labels, scores, .5),
        max_f1_selection_key=[best["f1"], best["auprc"]], recall95_selection_key=op["selection_key"],
        score_policy="historical validity masking: !training_valid logit=0 then sigmoid=.5; ALL VAL rows calibrated; raw_head_logit also retained",
        elapsed_s=time.perf_counter()-started), rows


def update_winners(winners, head_epoch, report):
    result = deepcopy(winners)
    # Match the historical auxiliary selector's full decision-coverage guard.
    if report["decision_valid_count"] != report["sample_count"]:
        return result
    for name, key in (("max_f1", "max_f1_selection_key"), ("recall95", "recall95_selection_key")):
        if name not in result or tuple(report[key]) > tuple(result[name]["selection_key"]):
            result[name] = dict(head_epoch=head_epoch, absolute_epoch=12+head_epoch,
                selection_key=report[key], primary_pair_threshold=report["operating_points"]["thresholds"]["max_f1" if name=="max_f1" else "recall_95"],
                checkpoint="head_epoch_%03d.pt" % head_epoch,
                validation="validation_head_%03d.json" % head_epoch)
    return result


def checkpoint_payload(model, optimizer, ident, completed, winners, role):
    return dict(schema=SCHEMA, identity=ident, model_metadata=model.metadata(),
        model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        optimizer_state_dict=optimizer.state_dict(), rng_state=capture_rng_state(),
        optimizer_parameter_steps={name: (float(optimizer.state[p]["step"])
            if p in optimizer.state and "step" in optimizer.state[p] else None)
            for name, p in model.named_parameters()},
        completed_segments=completed, head_epoch=(completed+SEGMENTS_PER_EPOCH-1)//SEGMENTS_PER_EPOCH,
        absolute_epoch=12+(completed+SEGMENTS_PER_EPOCH-1)//SEGMENTS_PER_EPOCH,
        classifier_pair_exposures=completed*SEGMENT_SIZE, optimizer_updates=completed*SEGMENT_SIZE//BATCH,
        role=role, winners=deepcopy(winners), matcher_updated=False, phase="classifier", formal_training_counted=True)


def restore(model, optimizer, saved, ident):
    n = saved.get("completed_segments")
    if (saved.get("schema") != SCHEMA or saved.get("identity") != ident
            or saved.get("model_metadata") != model.metadata() or saved.get("matcher_updated") is not False
            or saved.get("phase") != "classifier" or saved.get("formal_training_counted") is not True
            or type(n) is not int or not 0 <= n <= TOTAL_SEGMENTS
            or saved.get("head_epoch") != (n+SEGMENTS_PER_EPOCH-1)//SEGMENTS_PER_EPOCH
            or saved.get("absolute_epoch") != 12+(n+SEGMENTS_PER_EPOCH-1)//SEGMENTS_PER_EPOCH
            or saved.get("classifier_pair_exposures") != n*SEGMENT_SIZE
            or saved.get("optimizer_updates") != n*SEGMENT_SIZE//BATCH
            or not isinstance(saved.get("rng_state"), dict)
            or set(saved["rng_state"]) != {"python", "numpy", "torch", "cuda"}
            or "optimizer_state_dict" not in saved):
        raise ValueError("strict Scorer resume identity/progress/state mismatch")
    model.load_state_dict(saved["model_state_dict"], strict=True)
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    if len(optimizer.param_groups) != 1:
        raise ValueError("fresh Scorer requires one AdamW parameter group")
    steps = [float(s["step"]) for s in optimizer.state.values() if "step" in s]
    # Fallback and edge branches are conditional, so their Adam steps may be
    # smaller. They are restored verbatim, never padded to total update count.
    expected = n*SEGMENT_SIZE//BATCH
    # No single parameter need participate in EVERY batch: an all-fallback
    # batch and an all-evidence batch can touch disjoint parameter subsets.
    # The committed explicit update ledger is authoritative; per-parameter
    # counters only constrain the ledger, they cannot reconstruct it.
    if (expected == 0 and optimizer.state) or (expected > 0 and (not steps
            or min(steps) <= 0 or any(x > expected or x != int(x) for x in steps))):
        raise ValueError("Adam step counters do not match committed budget")
    actual_steps = {}
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            raise ValueError("nonfinite restored head parameter")
        state = optimizer.state.get(p)
        actual_steps[name] = float(state["step"]) if state is not None and "step" in state else None
        if state is not None and (not {"step", "exp_avg", "exp_avg_sq"} <= set(state)
                or any(state[k].shape != p.shape or not torch.isfinite(state[k]).all()
                       for k in ("exp_avg", "exp_avg_sq"))):
            raise ValueError("Adam moments missing/nonfinite/wrong shape")
    if saved.get("optimizer_parameter_steps") != actual_steps:
        raise ValueError("per-parameter Adam ledger differs")
    group = optimizer.param_groups[0]
    expected_lr = 1e-4 if n == 0 else learning_rate((n+SEGMENTS_PER_EPOCH-1)//SEGMENTS_PER_EPOCH)
    if group["lr"] != expected_lr or group["weight_decay"] != 1e-4 or tuple(group["betas"]) != (.9,.999) or group["eps"] != 1e-8:
        raise ValueError("Adam configuration changed")
    restore_rng_state(saved["rng_state"])
    return n, deepcopy(saved["winners"])


def freeze_endpoint(root, ident, head_epoch, winners, report):
    endpoint = dict(head_epoch=head_epoch, absolute_epoch=12+head_epoch,
        checkpoint="head_epoch_%03d.pt" % head_epoch,
        primary_pair_threshold=report["operating_points"]["thresholds"]["max_f1"],
        operating_points=report["operating_points"])
    selections = {"fixed_endpoint": endpoint, **deepcopy(winners)}
    for name, row in selections.items():
        row["checkpoint_sha256"] = data.sha(root / row["checkpoint"])
        validation_path = root / ("validation_head_%03d.json" % row["head_epoch"])
        validation = json.loads(validation_path.read_text())
        row["validation"] = validation_path.name
        row["validation_sha256"] = data.sha(validation_path)
        row["operating_points"] = validation["operating_points"]
        operating_name = "recall_95" if name == "recall95" else "max_f1"
        if row["primary_pair_threshold"] != row["operating_points"]["thresholds"][operating_name]:
            raise ValueError("winner threshold differs from selected SIMVAL report")
    save(root / "freezes" / ("c%d.json" % head_epoch), dict(schema=SCHEMA, status="complete_endpoint",
        budget_head_epochs=head_epoch, primary=(head_epoch==HEAD_EPOCHS),
        identity=ident, selection_population="clean SIMVAL3000 only", real_ood_used=False,
        primary_selection="fixed_endpoint", selections=selections))


def execute(args, model, optimizer, training, validation, ident, device):
    """Caller owns output-directory and GPU locks. Used by focused CPU tests."""
    root = Path(args.output)
    completed, winners = 0, {}
    target = args.stop_after_head_epoch*SEGMENTS_PER_EPOCH
    if args.resume:
        saved = torch.load(root / "last.pt", map_location="cpu", weights_only=False)
        completed, winners = restore(model, optimizer, saved, ident)
        if completed > target:
            raise ValueError("cannot resume backwards to an earlier endpoint")
    elif (root / "last.pt").exists():
        raise ValueError("new run cannot overwrite existing checkpoint")
    protocol = dict(schema=SCHEMA, status="running", identity=ident, arguments=vars(args),
        plan=plan(), completed_segments=completed, resumed_from_segments=completed,
        started_unix=time.time(), elapsed_scope="current invocation only", real_ood_used=False)
    save(root / "protocol.json", protocol)
    save(root / "status.json", dict(status="running", completed_segments=completed))
    if not args.resume:
        runner._atomic_torch_save(root / "last.pt", checkpoint_payload(model, optimizer, ident, 0, {}, "initial"))
    started = time.perf_counter()
    try:
        # last.pt is the atomic training commit. An interruption just after
        # it was written must not strand C8/C16 publication; rebuilding these
        # derived manifests performs no extra updates or validation inference.
        for endpoint_epoch in (8, 16):
            if completed >= endpoint_epoch*SEGMENTS_PER_EPOCH:
                endpoint_payload = torch.load(root / ("head_epoch_%03d.pt" % endpoint_epoch),
                                              map_location="cpu", weights_only=False)
                endpoint_report = json.loads((root / ("validation_head_%03d.json" % endpoint_epoch)).read_text())
                if (endpoint_payload.get("identity") != ident
                        or endpoint_payload.get("completed_segments") != endpoint_epoch*SEGMENTS_PER_EPOCH):
                    raise ValueError("committed endpoint anchor identity differs")
                freeze_endpoint(root, ident, endpoint_epoch, endpoint_payload["winners"], endpoint_report)
        for segment in plan():
            number, e = segment["number"], segment["head_epoch"]
            if number <= completed or e > args.stop_after_head_epoch:
                continue
            for group in optimizer.param_groups:
                group["lr"] = segment["learning_rate"]
            order = runner.epoch_indices(len(training), seed=DATA_SEED, epoch=segment["absolute_epoch"], limit=None)
            indices = order[segment["offset"]:segment["offset"]+segment["count"]]
            report = train_segment(model, training, indices, optimizer, device)
            if report["samples"] != SEGMENT_SIZE or report["optimizer_updates"] != SEGMENT_SIZE//BATCH:
                raise ValueError("segment exposure/update count differs")
            save(root / ("segment_%03d.json" % number), dict(segment=segment, training=report))
            if segment["epoch_complete"]:
                # Evaluation consumes no randomness; keep the exact future train
                # RNG anyway so additional logging cannot change continuation.
                rng = capture_rng_state()
                try:
                    val_report, rows = evaluate(model, validation, device)
                finally:
                    restore_rng_state(rng)
                save(root / ("validation_head_%03d.json" % e), val_report)
                save(root / ("validation_head_%03d_rows.json" % e), rows)
                winners = update_winners(winners, e, val_report)
            payload = checkpoint_payload(model, optimizer, ident, number, winners,
                "epoch_anchor" if segment["epoch_complete"] else "segment_recovery")
            if segment["epoch_complete"]:
                runner._atomic_torch_save(root / ("head_epoch_%03d.pt" % e), payload)
            runner._atomic_torch_save(root / "last.pt", payload)
            completed = number
            if segment["epoch_complete"] and e in (8, 16):
                freeze_endpoint(root, ident, e, winners, val_report)
            state = dict(status="running", completed_segments=completed, head_epoch=e,
                absolute_epoch=segment["absolute_epoch"], classifier_pair_exposures=completed*SEGMENT_SIZE,
                optimizer_updates=completed*SEGMENT_SIZE//BATCH, elapsed_s=time.perf_counter()-started)
            save(root / "status.json", state)
            protocol.update(state); save(root / "protocol.json", protocol)
            if segment["epoch_complete"]:
                print(json.dumps(dict(event="head_epoch_complete", arm=args.arm, head_epoch=e,
                    validation_f1=val_report["operating_points"]["validation"]["max_f1"]["f1"])), flush=True)
        if completed != target:
            raise ValueError("requested committed training budget incomplete")
        result = dict(status="complete" if completed==TOTAL_SEGMENTS else "paused", completed_segments=completed,
            completed_head_epochs=completed//SEGMENTS_PER_EPOCH, absolute_epoch=12+completed//SEGMENTS_PER_EPOCH,
            classifier_pair_exposures=completed*SEGMENT_SIZE, optimizer_updates=completed*SEGMENT_SIZE//BATCH,
            elapsed_s=time.perf_counter()-started, matcher_updated=False, real_ood_used=False)
        protocol.update(result); save(root / "protocol.json", protocol); save(root / "status.json", result)
        return result
    except BaseException as error:
        protocol.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=repr(error), completed_segments=completed, elapsed_s=time.perf_counter()-started,
            recovery="resume last.pt; repeat uncommitted segment only; never serialize partial updates as committed")
        save(root / "protocol.json", protocol); save(root / "status.json", protocol)
        raise


def run(args):
    if args.arm not in ARMS or args.stop_after_head_epoch not in (8, 16) or args.device != "cuda:0":
        raise ValueError("registered arms, C8/C16 endpoints and cuda:0 only")
    training, validation = data.FormalCache(args.train_cache, "train"), data.FormalCache(args.val_cache, "val")
    stage_paths=(getattr(args,"train_stage_cache",None),getattr(args,"val_stage_cache",None))
    if args.arm in stage_cache.STAGES:
        if not all(stage_paths): raise ValueError("stage arms require precomputed TRAIN and VAL side-caches")
        training=data.CandidateStageCache(training,stage_paths[0],args.arm)
        validation=data.CandidateStageCache(validation,stage_paths[1],args.arm)
    elif any(stage_paths):
        raise ValueError("original3 arms must not silently use a different stage side-cache")
    root = Path(args.output).resolve()
    input_roots=[training.root,validation.root]+[Path(p).resolve() for p in stage_paths if p]
    for source in input_roots:
        if root == source or source in root.parents or root in source.parents:
            raise ValueError("output must be separate from input cache trees")
    # Same child-owned, nonblocking lock as current C1/C2; never queue-jump.
    with lock_owner.gpu_lock():
        if platform.system() != "Linux" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("formal trainer requires remote Linux with exactly one exposed CUDA GPU")
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float32)
        runner._set_determinism(HEAD_SEED)
        model = make_scorer(args.arm, seed=HEAD_SEED)
        ident = identity(args.arm, model, training, validation)
        device = torch.device(args.device)
        model = model.to(device=device, dtype=torch.float32)
        optimizer = create_optimizer(model)
        if args.resume:
            if not root.is_dir():
                raise ValueError("resume output missing")
        else:
            root.mkdir(parents=True, exist_ok=False)
        args.output = str(root)
        with run_lock(root):
            return execute(args, model, optimizer, training, validation, ident, device)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--train-stage-cache",help="complete one-time stage side-cache; edge_seed/edge_multi only")
    p.add_argument("--val-stage-cache",help="complete one-time stage side-cache; edge_seed/edge_multi only")
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    p.add_argument("--stop-after-head-epoch", type=int, choices=(8, 16), default=16)
    p.add_argument("--resume", action="store_true")
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), sort_keys=True))

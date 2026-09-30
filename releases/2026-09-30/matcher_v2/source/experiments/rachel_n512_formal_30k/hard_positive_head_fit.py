"""CPU-only, TRAIN/VAL-only controls for hard positives and pose reliability.

Pairability keeps the original pair label even when its pose is wrong.  R uses
an explicitly separate, pose-recomputed feature vector and never gates P.
Sampling counts describe exposures, not newly generated examples.  The hard
and uniform augmented-positive draws are independent: their overlap is legal
and is reported, while replacement within each draw occurs only if necessary.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.rachel_n512_formal_30k import run_damage_separate_heads as damage

base, common, sealed = damage.base, damage.common, damage.sealed
FEATURE_NAMES, DECODER = damage.FEATURE_NAMES, damage.DECODER
SEED = 260911
OLD_PDAMAGE_THRESHOLD = 0.2178770750761032
PAIR_HEADS = ("pbase", "puniform", "phard")
RELIABILITY_HEADS = ("rnative", "rcandidate")
HEAD_NAMES = PAIR_HEADS + RELIABILITY_HEADS
SCHEMA = "rachel-hard-positive-head-fit/1"
LEARNING_RATE, WEIGHT_DECAY = 1e-3, 1e-4


def _features(rows, field="features"):
    values = np.asarray([row[field] for row in rows], dtype=np.float32)
    if values.shape != (len(rows), len(FEATURE_NAMES)) or not np.isfinite(values).all():
        raise ValueError(field + " must be finite and follow the 24-input schema")
    return torch.from_numpy(values)


def _labels(rows):
    if any(row.get("label") not in (False, True, 0, 1) for row in rows):
        raise ValueError("labels must be binary, not missing or inferred from pose")
    return np.asarray([row["label"] for row in rows], dtype=bool)


def _balanced_labels(labels):
    labels = np.asarray(labels, bool)
    positives = int(labels.sum())
    if (labels.ndim != 1 or not len(labels) or positives * 2 != len(labels)
            or positives % 4):
        raise ValueError("exact sampling quotas require balanced labels and each class divisible by 4")
    return labels


def _index_rows(rows, split):
    indexed = {}
    for row in rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id or pair_id in indexed:
            raise ValueError("TRAIN/VAL tables require nonempty unique pair IDs")
        if row.get("split", split) not in (("val", "validation") if split == "val" else ("train",)):
            raise ValueError("only the specified TRAIN/clean VAL populations may be used")
        indexed[pair_id] = row
    return indexed


def _aligned_inputs(clean_rows, base_rows, augmented_rows, candidate_rows, val_rows):
    clean, baseline, augmented, val = map(list, (clean_rows, base_rows, augmented_rows, val_rows))
    candidates = list(candidate_rows)
    maps = [_index_rows(rows, "train") for rows in (clean, baseline, augmented)]
    if not clean or any(set(m) != set(maps[0]) for m in maps[1:]):
        raise ValueError("the three TRAIN feature tables must have identical pair IDs")
    baseline = [maps[1][row["pair_id"]] for row in clean]
    augmented = [maps[2][row["pair_id"]] for row in clean]
    for left, middle, right in zip(clean, baseline, augmented):
        for key in ("pair_id", "fragment_a", "fragment_b", "label", "target_translation_rc"):
            if key not in left or key not in middle or key not in right:
                raise ValueError("TRAIN identity/label/GT field missing: " + key)
            if left[key] != middle[key] or left[key] != right[key]:
                raise ValueError("augmentation changed original TRAIN identity/label/GT: " + key)
    labels = _balanced_labels(_labels(clean))
    val_map = _index_rows(val, "val")
    if not val or set(val_map) & set(maps[0]) or len(np.unique(_labels(val))) != 2:
        raise ValueError("clean VAL must have both original labels and be disjoint from TRAIN")
    positive_ids = {row["pair_id"] for row in clean if row["label"]}
    if not candidates:
        raise ValueError("TRAIN true-pair pose candidates are required")
    for row in candidates:
        if (row.get("pair_id") not in positive_ids or row.get("split", "train") != "train"
                or not isinstance(row.get("kind"), str) or not row["kind"]):
            raise ValueError("every candidate must identify a TRAIN true pair and candidate kind")
    candidate_targets = _labels(candidates)
    if len(np.unique(candidate_targets)) != 2:
        raise ValueError("candidate training requires both pose-success and pose-failure pools")
    # Never append identities, targets, candidate kinds or old P scores to inputs.
    for rows in (clean, baseline, augmented, candidates, val):
        _features(rows)
    _features(augmented, "reliability_features")
    _features(val, "reliability_features")
    positive_augmented = [row for row in augmented if row["label"]]
    native_targets = damage.pose_targets(positive_augmented)
    damage.pose_targets([row for row in val if row["label"]])
    old_scores = np.asarray([row.get("old_pdamage_probability", np.nan) for row in augmented], float)
    if not np.isfinite(old_scores).all() or np.any((old_scores < 0) | (old_scores > 1)):
        raise ValueError("augmented TRAIN requires finite old_pdamage_probability in [0,1]")
    positive_indices = np.flatnonzero(labels)
    hard = positive_indices[(~native_targets) | (old_scores[positive_indices] < OLD_PDAMAGE_THRESHOLD)]
    if not len(hard):
        raise ValueError("hard-positive pool is empty; hard-positive treatment cannot run")
    return clean, baseline, augmented, candidates, val, hard


def _draw(pool, count, rng):
    pool = np.asarray(pool, dtype=np.int64)
    if not len(pool) and count:
        raise ValueError("cannot sample from an empty pool")
    replacement = len(pool) < count
    return rng.choice(pool, size=count, replace=replacement), replacement


def sample_epoch(labels, hard_pool_indices, candidate_targets, *, epoch):
    """Return deterministic indices and draw labels for exact epoch quotas.

P indices address concatenated [base_damage, augmented] tensors. Rnative
indices address augmented true pairs; Rcandidate indices address candidates.
This function handles sampling only; the fit validates source identities first.
"""
    labels = _balanced_labels(labels)
    candidates = np.asarray(candidate_targets, bool)
    hard = np.asarray(hard_pool_indices, dtype=np.int64)
    if (epoch < 1 or hard.ndim != 1 or not len(hard) or len(np.unique(hard)) != len(hard)
            or np.any(hard < 0) or np.any(hard >= len(labels)) or not labels[hard].all()):
        raise ValueError("hard-positive pool must contain unique eligible positive TRAIN indices")
    if candidates.ndim != 1 or len(np.unique(candidates)) != 2:
        raise ValueError("both candidate target classes are required")
    n, per_class = len(labels), int(labels.sum())
    half, hard_count = per_class // 2, per_class // 4
    rng = np.random.default_rng(SEED + 1000003 * epoch)
    pools = {False: np.flatnonzero(~labels), True: np.flatnonzero(labels)}
    bp, bp_replace = _draw(pools[True], half, rng)
    bn, bn_replace = _draw(pools[False], half, rng)
    ap, ap_replace = _draw(pools[True], half, rng)
    an, an_replace = _draw(pools[False], half, rng)
    hp, hp_replace = _draw(hard, hard_count, np.random.default_rng(SEED + 2000003 * epoch))
    cp, cp_replace = _draw(np.flatnonzero(candidates), half, rng)
    cn, cn_replace = _draw(np.flatnonzero(~candidates), half, rng)
    porder, rorder = rng.permutation(n), rng.permutation(per_class)

    def plan(indices, kinds, order, replacement):
        return dict(indices=np.asarray(indices, np.int64)[order],
            draw_kind=np.asarray(kinds)[order], replacement=replacement)

    shared_replacement = dict(base_positive=bp_replace, base_negative=bn_replace,
        uniform_augmented_positive=ap_replace, uniform_augmented_negative=an_replace)
    return {
        "pbase": plan(np.arange(n), ["base"] * n, porder, dict(base=False)),
        "puniform": plan(np.concatenate((bp, bn, ap + n, an + n)),
            ["base"] * per_class + ["uniform_augmented"] * per_class,
            porder, shared_replacement),
        "phard": plan(np.concatenate((bp, bn, ap[:hard_count] + n, hp + n, an + n)),
            ["base"] * per_class + ["uniform_augmented"] * hard_count
            + ["hard_augmented"] * hard_count + ["uniform_augmented"] * half,
            porder, dict(shared_replacement, hard_augmented_positive=hp_replace)),
        "rnative": plan(np.arange(per_class), ["native"] * per_class, rorder, dict(native=False)),
        "rcandidate": plan(np.concatenate((cp, cn)),
            ["candidate_success"] * half + ["candidate_failure"] * half,
            rorder, dict(candidate_success=cp_replace, candidate_failure=cn_replace)),
    }


def initialize_heads(clean_features, candidate_features):
    """Independent modules: shared P initialization/norm and shared R init/norm."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        reference = base.SeamGeometryPairHead()
    reference.set_training_normalization(clean_features)
    heads = {name: copy.deepcopy(reference) for name in PAIR_HEADS}
    reference.set_training_normalization(candidate_features)
    heads.update({name: copy.deepcopy(reference) for name in RELIABILITY_HEADS})
    return heads


def _sampling_receipt(name, plan, clean, augmented, candidates, hard):
    indices = plan["indices"]
    if name in PAIR_HEADS:
        n = len(clean)
        rows = [clean[int(i) % n] for i in indices]
        sources = np.where(indices < n, "base_damage", "augmented")
        eligible = (indices >= n) & np.isin(indices % n, hard)
        selected_rows = indices
    elif name == "rnative":
        positives = [row for row in augmented if row["label"]]
        rows = [positives[int(i)] for i in indices]
        sources = np.full(len(indices), "augmented_native")
        eligible, selected_rows = np.zeros(len(indices), bool), indices
    else:
        rows = [candidates[int(i)] for i in indices]
        sources = np.full(len(indices), "candidate")
        eligible, selected_rows = np.zeros(len(indices), bool), indices
    labels = _labels(rows) if name != "rnative" else damage.pose_targets(rows)
    source_counts = {}
    for source in sorted(set(sources)):
        mask = sources == source
        source_counts[str(source)] = dict(total=int(mask.sum()), positive=int((mask & labels).sum()),
            negative=int((mask & ~labels).sum()))
    unique_pairs = len({row["pair_id"] for row in rows})
    receipt = dict(pair_exposures=len(indices), unique_pair_count=unique_pairs,
        repeated_pair_exposures=len(indices) - unique_pairs,
        unique_feature_row_count=len(np.unique(selected_rows)),
        positive_target_exposures=int(labels.sum()), negative_target_exposures=int((~labels).sum()),
        by_source=source_counts, hard_draw_exposures=int((plan["draw_kind"] == "hard_augmented").sum()),
        hard_eligible_augmented_positive_exposures=int(eligible.sum()),
        replacement=plan["replacement"])
    if name == "rcandidate":
        receipt["by_candidate_kind"] = {kind: sum(row["kind"] == kind for row in rows)
            for kind in sorted({row["kind"] for row in rows})}
    return receipt, {row["pair_id"] for row in rows}


def fit_heads(clean_rows, base_damage_rows, augmented_rows, candidate_rows, val_rows, *,
              output, identity, epochs=10, batch_size=256, callback=None):
    if not isinstance(epochs, int) or epochs < 1 or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive integers")
    clean, baseline, augmented, candidates, val, hard = _aligned_inputs(
        clean_rows, base_damage_rows, augmented_rows, candidate_rows, val_rows)
    output = Path(output)
    if ((output / "heads.pt").exists() or (output / "validation_freeze.json").exists()
            or list(output.glob("epoch_*.json"))):
        raise FileExistsError("head fit artifacts already exist; load completed heads explicitly")
    identity = common.clean(copy.deepcopy(identity))
    rnames = identity.get("reliability_feature_names", list(FEATURE_NAMES))
    if (not isinstance(rnames, (list, tuple)) or len(rnames) != len(FEATURE_NAMES)
            or len(set(rnames)) != len(rnames) or any(not isinstance(n, str) or not n for n in rnames)):
        raise ValueError("identity reliability_feature_names must contain 24 unique names")
    output.mkdir(parents=True, exist_ok=True)
    labels, candidate_targets = _labels(clean), _labels(candidates)
    positive_augmented = [row for row in augmented if row["label"]]
    positive_val = [row for row in val if row["label"]]
    px = torch.cat((_features(baseline), _features(augmented)))
    py = torch.tensor(np.tile(labels, 2), dtype=torch.float32)
    nx = _features(positive_augmented, "reliability_features")
    ny = torch.tensor(damage.pose_targets(positive_augmented), dtype=torch.float32)
    cx, cy = _features(candidates), torch.tensor(candidate_targets, dtype=torch.float32)
    vx, rvx = _features(val), _features(positive_val, "reliability_features")
    vy, rvy = _labels(val), damage.pose_targets(positive_val)
    heads = initialize_heads(_features(clean), cx)
    optimizers = {name: torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY) for name, head in heads.items()}
    best, states, selected = {}, {}, {}
    steps = {name: 0 for name in HEAD_NAMES}
    totals = {name: dict(completed_pair_exposures=0, positive_target_exposures=0,
        negative_target_exposures=0, hard_draw_exposures=0,
        hard_eligible_augmented_positive_exposures=0, by_source={}) for name in HEAD_NAMES}
    observed = {name: set() for name in HEAD_NAMES}
    for epoch in range(1, epochs + 1):
        plans = sample_epoch(labels, hard, candidate_targets, epoch=epoch)
        record = dict(schema_version=SCHEMA, epoch=epoch, heads={})
        for name, head in heads.items():
            plan = plans[name]
            x, y = ((px, py) if name in PAIR_HEADS else ((nx, ny) if name == "rnative" else (cx, cy)))
            order, total_loss = plan["indices"], 0.
            head.train()
            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size]
                optimizers[name].zero_grad(set_to_none=True)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(head(x[idx]), y[idx])
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite independent head loss: " + name)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
                optimizers[name].step()
                steps[name] += 1
                total_loss += float(loss.detach()) * len(idx)
            with torch.inference_mode():
                scores = torch.sigmoid(head.eval()(vx if name in PAIR_HEADS else rvx)).numpy()
            if not np.isfinite(scores).all():
                raise RuntimeError("nonfinite VAL scores: " + name)
            targets = vy if name in PAIR_HEADS else rvy
            threshold = common.fit_threshold(targets, scores)
            metrics = (common.classification(targets, scores, threshold) if name in PAIR_HEADS else
                damage.reliability_metrics(positive_val, scores, threshold))
            ap = metrics.get("auprc")
            key = ((metrics["f1"], ap if ap is not None else -1.) if name in PAIR_HEADS else
                (-metrics["brier"], ap if ap is not None else -1.))
            if name not in best or key > best[name]:
                best[name], states[name] = key, copy.deepcopy(head.state_dict())
                selected[name] = dict(epoch=epoch, threshold=float(threshold), validation_metrics=metrics)
            sampling, pair_ids = _sampling_receipt(name, plan, clean, augmented, candidates, hard)
            observed[name].update(pair_ids)
            totals[name]["completed_pair_exposures"] += sampling["pair_exposures"]
            for field in ("positive_target_exposures", "negative_target_exposures", "hard_draw_exposures",
                          "hard_eligible_augmented_positive_exposures"):
                totals[name][field] += sampling[field]
            for source, counts in sampling["by_source"].items():
                dest = totals[name]["by_source"].setdefault(source, dict(total=0, positive=0, negative=0))
                for field, count in counts.items():
                    dest[field] += count
            record["heads"][name] = dict(train_loss=total_loss / len(order), sampling=sampling,
                pair_exposures=totals[name]["completed_pair_exposures"], optimizer_updates=steps[name],
                validation_metrics=metrics)
        common.write_json(output / ("epoch_%02d.json" % epoch), record)
        if callback:
            callback(record)
    for name, head in heads.items():
        head.load_state_dict(states[name], strict=True)
        head.eval().requires_grad_(False)
    budgets = {name: dict(completed_epochs=epochs, completed_optimizer_updates=steps[name],
        train_unique_pair_count=(len(clean) if name in PAIR_HEADS else
            (len(positive_augmented) if name == "rnative" else len({r["pair_id"] for r in candidates}))),
        actual_unique_pair_count=len(observed[name]), **totals[name]) for name in HEAD_NAMES}
    budgets["rcandidate"]["available_candidate_row_count"] = len(candidates)
    freeze = dict(schema_version=SCHEMA, status="complete", identity=identity,
        matcher_identity=identity, matcher_checkpoint_id=identity.get("checkpoint_sha256"),
        selected=selected, budgets=budgets, seed=SEED, precision="fp32", head_training_device="cpu",
        head_batch_size=batch_size, epochs=epochs, feature_names=list(FEATURE_NAMES),
        reliability_feature_names=list(rnames),
        feature_schema=dict(pairability=dict(field="features", names=list(FEATURE_NAMES), count=24),
            reliability=dict(native_field="reliability_features", candidate_field="features",
                names=list(rnames), count=24, semantics="target-blind evidence recomputed at the supplied pose")),
        selection_rules=dict(pairability="maximum original clean VAL equal-row F1, then AP, then earliest epoch",
            reliability="minimum original clean VAL true-pair native-pose Brier, then AP, then earliest epoch"),
        threshold_rule="selected head clean VAL target F1; largest threshold among ties; score >= threshold",
        pairability_targets="original binary pair labels, never pose-success labels",
        reliability_target="valid original predicted Top2 translation error <=10 canvas pixels on true pairs",
        candidate_target="provided TRAIN true-pair candidate pose success <=10px; not pairability",
        actual_hard_pool=dict(unique_pair_count=len(hard), pair_ids=[clean[int(i)]["pair_id"] for i in hard],
            old_pdamage_threshold=OLD_PDAMAGE_THRESHOLD,
            rule="augmented true pair with invalid native pose, error >10px, or old_pdamage_probability < threshold",
            actual_hard_draw_exposures=totals["phard"]["hard_draw_exposures"],
            actual_hard_eligible_augmented_positive_exposures=totals["phard"]["hard_eligible_augmented_positive_exposures"]),
        sampling_policy=dict(pairability_epoch_exposures=len(clean), reliability_epoch_exposures=len(positive_augmented),
            augmented_fraction_each_label=.5, hard_fraction_augmented_positives=.5,
            replacement="within a draw only when its pool is insufficient",
            overlap="independent hard/uniform draws may reuse the same pair; never counted as new data",
            comparison="puniform/phard share base draws and augmented-negative draws"),
        normalization_rules=dict(pairability="all clean TRAIN features only; identical P buffers",
            reliability="all TRAIN candidate feature rows, both candidate targets; identical R buffers"),
        pairability_shared_initialization=True, reliability_shared_initialization=True,
        optimizer="AdamW", learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        validation_sample_count=len(val), validation_positive_count=len(positive_val),
        native_train_target_positive_count=int(ny.sum()), native_validation_target_positive_count=int(rvy.sum()),
        candidate_target_counts=dict(positive=int(candidate_targets.sum()), negative=int((~candidate_targets).sum())),
        test_or_real_used_for_fit=False, reliability_gates_pairability=False,
        layout_modified=False, matcher_frozen=True, geometry_target_blind=True, selected_full_decoder=DECODER,
        heads={name: head.metadata() for name, head in heads.items()})
    checkpoint = dict(schema_version=SCHEMA, identity=identity, feature_names=list(FEATURE_NAMES),
        reliability_feature_names=list(rnames), selected=selected,
        heads={name: dict(config=asdict(head.config), state_dict=states[name]) for name, head in heads.items()})
    torch.save(checkpoint, output / "heads.pt")
    freeze["head_checkpoint_sha256"] = sealed._sha256_file(output / "heads.pt")
    common.write_json(output / "validation_freeze.json", freeze)
    return heads, freeze


def load_heads(output):
    """Load only a complete, hash-bound freeze; no training or threshold fitting."""
    output = Path(output)
    freeze = json.loads((output / "validation_freeze.json").read_text())
    if (freeze.get("schema_version") != SCHEMA or freeze.get("status") != "complete"
            or freeze.get("feature_names") != list(FEATURE_NAMES)
            or freeze.get("test_or_real_used_for_fit") is not False
            or freeze.get("reliability_gates_pairability") is not False
            or freeze.get("layout_modified") is not False
            or sealed._sha256_file(output / "heads.pt") != freeze.get("head_checkpoint_sha256")):
        raise ValueError("incomplete, changed or incompatible hard-positive head freeze")
    saved = sealed._torch_load_checkpoint(output / "heads.pt")
    for key in ("schema_version", "identity", "feature_names", "reliability_feature_names", "selected"):
        if saved.get(key) != freeze.get(key):
            raise ValueError("head checkpoint and freeze differ: " + key)
    if set(saved.get("heads", {})) != set(HEAD_NAMES) or set(freeze.get("budgets", {})) != set(HEAD_NAMES):
        raise ValueError("all five independent heads and budgets are required")
    heads = {}
    for name in HEAD_NAMES:
        config = base.SeamGeometryHeadConfig(**saved["heads"][name]["config"])
        if config != base.SeamGeometryHeadConfig():
            raise ValueError("head does not use the registered 24-input MLP")
        head = base.SeamGeometryPairHead(config)
        head.load_state_dict(saved["heads"][name]["state_dict"], strict=True)
        if any(not torch.isfinite(t).all() for t in head.state_dict().values()):
            raise ValueError("head checkpoint has nonfinite parameters/buffers")
        budget = freeze["budgets"][name]
        count = freeze["sampling_policy"]["pairability_epoch_exposures" if name in PAIR_HEADS else
            "reliability_epoch_exposures"]
        if (budget["completed_epochs"] != freeze["epochs"]
                or budget["completed_pair_exposures"] != freeze["epochs"] * count
                or budget["completed_optimizer_updates"] != freeze["epochs"] * math.ceil(count / freeze["head_batch_size"])):
            raise ValueError("head budget differs from its registered exposure schedule")
        heads[name] = head.eval().requires_grad_(False)
    return heads, freeze


def score_heads(heads, rows):
    """Attach P and R independently without reading GT or changing any layout."""
    if set(heads) != set(HEAD_NAMES):
        raise ValueError("all five independent heads are required")
    if not rows:
        return rows
    x, rx = _features(rows), _features(rows, "reliability_features")
    original = np.asarray([row["classification"]["fused"] for row in rows], float)
    if not np.isfinite(original).all() or np.any((original < 0) | (original > 1)):
        raise ValueError("original fused classification must be finite probabilities")
    with torch.inference_mode():
        scores = {name: torch.sigmoid(head.eval()(x if name in PAIR_HEADS else rx)).cpu().numpy()
            for name, head in heads.items()}
    if any(not np.isfinite(values).all() for values in scores.values()):
        raise ValueError("nonfinite frozen head predictions")
    for i, row in enumerate(rows):
        row["head_scores"] = {name: float(scores[name][i]) for name in PAIR_HEADS}
        row["head_scores"]["original_fused"] = float(original[i])
        row["pose_reliability"] = {name: dict(probability=float(scores[name][i])) for name in RELIABILITY_HEADS}
    return rows

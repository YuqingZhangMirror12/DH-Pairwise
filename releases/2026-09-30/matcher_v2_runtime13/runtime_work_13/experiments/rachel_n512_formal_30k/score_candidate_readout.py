"""Offline diagnostics of already-frozen P/R outputs; never change a decision.

P describes pair adjacency; R describes one proposed translation being correct.
Candidate correctness uses post-prediction GT and the registered 20 px radius.
It is an oracle diagnostic, not a rescue/reranking policy. OOD has no pose GT.
No torch, checkpoint loading, inference, threshold fitting or model selection.
"""
import math


TOLERANCE_PX = 20.0
DIAGNOSTIC_R_THRESHOLD = 0.5
QUANTILES = (0, .1, .25, .5, .75, .9, 1)


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(count=0, mean=None, quantile_levels=list(QUANTILES), quantiles=None)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        raise ValueError("nonfinite diagnostic values")
    def quantile(q):
        index = (len(values) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return dict(count=len(values), mean=sum(values) / len(values),
                quantile_levels=list(QUANTILES), quantiles=[quantile(q) for q in QUANTILES])


def _finite_point(point):
    return (isinstance(point, (list, tuple)) and len(point) == 2
            and all(isinstance(x, (int, float)) and math.isfinite(x) for x in point))


def _candidate_row(row, split, decoder, thresholds):
    details = row.get("candidate_details")
    if details is None:
        return None
    fields = ("valid", "translations_rc", "candidate_pair_probability", "candidate_correct_probability")
    if any(not isinstance(details.get(key), list) for key in fields):
        raise ValueError("candidate details missing registered arrays")
    size = len(details["valid"])
    if any(len(details[key]) != size for key in fields):
        raise ValueError("candidate details lengths disagree")
    if any(type(x) is not bool for x in details["valid"]):
        raise ValueError("candidate validity must be boolean")
    valid = [i for i, value in enumerate(details["valid"]) if value]
    p = details["candidate_pair_probability"]
    r = details["candidate_correct_probability"]
    translations = details["translations_rc"]
    for i in valid:
        if (not _finite_point(translations[i]) or any(not isinstance(scores[i], (int, float))
                or not math.isfinite(scores[i]) or not 0 <= scores[i] <= 1 for scores in (p, r))):
            raise ValueError("valid candidate has invalid probability/translation")
    positive = bool(row["label"])
    gt = row.get("target_translation_rc")
    if split == "ood" and gt is not None:
        raise ValueError("OOD pose GT must be unavailable")
    pose_known = split != "ood" and positive and _finite_point(gt)
    errors = {i: math.hypot(translations[i][0] - gt[0], translations[i][1] - gt[1])
              for i in valid} if pose_known else {}
    correct = [i for i in valid if errors.get(i, float("inf")) <= TOLERANCE_PX]
    # R=0 for a negative pair; positive pairs with missing GT are unscored.
    r_supervised = split != "ood" and (not positive or pose_known)
    best_p = max(valid, key=lambda i: p[i]) if valid else None
    best_r = max(valid, key=lambda i: r[i]) if valid else None
    source_layout = row.get("layouts", {}).get(decoder, {})
    canonical_error = source_layout.get("translation_l2_px")
    canonical_good = (bool(source_layout.get("valid")) and isinstance(canonical_error, (int, float))
                      and math.isfinite(canonical_error) and canonical_error <= TOLERANCE_PX) if pose_known else None
    rejected = {op: not (row["decision_valid"] and row["classification"]["fused"] >= threshold)
                for op, threshold in thresholds.items()}
    return dict(pair_id=row["pair_id"], positive=positive, valid_candidate_count=len(valid),
        valid_candidate_indices=valid, max_p=p[best_p] if best_p is not None else None,
        max_r=r[best_r] if best_r is not None else None, top_p_index=best_p, top_r_index=best_r,
        canonical_candidate_p=p[0] if 0 in valid else None,
        canonical_candidate_r=r[0] if 0 in valid else None,
        pose_gt_available=pose_known, oracle_has_correct_candidate=bool(correct) if pose_known else None,
        oracle_min_translation_error_px=min(errors.values()) if errors else None,
        top_r_correct=(best_r in correct) if pose_known else None,
        canonical_layout_correct=canonical_good, rejected=rejected,
        r_brier_pair_mean=(sum((r[i] - float(i in correct)) ** 2 for i in valid) / len(valid))
            if r_supervised and valid else None,
        diagnostic_high_r_wrong_count=sum(r[i] >= DIAGNOSTIC_R_THRESHOLD and i not in correct for i in valid)
            if r_supervised else None,
        r_correct_values=[r[i] for i in correct] if r_supervised else None,
        r_wrong_values=[r[i] for i in valid if i not in correct] if r_supervised else None)


def candidate_diagnostics(rows, *, split, decoder, thresholds):
    """One requested population. Include every pair in denominators, even no mode."""
    if split not in ("test", "real", "ood"):
        raise ValueError("only frozen held-out evaluation splits are supported")
    if set(thresholds) != {"max_f1", "recall_95"} or any(
            not isinstance(t, (int, float)) or not math.isfinite(t) or not 0 <= t <= 1
            for t in thresholds.values()):
        raise ValueError("requires both unchanged frozen thresholds")
    records = [item for row in rows if (item := _candidate_row(row, split, decoder, thresholds)) is not None]
    if records and len(records) != len(rows):
        raise ValueError("partial candidate details would shrink the population denominator")
    result = dict(status="available" if records else "unavailable", sample_count=len(rows),
        candidate_details_pair_count=len(records), missing_details_pair_count=len(rows) - len(records),
        reason=None if records else "No candidate-head outputs in supplied rows; original baseline normally has none.",
        pose_correctness_tolerance_px=TOLERANCE_PX if split != "ood" else None,
        diagnostic_r_threshold=DIAGNOSTIC_R_THRESHOLD,
        diagnostic_threshold_used_for_pair_decisions=False, threshold_fitting_performed=False,
        layout_reranking_performed=False, gt_used_for_prediction=False,
        caveat="Max R is a model output, not proof of reliable layout. Oracle GT diagnostics never rescue a pair.",
        valid_candidate_count=distribution([x["valid_candidate_count"] for x in records]),
        no_valid_candidate_pair_ids=[x["pair_id"] for x in records if not x["valid_candidate_count"]],
        pair_score_distributions={}, score_only_flags={}, pose_diagnostics=None, cases=records)
    for name, positive in (("positive", True), ("negative", False)):
        cohort = [x for x in records if x["positive"] == positive]
        result["pair_score_distributions"][name] = {key: distribution([x[key] for x in cohort if x[key] is not None])
            for key in ("max_p", "max_r", "canonical_candidate_p", "canonical_candidate_r")}
    for op in thresholds:
        result["score_only_flags"][op] = dict(
            rejected_positive_high_r_pair_ids=[x["pair_id"] for x in records if x["positive"]
                and x["rejected"][op] and x["max_r"] is not None and x["max_r"] >= DIAGNOSTIC_R_THRESHOLD],
            negative_high_r_pair_ids=[x["pair_id"] for x in records if not x["positive"]
                and x["max_r"] is not None and x["max_r"] >= DIAGNOSTIC_R_THRESHOLD],
            high_r_does_not_establish_pose_correctness=True)
    if split == "ood":
        result["pose_unavailable_reason"] = "No Turufan pose GT: no R correctness, oracle recall or layout accuracy."
        return result
    known = [x for x in records if x["pose_gt_available"]]
    oracle = [x for x in known if x["oracle_has_correct_candidate"]]
    # Equal pair weighting, as in the R objective; this is diagnostic Brier,
    # not a replay of training BCE or its unavailable training_valid mask.
    brier = [x["r_brier_pair_mean"] for x in records if x["r_brier_pair_mean"] is not None]
    pose = dict(positive_with_gt_count=len(known), positive_with_gt_without_candidate_count=sum(
        not x["valid_candidate_count"] for x in known),
        oracle_candidate_recall=sum(x["oracle_has_correct_candidate"] for x in known) / len(known) if known else None,
        top_r_correct_positive_recall=sum(x["top_r_correct"] for x in known) / len(known) if known else None,
        top_r_correct_given_oracle=sum(x["top_r_correct"] for x in oracle) / len(oracle) if oracle else None,
        oracle_correct_pair_ids=[x["pair_id"] for x in oracle],
        canonical_wrong_but_oracle_correct_pair_ids=[x["pair_id"] for x in oracle if not x["canonical_layout_correct"]],
        r_brier_pair_mean=distribution(brier), r_score_by_target_pair_mean={}, missed_evidence={})
    for name, key in (("correct_candidate", "r_correct_values"), ("wrong_candidate", "r_wrong_values")):
        pose["r_score_by_target_pair_mean"][name] = distribution([
            sum(x[key]) / len(x[key]) for x in records if x[key]])
    for op in thresholds:
        pose["missed_evidence"][op] = dict(
            false_negative_but_oracle_correct_pair_ids=[x["pair_id"] for x in oracle if x["rejected"][op]],
            false_negative_but_canonical_correct_pair_ids=[x["pair_id"] for x in known
                if x["rejected"][op] and x["canonical_layout_correct"]],
            false_negative_high_r_pair_ids=[x["pair_id"] for x in records if x["positive"]
                and x["rejected"][op] and x["max_r"] is not None and x["max_r"] >= DIAGNOSTIC_R_THRESHOLD],
            negative_high_r_pair_ids=[x["pair_id"] for x in records if not x["positive"]
                and x["max_r"] is not None and x["max_r"] >= DIAGNOSTIC_R_THRESHOLD])
    result["pose_diagnostics"] = pose
    return result

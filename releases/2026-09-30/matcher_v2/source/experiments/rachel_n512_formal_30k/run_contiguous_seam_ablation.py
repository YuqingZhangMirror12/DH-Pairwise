"""Validation-selected translation geometry with frozen pair scores.

Compare unchanged v2 Top2 with gap20/gap40 seam decoders on one shared forward.
Use an original completed --run, or an architecture --checkpoint together with
its already-frozen --pair-threshold. Test requires a complete validation freeze.
The optional physical reranking comparison is independent of seam continuity.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.utils.data import Subset

from experiments.rachel_n512_formal_30k import run_layout_decoder_experiment as common
from experiments.rachel_n512_formal_30k.resampled_input_support import make_ablation_loader
from experiments.rachel_n512_formal_30k.fragment_size_strata import pair_size_metadata, summarize_size_strata
from staging.pairwise_v0_2.models.contiguous_seam_layout import ContiguousSeamConfig, estimate_contiguous_seam_layout
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config, RachelN512Pairwise
from staging.pairwise_v0_2.models.rachel_model_factory import load_rachel_checkpoint
from staging.pairwise_v0_2.models.translation_layout import TranslationLayoutConfig, estimate_translation_layout
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset
from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed

SCHEMA_VERSION = "contiguous-seam-ablation/1"


def decoder_configs(include_mode_refinement=False, include_physical=False):
    if include_mode_refinement and include_physical:
        raise ValueError("keep refinement and physical reranking as separate ablations")
    configs = {
        "full_top2_mode": ("top2", TranslationLayoutConfig(correspondence_mode="topk_union", top_k=2,
            max_candidates=512, min_inliers=3, inlier_radius_px=10.)),
        "full_contiguous_seam_gap20": ("seam", ContiguousSeamConfig(max_gap_px=20.)),
        "full_contiguous_seam_gap40": ("seam", ContiguousSeamConfig(max_gap_px=40.)),
    }
    if include_mode_refinement:
        for gap in (20, 40):
            configs["full_contiguous_seam_gap%d_mode_refine" % gap] = (
                "seam", ContiguousSeamConfig(max_gap_px=float(gap), refinement_support="mode"))
    if include_physical:
        from staging.pairwise_v0_2.models.physical_translation_layout import PhysicalTranslationConfig
        configs = {"full_top2_mode": configs["full_top2_mode"]}
        for name, normals, overlap in (("normals", True, False), ("overlap", False, True), ("both", True, True)):
            configs["full_physical_" + name] = (
                "physical", PhysicalTranslationConfig(use_normals=normals, use_overlap=overlap))
    return configs


def load_model(args):
    if args.checkpoint:
        if args.pair_threshold is None or not np.isfinite(args.pair_threshold):
            raise ValueError("--checkpoint requires the frozen --pair-threshold from train_val_freeze.json")
        path = Path(args.checkpoint)
        checkpoint = sealed._torch_load_checkpoint(path)
        model = load_rachel_checkpoint(checkpoint)
        return model, {"checkpoint_sha256": sealed._sha256_file(path), "checkpoint_path": str(path.resolve()),
            "checkpoint_epoch": checkpoint.get("epoch"), "variant": checkpoint.get("variant"),
            "model_kind": checkpoint.get("model_kind", "full"),
            "resample_contour_cap": checkpoint.get("resample_contour_cap"),
            "original_fused_threshold": float(args.pair_threshold), "seed": args.seed,
            "precision": args.precision or "fp32"}
    if args.pair_threshold is not None:
        raise ValueError("--pair-threshold is only used with --checkpoint; --run keeps its frozen threshold")
    receipt, _, winners = sealed._freeze_completed_winners(Path(args.run or common.DEFAULT_RUN))
    full = next(w for w in winners if w.arm == "full_n512")
    return full.model, {"checkpoint_sha256": full.checkpoint_sha256, "checkpoint_path": str(full.checkpoint_path),
        "checkpoint_epoch": full.epoch, "variant": "frozen_full_n512",
        "original_fused_threshold": float(full.threshold.threshold), "seed": int(receipt["config"]["seed"]),
        "precision": args.precision or receipt["config"]["precision"]}


def load_geometry_freeze(path, identity):
    authority = json.loads(Path(path).read_text(encoding="utf-8"))
    if (authority.get("schema_version") != SCHEMA_VERSION or authority.get("source_split") != "validation"
            or authority.get("probe_only") is not False or authority.get("sample_count") != 3000
            or authority.get("test_or_real_used_for_fit") is not False):
        raise ValueError("test requires a complete non-probe validation-only seam freeze")
    for key in ("checkpoint_sha256", "original_fused_threshold", "precision"):
        if authority[key] != identity[key]:
            raise ValueError("validation freeze differs in " + key)
    if authority.get("resample_contour_cap") != identity.get("resample_contour_cap"):
        raise ValueError("validation freeze differs in contour resampling policy")
    configs = {}
    for name, value in authority["decoders"].items():
        kind = value["kind"]
        if kind not in ("top2", "seam", "physical"):
            raise ValueError("unknown decoder kind in freeze")
        if kind == "physical":
            from staging.pairwise_v0_2.models.physical_translation_layout import PhysicalTranslationConfig
            config_type = PhysicalTranslationConfig
        else:
            config_type = TranslationLayoutConfig if kind == "top2" else ContiguousSeamConfig
        configs[name] = (kind, config_type(**value["config"]))
    allowed = (set(decoder_configs()), set(decoder_configs(True)))
    if any(kind == "physical" for kind, _ in configs.values()):
        allowed += (set(decoder_configs(include_physical=True)),)
    if set(configs) not in allowed or authority["selected_full_decoder"] not in configs:
        raise ValueError("validation freeze does not contain a registered decoder comparison")
    return authority, configs


def decode_pair(points_a, points_b, valid_a, valid_b, assignment, configs, save_membership=False,
                *, mask_a=None, mask_b=None):
    layouts = {}
    for name, (kind, config) in configs.items():
        if kind == "physical":
            from staging.pairwise_v0_2.models.physical_translation_layout import estimate_physical_translation_layout
            if mask_a is None or mask_b is None:
                raise ValueError("physical reranking requires both input fragment masks")
            estimate = estimate_physical_translation_layout(points_a, points_b, assignment, valid_a, valid_b,
                mask_a=mask_a, mask_b=mask_b, config=config)
        else:
            decoder = estimate_translation_layout if kind == "top2" else estimate_contiguous_seam_layout
            estimate = decoder(points_a, points_b, assignment, valid_a, valid_b, config=config)
        diagnostics = asdict(estimate)
        diagnostics.pop("t_a_to_b_rc")
        if not save_membership:
            for key in ("candidate_indices", "inlier_mask", "seam_membership"):
                diagnostics.pop(key, None)
        layouts[name] = {"translation_rc": estimate.t_a_to_b_rc, "valid": bool(estimate.valid), "diagnostics": diagnostics}
    return layouts


def attach_seam_quality(layouts, points_a, points_b, valid_a, valid_b, target_a, target_b,
                        gt_translation, *, label, translation_valid, save_membership=False):
    """Score already-decoded memberships; targets never enter a decoder.

    The reference is the dataset's sampled correspondence targets, not an
    annotation of the entire physical seam. Full-original has no explicit
    emitted correspondence set, so no seam is inferred from its translation.
    """
    from experiments.rachel_n512_formal_30k.seam_correspondence_metrics import (
        gt_edges_from_targets, measure_seam_correspondences)
    gt_edges = gt_edges_from_targets(target_a, target_b, valid_a=valid_a, valid_b=valid_b)
    for layout in layouts.values():
        diagnostics = layout.get("diagnostics", {})
        membership_key = "seam_membership" if "seam_membership" in diagnostics else "inlier_mask"
        if "candidate_indices" not in diagnostics or membership_key not in diagnostics:
            layout["seam_quality_unavailable"] = "no_explicit_selected_correspondences"
            continue
        candidates = np.asarray(diagnostics["candidate_indices"], dtype=np.int64).reshape(-1, 2)
        membership = np.asarray(diagnostics[membership_key])
        if membership.dtype != np.bool_ or membership.shape != (len(candidates),):
            raise ValueError("decoder membership must be Boolean and align with candidates")
        layout["seam_quality"] = measure_seam_correspondences(
            points_a, points_b, candidates[membership], gt_edges,
            valid_a=valid_a, valid_b=valid_b,
            gt_translation_a_to_b_rc=gt_translation if translation_valid else None,
            label=bool(label), pose_valid=bool(layout["valid"]))
        if not save_membership:
            for key in ("candidate_indices", "inlier_mask", "seam_membership"):
                diagnostics.pop(key, None)


def run(args):
    if args.batch_size <= 0 or args.workers < 0 or args.limit < 0:
        raise ValueError("batch size must be positive; workers and limit must be nonnegative")
    if args.split == "test" and not args.freeze:
        raise ValueError("test requires --freeze from a completed validation run")
    torch.set_num_threads(1)
    model, identity = load_model(args)
    authority, configs = load_geometry_freeze(args.freeze, identity) if args.freeze else (
        None, decoder_configs(args.refinement_ablation, args.physical_ablation))
    if args.seam_quality and authority and authority.get("seam_quality_evaluated") is not True:
        raise ValueError("requested seam-quality test needs validation with the same measurement enabled")
    sealed._set_determinism(identity["seed"])
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    model = model.to(device).eval()
    dataset = RachelPairDataset(Path(args.dataset), args.split)
    if identity.get("resample_contour_cap"):
        from staging.pairwise_v0_2.pairwise_data.rachel_resampled_dataset import RachelResampledDataset
        dataset = RachelResampledDataset(dataset, contour_cap=identity["resample_contour_cap"])
    with (Path(args.dataset) / "pairs" / (args.split + ".jsonl")).open(encoding="utf-8") as stream:
        manifest = [json.loads(line) for line in stream if line.strip()]
    source_units = {r["pair_id"]: sorted({r["fragment_a"]["split_unit_id"], r["fragment_b"]["split_unit_id"]}) for r in manifest}
    if args.limit:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    if not len(dataset):
        raise ValueError("evaluation population is empty")
    loader = make_ablation_loader(dataset, tuple(range(len(dataset))), batch_size=args.batch_size,
        num_workers=args.workers, seed=identity["seed"], contour_cap=model.config.contour_cap)
    serialized_configs = {name: {"kind": kind, "config": asdict(config)} for name, (kind, config) in configs.items()}
    common.write_json(destination / "protocol.json", dict(identity, schema_version=SCHEMA_VERSION,
        split=args.split, population_size=len(dataset), decoders=serialized_configs, classifier_modified=False,
        all_decoders_share_one_forward=True, rotation_estimated=False, routing_used=False,
        mask_normals_used=False,
        contour_normals_used=any(getattr(config, "use_normals", False) for _, config in configs.values()),
        fragment_overlap_used=any(getattr(config, "use_overlap", False) for _, config in configs.values()),
        seam_quality_evaluated=bool(args.seam_quality),
        seam_quality_tolerances_px=[3.0, 5.0] if args.seam_quality else None,
        seam_quality_reference="sampled_dataset_correspondence_targets" if args.seam_quality else None,
        seam_quality_used_for_selection=False,
        probe_only=bool(args.limit)))
    rows, started = [], time.perf_counter()
    with (destination / "pair_results.jsonl").open("x", encoding="utf-8") as stream, torch.inference_mode():
        for batch_number, batch in enumerate(loader):
            tensors = [sealed._tensor(getattr(batch, name), device, dtype) for name, dtype in (
                ("mask_a", torch.float32), ("mask_b", torch.float32), ("points_rc_a", torch.float32),
                ("points_rc_b", torch.float32), ("contour_valid_a", torch.bool), ("contour_valid_b", torch.bool))]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=identity["precision"] == "bf16"):
                output = model(*tensors)
            values = {name: getattr(output, name).detach().float().cpu().numpy() for name in (
                "assignment", "translation_hat_rc", "coarse_probability", "local_probability", "fused_probability")}
            for i, pair_id in enumerate(batch.pair_ids):
                layouts = decode_pair(batch.points_rc_a[i], batch.points_rc_b[i], batch.contour_valid_a[i],
                    batch.contour_valid_b[i], values["assignment"][i], configs,
                    args.save_seam_membership or args.seam_quality,
                    mask_a=batch.mask_a[i], mask_b=batch.mask_b[i])
                layouts["full_original"] = {"translation_rc": values["translation_hat_rc"][i],
                                            "valid": bool(output.decision_valid[i].item())}
                # Every target-blind decoder above is complete before targets
                # enter metric calculation; no target influences a mode/chain.
                gt = np.asarray(batch.translation_a_to_b_rc[i], dtype=float)
                target_valid = bool(batch.translation_valid[i])
                if args.seam_quality:
                    attach_seam_quality(layouts, batch.points_rc_a[i], batch.points_rc_b[i],
                        batch.contour_valid_a[i], batch.contour_valid_b[i], batch.target_a[i], batch.target_b[i],
                        gt, label=bool(batch.labels[i]), translation_valid=target_valid,
                        save_membership=args.save_seam_membership)
                for layout in layouts.values():
                    translation = np.asarray(layout["translation_rc"], dtype=float)
                    layout["offset_b_in_a_rc"] = -translation
                    layout["translation_l2_px"] = float(np.linalg.norm(translation - gt)) if target_valid and layout["valid"] else None
                row = common.clean({"pair_id": pair_id, "source_unit_ids": source_units[pair_id],
                    "fragment_a": batch.fragment_a_tokens[i], "fragment_b": batch.fragment_b_tokens[i],
                    "label": bool(batch.labels[i]), "target_translation_rc": gt if target_valid else None,
                    "classification": {b: float(values[b + "_probability"][i]) for b in ("coarse", "local", "fused")},
                    "layouts": layouts})
                row.update(pair_size_metadata(batch.mask_a[i], batch.mask_b[i]))
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            if batch_number % 10 == 0 or len(rows) == len(dataset):
                print(json.dumps({"processed": len(rows), "total": len(dataset), "elapsed_s": round(time.perf_counter() - started, 2)}), flush=True)
    branch_thresholds = authority["branch_validation_thresholds"] if authority else {
        b: common.fit_threshold([r["label"] for r in rows], [r["classification"][b] for r in rows]) for b in ("coarse", "local", "fused")}
    summary = common.summarize(rows, identity["original_fused_threshold"], branch_thresholds)
    summary["size_strata"] = summarize_size_strata(rows, identity["original_fused_threshold"], branch_thresholds)
    if args.seam_quality:
        from experiments.rachel_n512_formal_30k.seam_correspondence_metrics import summarize_seam_correspondences
        summary["seam_quality"] = {name: summarize_seam_correspondences(
            [row["layouts"][name]["seam_quality"] for row in rows]) for name in configs}
        summary["seam_quality_unavailable"] = {"full_original": "no_explicit_selected_correspondences"}
    if not authority:
        def selection_key(name):
            metrics = summary["layout"][name]
            return (float(np.mean(list(metrics["recall"].values()))), metrics["assembly"]["10"]["f1"],
                    -metrics["p90_px_conditional"] if metrics["p90_px_conditional"] is not None else -np.inf)
        selected = max(configs, key=selection_key)
        authority = dict(identity, schema_version=SCHEMA_VERSION, source_split="validation", sample_count=len(rows),
            selected_full_decoder=selected, decoders=serialized_configs, branch_validation_thresholds=branch_thresholds,
            seam_quality_evaluated=bool(args.seam_quality), seam_quality_used_for_selection=False,
            selection_rule="maximize mean R@2/5/8/10, then assembly F1@10, then minimize conditional P90; ties preserve v2 first",
            test_or_real_used_for_fit=False, probe_only=bool(args.limit))
        common.write_json(destination / "validation_freeze.json", authority)
    summary.update(status="complete", split=args.split, selected_full_decoder=authority["selected_full_decoder"],
        checkpoint_sha256=identity["checkpoint_sha256"], precision=identity["precision"],
        elapsed_s=time.perf_counter() - started, original_classification_preserved=True,
        seam_quality_evaluated=bool(args.seam_quality), seam_quality_used_for_selection=False,
        all_decoders_share_one_forward=True, rotation_estimated=False, routing_used=False, probe_only=bool(args.limit))
    common.write_json(destination / "summary.json", summary)
    print(json.dumps(common.clean(summary), ensure_ascii=False), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--run", help="Original completed Full run; defaults to the frozen v2 run")
    model.add_argument("--checkpoint", help="Architecture winner.pt with model_config and model_state_dict")
    parser.add_argument("--pair-threshold", type=float, help="Frozen fused threshold; required with --checkpoint")
    parser.add_argument("--dataset", default=common.DEFAULT_DATA)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--freeze")
    parser.add_argument("--precision", choices=("fp32", "bf16"), help="Defaults to run receipt precision or FP32 for --checkpoint")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-seam-membership", action="store_true")
    parser.add_argument("--seam-quality", action="store_true",
                        help="Measure decoded matches against sampled dataset GT at fixed 3/5px tolerances; never used for selection")
    geometry = parser.add_mutually_exclusive_group()
    geometry.add_argument("--refinement-ablation", action="store_true",
                        help="Validation-only extension: same seam mode ranking, refine using all mode inliers")
    geometry.add_argument("--physical-ablation", action="store_true",
                        help="Independent validation comparison: Top2 versus normals/overlap/both mode reranking")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

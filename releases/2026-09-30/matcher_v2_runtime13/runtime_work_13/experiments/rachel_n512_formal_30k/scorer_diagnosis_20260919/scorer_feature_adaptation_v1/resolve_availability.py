"""Finite source-provenance/index check; CPU only, no model or data rebuild.

Seal reviewed local source hashes first. Resolve compares that evidence with
the existing remote reference source, reads manifest metadata once, and loads
only two auxiliary groups through the ordinary materialized loader.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time

EXPECTED_MANIFEST_SHA = "79a9e959f32ef9899116e299425d6350b17f6a04bd5070b5aa9a730319447c36"
CANONICAL_ROOT = "/root/autodl-tmp/dataset_rachel_pairwise_n512_v1"
NATIVE_RECIPES = ["reference_e1", "wave", "local", "seam_gaps", "partial_curve"]
CLAIMS = {
    "staging/pairwise_v0_2/pairwise_data/rachel_preprocess.py":
        "centerpad_mask_with_transform: tight crop and integer translation/pad, no resize; contour coordinates stay in mask pixels",
    "staging/pairwise_v0_2/pairwise_data/rachel_training_dataset.py":
        "_fragment_input/_load_mask read per-token model_mask_path at800; only diagnostic coarse128 is resized; collate_rachel_pairs stacks masks and zero-pads points",
    "staging/pairwise_v0_2/pairwise_data/rachel_composite_training.py":
        "CompositeRachelPairDataset routes source rows unchanged to RachelPairDataset; no pair-dependent normalization",
    "experiments/rachel_n512_formal_30k/materialize_recall_training.py":
        "Materializer uses RachelWeatheredDataset and save_sample without full-input resize; fixed source_row retained",
    "staging/pairwise_v0_2/pairwise_data/rachel_edge_weathering.py":
        "weather_fragment_edges requires800x800; deletes original raster pixels only; frame_unchanged=True",
    "staging/pairwise_v0_2/pairwise_data/rachel_weathered_dataset.py":
        "_fragment preserves800 frame and re-extracts contour there; __getitem__ replaces only same-frame mask/points; only coarse128 resize",
    "experiments/rachel_n512_formal_30k/materialize_s7_training.py":
        "materialize_group selects referenceE1/strong/guided/Gen5 branches then save_sample, preserving source identities",
    "staging/pairwise_v0_2/pairwise_data/rachel_s7_dataset.py":
        "strong_pair replaces mask and contour in unchanged canvas, only coarse128 resize; strong_group couples acceptance, not fragment identity",
    "staging/pairwise_v0_2/pairwise_data/rachel_strong_weathering.py":
        "strong_weather_fragment retains H/W and original pixel lattice; shape deletion only, not geometric scaling",
    "staging/pairwise_v0_2/pairwise_data/rachel_guided_partial_dataset.py":
        "GuidedPartialDataset passes retained same-shape masks into crop_training_pair; curve angle modifies cut, not fragment rotation",
    "staging/pairwise_v0_2/pairwise_data/rachel_partial_seam_dataset.py":
        "crop_training_pair rejects shape changes/new material, zeroes excluded original pixels and extracts points in same800 frame",
    "experiments/rachel_n512_formal_30k/build_s7_gen5_pool.py":
        "_process_group builds unions from canonical TRAIN parent masks; candidate_sample copies centerpadded model masks/contours without resize",
    "staging/pairwise_v0_2/pairwise_data/rachel_gen5_partition.py":
        "build_gen5_partition rejects scale!=1; logical unions in parent raster; metadata resized=False/rotation_degrees=0",
    "staging/pairwise_v0_2/pairwise_data/rachel_union_augmentation.py":
        "normalize_boundary_ownership changes pixel ownership not coordinates; _fragment uses translation-only centerpad",
    "staging/pairwise_v0_2/pairwise_data/rachel_materialized_dataset.py":
        "save/load_sample pack/unpack masks and copy stored points; no input normalization/resize in original512 loading",
    "experiments/rachel_n512_formal_30k/train_score_decoupled.py":
        "make_populations(original512) chooses MaterializedWeatheredDataset cap512 directly, not dense/rescaled input wrappers",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def seal(source_root, output):
    root = Path(source_root).resolve(strict=True)
    record = dict(schema="scorer-triplet-scale-source-review/1", source_root=str(root),
        evidence=[dict(path=path, sha256=sha(root/path), claim=claim) for path, claim in CLAIMS.items()])
    write_new(output, record)
    print(json.dumps(dict(status="source_review_sealed", file_count=len(record["evidence"]))))


def resolve(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("explicit CUDA_VISIBLE_DEVICES='' required; this is CPU-only")
    started = time.monotonic()
    source = Path(args.source_root).resolve(strict=True)
    evidence = json.loads(Path(args.evidence).read_text())
    if set(x["path"] for x in evidence["evidence"]) != set(CLAIMS):
        raise ValueError("source review is incomplete")
    mismatches = [x["path"] for x in evidence["evidence"] if sha(source/x["path"]) != x["sha256"]]
    if mismatches:
        raise ValueError("source hashes differ from reviewed implementation: " + repr(mismatches))
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("isolated_triplet_data", Path(__file__).with_name("data.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    import torch
    torch.set_num_threads(1)
    blob = Path(args.manifest).read_bytes()
    manifest_sha = hashlib.sha256(blob).hexdigest()
    if manifest_sha != EXPECTED_MANIFEST_SHA:
        raise ValueError("actual S7 TRAIN manifest SHA differs from frozen source")
    manifest = json.loads(blob)
    if (len(manifest["entries"]) != 24000 or manifest["protocol"]["dimensions_px"] != [800,800]
            or manifest["protocol"]["contour_cap"] != 512):
        raise ValueError("S7 population/input contract differs")
    # Check original-token->source-mask identity using metadata only, no mask scan.
    tokens, native_entries, gen5_entries = {}, 0, 0
    for entry in manifest["entries"]:
        if entry["source_root"] != CANONICAL_ROOT:
            continue  # No cohort proof for unrelated derivatives; excluded by data.py.
        row = entry["source_row"]
        if entry["s7_recipe"] == "gen5_partition":
            if any(not row["fragment_"+s]["fragment_token"].startswith("gen5-group-") for s in "ab"):
                raise ValueError("Gen5 source identity differs")
            gen5_entries += 1
            continue
        if entry["s7_recipe"] not in NATIVE_RECIPES:
            raise ValueError("unreviewed native recipe")
        native_entries += 1
        paths = []
        for side in "ab":
            fragment = row["fragment_"+side]
            signature = (entry["source_root"], fragment["split_unit_id"],
                         fragment["model_mask_path"], fragment["contour_path"])
            key = fragment["fragment_token"]
            if key in tokens and tokens[key] != signature:
                raise ValueError("same original token resolves to different source mask/contour")
            tokens[key] = signature
            paths.append(PurePosixPath(fragment["model_mask_path"]))
        if (entry["label"] or row.get("label_origin") == "rachel_csv_same_folder_nonneighbor_no_seam"):
            if paths[0].parent != paths[1].parent:
                raise ValueError("native within-source pair is not in one original fragment group")
    proof = dict(verified=True, frame_namespace="canonical_rachel_parent_raster_pixels",
        pixel_scale_rc=[1.,1.], canvas_hw=[800,800], mask_points_same_transform=True,
        both_endpoints_same_scale=True, pair_dependent_rescaling=False, rotation_degrees=0,
        evidence=deepcopy(evidence["evidence"]),
        scope="same-source existing labelled within-group pairs only; unit is canonical source raster pixel, not physical millimetres",
        reasoning="per-token source-mask identity; translation-only centerpad; all S7 damage deletes pixels in unchanged frame; contour coordinates use that same frame; no pair-dependent full-input resize")
    ledger = dict(schema=module.SCALE_SCHEMA, manifest_sha256=manifest_sha,
        review_source_root=str(source), source_evidence_sha256=sha(args.evidence),
        cohorts=[dict(source_roots=[CANONICAL_ROOT], recipes=NATIVE_RECIPES, proof=proof),
                 dict(source_roots=[CANONICAL_ROOT], recipes=["gen5_partition"], proof=deepcopy(proof))])
    index = module.build_triplet_index(blob, scale_ledger=ledger)
    if index["candidate_positive_count"] != 1470:
        raise ValueError("known-token candidate count differs from existing availability audit")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/"scale_ledger.json", ledger)
    write_new(out/"triplet_index.json", index)
    dataset = module.SameAnchorTripletDataset(args.manifest, scale_ledger=ledger)
    if dataset.index != index:
        raise ValueError("independent dataset index construction differs")
    selected = []
    for wanted in ("partial_curve", "gen5_partition"):
        hit = next((i for i,g in enumerate(index["groups"])
                    if g["positive"]["entry"]["s7_recipe"] == wanted), None)
        if hit is not None:
            selected.append(hit)
    if len(selected) != 2:
        raise ValueError("expected two distinct eligible native/Gen5 smoke examples")
    checks = []
    for i in selected:
        group = index["groups"][i]
        swaps = [False] + [j % 2 == 0 for j in range(len(group["negatives"]))]
        batch = dataset.get(i, swap=swaps)
        anchor_hashes = []
        for kind in range(3):
            hashes = [hashlib.sha256(batch.inputs[2*kind+int(s)][j].numpy().tobytes()).hexdigest()
                      for j,s in enumerate(swaps)]
            if len(set(hashes)) != 1:
                raise ValueError("AB/AC anchor bytes differ")
            anchor_hashes.append(hashes[0])
        checks.append(dict(group_id=group["group_id"], positive_pair_id=group["positive"]["entry"]["pair_id"],
            recipe=group["positive"]["entry"]["s7_recipe"], pair_count=len(swaps),
            anchor_mask_points_valid_sha256=anchor_hashes, swaps=swaps,
            anchor_bytes_identical=True, provenance=batch.provenance))
    write_new(out/"two_group_assembly.json", checks)
    result = dict(status="complete", manifest_sha256=manifest_sha,
        ordinary_pair_count=index["ordinary_pair_count"], candidate_positive_count=index["candidate_positive_count"],
        usable_group_count=index["usable_group_count"], usable_negative_pair_count=index["usable_negative_pair_count"],
        group_negative_count_histogram=dict(Counter(len(g["negatives"]) for g in index["groups"])),
        eligible_positive_strata=dict(Counter(g["positive"]["entry"]["source_stratum"] for g in index["groups"])),
        eligible_positive_recipes=dict(Counter(g["positive"]["entry"]["s7_recipe"] for g in index["groups"])),
        excluded_reason_counts=index["excluded_reason_counts"],
        native_entries_metadata_checked=native_entries, gen5_entries_metadata_checked=gen5_entries,
        unique_native_original_tokens_checked=len(tokens), source_files_verified=len(CLAIMS),
        assembled_group_count=len(checks), assembled_pair_count=sum(x["pair_count"] for x in checks),
        all_assembled_anchor_bytes_identical=True, gpu_used=False, inference_run=False,
        training_run=False, original_train_population_modified=False,
        elapsed_seconds=time.monotonic()-started,
        evidence_scope="source-code/cohort and manifest metadata proof, not per-file audit; only two groups loaded",
        limitations=["S7 persisted artifacts are trusted to follow their recorded source pipeline; no full24K artifact content audit.",
            "Auxiliary AC pairs inherit original-token nonjoin labels after shape deletion; this is a new auxiliary population, not unchanged original AC pixels.",
            "Canonical raster scale is equal; physical-world scale or pre-release source-image rescaling is not asserted.",
            "Availability and byte checks do not establish improved model performance."],
        artifacts={name:sha(out/name) for name in ("scale_ledger.json","triplet_index.json","two_group_assembly.json")})
    write_new(out/"summary.json", result)
    print(json.dumps(result, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("seal", "resolve"))
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--evidence")
    args = parser.parse_args()
    if args.mode == "seal":
        seal(args.source_root, args.output)
    else:
        if not args.manifest or not args.evidence:
            parser.error("resolve requires --manifest and --evidence")
        resolve(args)


if __name__ == "__main__":
    main()

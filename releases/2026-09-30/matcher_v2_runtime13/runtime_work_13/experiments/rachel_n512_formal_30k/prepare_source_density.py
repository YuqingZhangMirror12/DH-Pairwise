"""Bounded CPU prototype: two fixed E1 pairs -> true-source512/1024 targets.

Reads an explicit small local source bundle. No24K rewrite, GPU or training.
Outputs are marked prototype-only and do not use the formal TRAIN manifest
schema. Full-population preparation requires a separate authorized run.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample, save_sample
from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import centerpad_mask_with_transform
from staging.pairwise_v0_2.pairwise_data.rachel_union_augmentation import normalize_boundary_ownership
from staging.pairwise_v0_2.pairwise_data.rachel_source_density import resample_source_density


def png(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def source_geometry(record, inputs):
    group = record["group"]
    if group["status"] != "processed_group" or record["entry"]["source_row"]["split"] != "train":
        raise ValueError("only existing processed TRAIN source groups are allowed")
    parent = {k: png(inputs / name) for k, name in record["local"]["parent_masks"].items()}
    normalized, ownership = normalize_boundary_ownership(parent)
    clean = {s: png(inputs / record["local"]["clean_model_masks"][s]) for s in "ab"}
    if record["union_record"] is None:
        endpoints = {s:record["entry"]["source_row"]["fragment_"+s]["fragment_token"] for s in "ab"}
        fragments = {f["fragment_token"]:f for f in group["fragments"]}
        ids = {s:fragments[endpoints[s]]["fragment_id"] for s in "ab"}
        neighbors = [c for c in group["candidates"] if c["label"] and
                     {c["fragment_a_token"],c["fragment_b_token"]} == set(endpoints.values())]
        if len(neighbors) != 1:
            raise ValueError("missing original source CSV-positive adjacency")
        for s in "ab":
            if not np.array_equal(centerpad_mask_with_transform(parent[ids[s]]).model_mask, clean[s]):
                raise ValueError("original parent mask does not reproduce the saved clean model mask")
        source = {s:normalized[ids[s]] for s in "ab"}
        offsets = {s:fragments[endpoints[s]]["target_audit"]["parent_to_model_offset_rc"] for s in "ab"}
        allowance = 3. if ownership["applied"] else 0.
    else:
        u = record["union_record"]
        if u["split"] != "train" or u["resized"] or u["rotation_degrees"] or u["scale"] != 1:
            raise ValueError("unsupported union frame transform")
        allowed = {frozenset((c["fragment_a_token"].split('/')[-1],c["fragment_b_token"].split('/')[-1]))
                   for c in group["candidates"] if c["label"]}
        if not any(frozenset((x,u["singleton_parent_member"])) in allowed for x in u["merged_parent_members"]):
            raise ValueError("union has no surviving original adjacency")
        source = dict(a=np.logical_or.reduce([normalized[k] for k in u["merged_parent_members"]]),
                      b=normalized[u["singleton_parent_member"]])
        offsets = {s:u["model_offset_"+s+"_rc"] for s in "ab"}
        for s in "ab":
            centered = centerpad_mask_with_transform(source[s])
            if not np.array_equal(centered.model_mask, clean[s]) or tuple(offsets[s]) != centered.parent_to_model_offset_rc:
                raise ValueError("reconstructed union differs from exact saved physical clean inputs")
        allowance = 0.
    return source, offsets, clean, ownership, allowance


def run(args):
    bundle = Path(args.source_bundle).resolve(strict=True)
    records = json.loads(bundle.read_text())
    receipt_path = bundle.parent / 'origin_replay_receipt.json'
    receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    if receipts is not None and (receipts.get('schema_version') != 'rachel-density-e1-origin-receipt/1'
            or receipts.get('read_only_remote_replay') is not True):
        raise ValueError('invalid original-runtime origin receipt')
    if not 1 <= len(records) <= 8:
        raise ValueError("explicit prototype source bundle must contain1..8 pairs")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(schema_version="rachel-source-density-small-pilot/1", status="running",
        source_bundle=str(bundle), source_bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
        formal_training_eligible=False, full24k_materialized=False, gpu_used=False,
        origin_receipt=str(receipt_path) if receipts else None,
        physical_masks_modified=False, gt_translation_used_for_matches=False, pairs=[])
    for record in records:
        name = "native" if record["union_record"] is None else "union"
        sample, report = load_sample(bundle.parent / record["local"]["materialized"])
        origin = receipts['pairs'].get(sample.pair_id) if receipts else None
        source, offsets, clean, ownership, allowance = source_geometry(record,bundle.parent)
        entry = dict(pair_id=sample.pair_id, kind=record["kind"], ownership_normalization=ownership,
            ownership_changes_reference_only=record["union_record"] is None,
            parent_to_model_offsets=offsets, caps={})
        for cap in (512,1024):
            changed, derived_report, views = resample_source_density(sample,report,source,offsets,clean,
                cap=cap,ownership_allowance_px=allowance,origin_receipts=origin)
            # Canary: arbitrary translation perturbation cannot change labels.
            altered = replace(sample, translation_a_to_b_rc=sample.translation_a_to_b_rc + 1000.)
            canary, _, _ = resample_source_density(altered,report,source,offsets,clean,
                cap=cap,ownership_allowance_px=allowance,origin_receipts=origin)
            if not np.array_equal(changed.target_a,canary.target_a) or not np.array_equal(changed.target_b,canary.target_b):
                raise AssertionError("GT translation leaked into target reconstruction")
            for side in "ab":
                if not np.array_equal(getattr(changed,'mask_'+side),getattr(sample,'mask_'+side)):
                    raise AssertionError("physical mask changed")
            good = np.flatnonzero(changed.target_a>=0)
            if not np.array_equal(changed.target_b[changed.target_a[good]],good):
                raise AssertionError("targets are not one-to-one reciprocal")
            dest = output / (name+'_'+str(cap));dest.mkdir()
            save_sample(dest/'sample.npz',changed,derived_report)
            ancestry = {s+'_'+k:v for s in "ab" for k,v in views[s].items() if isinstance(v,np.ndarray)}
            np.savez_compressed(dest/'source_ancestry.npz',**ancestry)
            save(dest/'report.json',derived_report)
            d = derived_report['density']
            entry['caps'][str(cap)] = dict(points_a=len(changed.points_rc_a),points_b=len(changed.points_rc_b),
                old_correspondences=d['old_match_count'],new_correspondences=d['new_match_count'],
                ignored_a=d['sides']['a']['ignored'],ignored_b=d['sides']['b']['ignored'],
                shared_source_edges=d['topology']['shared_cell_edges'],
                accepted_source_edges=d['topology']['accepted_edges'],components=d['components'],
                exact_weathering_replay=d['replay'],source_GT_translation_canary_passed=True,
                sample_path=str(dest/'sample.npz'))
        summary['pairs'].append(entry)
    summary['status']='complete'
    save(output/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-bundle',required=True)
    p.add_argument('--output',required=True)
    run(p.parse_args())

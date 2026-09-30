"""Join existing numerical diagnoses only; no model, images or raw inference."""
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
BINS = ((1,16), (17,32), (33,64), (65,128), (129,512))


def read(path):
    return json.loads(path.read_text())


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == len({(r.get("split"), r["pair_id"]) for r in rows})
    return rows


def main():
    paths = dict(recount=HERE/"independent_recount.json",
        support=HERE.parent/"s7_matched_support_strata_v2"/"cases.jsonl",
        support_protocol=HERE.parent/"s7_matched_support_strata_v2"/"results.json",
        geometry=HERE.parent/"s7_seed_geometry_diagnostic_v1"/"cases.jsonl",
        geometry_protocol=HERE.parent/"s7_seed_geometry_diagnostic_v1"/"results.json")
    recount = read(paths["recount"])
    assert recount["status"] == read(paths["support_protocol"])["status"] == read(paths["geometry_protocol"])["status"] == "complete"
    # The producer for this support file explicitly rejected any invalid decision
    # or fallback. Thus its stored scores reconstruct the same frozen operating
    # points without silently assuming invalid decisions were accepted.
    support = {r["pair_id"]:r for r in read_rows(paths["support"])
               if r["split"] == "real" and r["label"] and r["review_status"] == "keep" and r["layout20"] is True}
    geometry = {r["pair_id"]:r for r in read_rows(paths["geometry"])}
    assert support.keys() == geometry.keys() and len(support) == 216
    for pair_id, s in support.items():
        g = geometry[pair_id]
        assert s["min_endpoints"] == g["final_unique_endpoints_min"]
        assert s["inlier_edges"] == g["final_edge_count"]
        assert s["score"] == g["matched_tokens_score"]
        assert g["final_error_px"] <= 20
    def describe(ids):
        ids = sorted(ids)
        rows = [support[i] for i in ids]
        bins = {f"{lo}-{hi}":sum(lo <= r["min_endpoints"] <= hi for r in rows) for lo,hi in BINS}
        assert sum(bins.values()) == len(rows)
        def med(values):
            return statistics.median(values) if values else None
        le32 = sum(r["min_endpoints"] <= 32 for r in rows)
        return dict(n=len(ids), pair_ids=ids, bins=bins, endpoints_min_le32=le32,
            endpoints_min_le32_fraction=le32/len(ids) if ids else None,
            medians=dict(final_min_unique_endpoints=med([r["min_endpoints"] for r in rows]),
                final_inlier_edges=med([r["inlier_edges"] for r in rows]),
                final_residual_px=med([r["residual_px"] for r in rows]),
                area_ratio=med([r["area_ratio"] for r in rows]),
                overlap_smaller_fragment_fraction=med([r["overlap_small_fraction"] for r in rows]),
                final_translation_error_px=med([geometry[i]["final_error_px"] for i in ids]),
                raw_seed_to_final_shift_px=med([geometry[i]["seed_to_final_shift_px"] for i in ids])))
    result = dict(schema="final-edge-c16-existing-support-join/1",status="complete",
        sources={k:dict(path=str(v),sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for k,v in paths.items()},
        all_correct_layout=describe(support), operating_points={})
    for op, expected_loss in (("max_f1",14),("recall_99",32)):
        prior = recount["results"]["matched_tokens"]["real"]["operating_points"][op]
        after = recount["results"]["matched_edges"]["real"]["operating_points"][op]
        paired = recount["paired_changes"]["real"]["matched_tokens"][op]
        lost = set(paired["correct_layout_accept_lost"]["pair_ids"])
        gained = set(paired["correct_layout_accept_gained"]["pair_ids"])
        old_accept = {i for i,r in support.items() if r["score"] >= prior["threshold"]}
        retained = old_accept-lost
        current_accept = retained|gained
        assert len(lost) == expected_loss and lost <= old_accept and gained.isdisjoint(old_accept)
        assert len(old_accept) == prior["accepted_layout_le20"]
        assert len(current_accept) == after["accepted_layout_le20"]
        by_bin = {}
        for lo,hi in BINS:
            ids = {i for i in old_accept if lo <= support[i]["min_endpoints"] <= hi}
            n_lost = len(ids&lost)
            by_bin[f"{lo}-{hi}"] = dict(previously_accepted=len(ids),newly_lost=n_lost,
                loss_fraction=n_lost/len(ids) if ids else None)
        result["operating_points"][op] = dict(previous_threshold=prior["threshold"],new_threshold=after["threshold"],
            newly_lost=describe(lost),retained_acceptance=describe(retained),previously_accepted=describe(old_accept),
            newly_gained=describe(gained),current_accepted=describe(current_accept),
            loss_rate_within_previously_accepted_by_bin=by_bin)
    result["limits"] = [
        "Selected contour endpoint counts are not physical seam length, independent observations, or a density measurement.",
        "Groups condition on already-correct model Layout and classification changes: descriptive failure diagnosis, not causal identification.",
        "Both heads and their separately SIMVAL-calibrated thresholds differ; this join cannot separate feature binding, head learning and calibration effects.",
        "No new inference, threshold fitting, visual judgment, training or source-data changes."]
    (HERE/"support_followup.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    lines=["# 最终点对 Scorer 新增漏判：已有支持证据复核","",
        "全部216个已保留、最终位移误差≤20px的正例；只连接已有诊断记录。点数指最终位移内点对应两侧的较少唯一端点数，不是接缝像素长度。","",
        "| 分组 | n | ≤32点 | 比例 | 点数中位数 | 1–16 / 17–32 / 33–64 / 65–128 / 129+ |",
        "|---|---:|---:|---:|---:|---|"]
    groups=[("全部正确布局",result["all_correct_layout"])]
    for op,entry in result["operating_points"].items():
        groups.extend([(op+" 新丢",entry["newly_lost"]),(op+" 两者均通过",entry["retained_acceptance"]),
                       (op+" 原先通过",entry["previously_accepted"])])
    for name,item in groups:
        lines.append(f"| {name} | {item['n']} | {item['endpoints_min_le32']} | {item['endpoints_min_le32_fraction']:.1%} | {item['medians']['final_min_unique_endpoints']} | {' / '.join(str(v) for v in item['bins'].values())} |")
    lines.extend(["", "新增漏判在较少端点支持中富集，尤其R99；不是只有≤32点受影响，两个阈值下都另有8例来自33–64点组。原先已通过的≥65点组没有新增漏判。",
        "", "在原先通过且正确布局的样本内，maxF1：≤32点丢6/11（54.5%），33–64点丢8/90（8.9%）；R99：≤32点丢24/41（58.5%），33–64点丢8/103（7.8%）。这是条件分组的关联，不能归因为head直接依赖点数。",
        "","## 解释边界","",*['- '+v for v in result["limits"]],"",
        "可复算来源及逐例ID、组内损失比例、最终残差/位移误差等中位数见同目录 support_followup.json；生成脚本 support_followup.py 未读取图片或模型。"])
    (HERE/"support_followup.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({op:{name:{k:v for k,v in entry[name].items() if k != "pair_ids"}
        for name in ("newly_lost", "retained_acceptance")} for op,entry in result["operating_points"].items()},
        ensure_ascii=False))


if __name__ == "__main__":
    main()

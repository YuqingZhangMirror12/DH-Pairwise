"""Create compact, paired selected40 evidence from completed S4/S7 SVD outputs."""
import json
from pathlib import Path

from spectral_diagnostics import NOTES, distribution, sha256

ROOT = Path(__file__).resolve().parent / "spectral"


def main():
    arms = {arm: json.loads((ROOT / arm / "metrics.json").read_text()) for arm in ("s4", "s7")}
    assert all(d["status"] == "complete" and d["case_count"] == 40 for d in arms.values())
    cases = []
    for arm, data in arms.items():
        for row in data["cases"]:
            m = row["metrics"]
            cases.append({
                "arm": arm, "dataset": row["dataset"], "pair_id": row["pair_id"],
                "label": row["label"], "stratum": row["stratum"], "score": row["score"],
                "arrays_sha256": row["arrays_sha256"], "shape": m["shape"],
                "transport_mass": m["raw_transport_total_mass"],
                "raw_sigma1": m["raw_sigma1"], "raw_fro": m["raw_frobenius_norm"],
                "mass_normalized": m["mass_normalized"], "spectral_shape": m["spectral_shape"],
                "row_maximum": m["rows_select_columns"]["raw_maximum"],
                "column_maximum": m["columns_select_rows"]["raw_maximum"],
                "row_argmax_column_coverage": m["rows_select_columns"]["argmax_opposite_coverage_fraction"],
                "column_argmax_row_coverage": m["columns_select_rows"]["argmax_opposite_coverage_fraction"],
                "mutual_argmax": m["mutual_argmax"],
            })
    by_arm = {arm: {(r["dataset"], r["pair_id"]): r for r in cases if r["arm"] == arm} for arm in arms}
    assert set(by_arm["s4"]) == set(by_arm["s7"])
    deltas = []
    for key, a in by_arm["s4"].items():
        b = by_arm["s7"][key]
        assert (a["label"], a["stratum"], a["shape"]) == (b["label"], b["stratum"], b["shape"])
        deltas.append({"dataset": key[0], "pair_id": key[1], "stratum": a["stratum"],
                       "transport_mass": b["transport_mass"] - a["transport_mass"],
                       "sigma1_over_frobenius": b["spectral_shape"]["sigma1_over_frobenius"] - a["spectral_shape"]["sigma1_over_frobenius"],
                       "effective_rank": b["spectral_shape"]["effective_rank_entropy_singular"] - a["spectral_shape"]["effective_rank_entropy_singular"]})
    paired_summary = {field: distribution([r[field] for r in deltas]) for field in
                      ("transport_mass", "sigma1_over_frobenius", "effective_rank")}
    paired_summary["mass_increase_count"] = sum(r["transport_mass"] > 0 for r in deltas)
    result = {"schema": "selected40-compact-spectral/1", "status": "complete", "case_count_per_arm": 40,
              "independent_pair_count": 40, "arm_case_rows": 80, "notes": NOTES,
              "source_files": {arm: {"path": str(ROOT / arm / "metrics.json"),
                                     "sha256": sha256(ROOT / arm / "metrics.json"),
                                     "original_sources": arms[arm]["sources"],
                                     "runtime": arms[arm]["runtime"]} for arm in arms},
              "summaries": {arm: d["summary"] for arm, d in arms.items()},
              "paired_s7_minus_s4_summary": paired_summary, "paired_deltas": deltas, "cases": cases}
    destination = ROOT / "compact.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    lines = ["# S4 / S7 — selected40 transport 谱诊断", "",
             "完成80个 arm×case 统计，只有40个不同案例；S4、S7各40例，不把重复Matcher的D2/D4算作额外样本。", "",
             "仅分析complete的冻结NPZ，无模型推理/训练；CPU affinity=[0]，BLAS/OMP等线程均1。输入协议、cases及80个NPZ的SHA256前后不变。本地合成数值、空间排列反例和完成门控测试通过。", "",
             "每个arm包含38个512×512、1个512×464、1个462×512有效矩阵；padding排除质量均0。使用float64矩形SVD，不做非方矩阵eig。", "",
             "下表为每组中位数。E4=前4个奇异值平方/全部奇异值平方；rank=exp(H(sigma/sum sigma))；PR=能量参与率。", "",
             "| arm / 选择组 | n | raw transport mass | sigma1/Fro | E4 | effective rank | energy PR |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    groups = ("dataset_label/real/0", "dataset_label/real/1", "dataset_label/ood/1", "dataset_label/test/0", "dataset_label/test/1")
    for arm, data in arms.items():
        for group in groups:
            s = data["summary"][group]
            values = [arm + " / " + group, str(s["case_count"])] + [f"{s[k]['median']:.5f}" for k in
                      ("transport_mass", "sigma1_over_fro", "energy_top4", "effective_rank", "energy_pr")]
            lines.append("| " + " | ".join(values) + " |")
    lines += ["", "## 质量与形状必须分开", "",
              "- P→P/sum(P)只去全局质量尺度，不改变sigma1/Fro、top-k能量、有效秩或参与率。raw sigma1和Fro本身随尺度变。", "- 本选例中，两模型REAL负例的谱较集中（E4中位数约0.94/0.96），但正例也可集中；负例并非低秩、正例并非高秩的定律。正负组的指标范围明显重叠。", "- 只报告分布，不拟合分类阈值、不计算显著性/泛化准确率或因果结论。REAL分层由S6的分数/布局定义；OOD包括跨模型分歧和S6分数四分位，存在主动选择偏差。", "- 行列最大值、argmax覆盖与mutual argmax是矩阵索引统计，不是空间连续性、GT拼缝长度或可拼性。OOD只有正例，且没有layout GT。", "",
              "配对S7−S4：transport mass差值中位数=" + f"{paired_summary['transport_mass']['median']:.5f}" +
              "，sigma1/Fro差值中位数=" + f"{paired_summary['sigma1_over_frobenius']['median']:.5f}" +
              "，effective rank差值中位数=" + f"{paired_summary['effective_rank']['median']:.5f}" + "。这些只是同40选例的配对描述，不是训练改动的因果效应。", "",
              "## 文件", "", "- `compact.json`：80行紧凑逐例统计、配对差值、分层汇总及来源哈希。", "- `s4/metrics.json`、`s7/metrics.json`：完整奇异值、top-1/2/4/8/16能量、所有行列最大值分布/覆盖、运行证据。", "- `SPECTRAL_HEAD_PROPOSAL.md`：谱不保留排列信息的限制，以及未实施的CA兼容消融建议。", ""]
    (ROOT / "SUMMARY.md").write_text("\n".join(lines))
    print(json.dumps({"compact_json": str(destination), "bytes": destination.stat().st_size,
                      "arm_case_rows": len(cases), "paired_summary": paired_summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()

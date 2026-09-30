"""Export compact saved positive-pair predictions for a CPU-only case gallery.

No inference, model selection, fitting, or source-image access is performed.
"""
import argparse
import json
from pathlib import Path


def export_model(directory):
    directory = Path(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    decoder = protocol["selected_full_decoder"]
    rows = []
    with (directory / "pair_results.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if not row["label"]:
                continue
            layout = row["layouts"][decoder]
            rows.append({
                key: row[key] for key in (
                    "pair_id", "fragment_a", "fragment_b", "classification",
                    "label", "case_cluster", "strict_member",
                    "target_translation_rc", "area_ratio",
                )
            })
            rows[-1]["layout"] = {key: layout.get(key) for key in (
                "translation_rc", "valid", "translation_l2_px",
            )}
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate pair IDs")
    return {"source_predictions": str(directory / "pair_results.jsonl"),
            "protocol": protocol, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {"schema": "saved-real-layout-gallery-input/1",
              "baseline": export_model(args.baseline),
              "candidate": export_model(args.candidate)}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(destination),
                      "positive_counts": {key: len(result[key]["rows"])
                                          for key in ("baseline", "candidate")}}))


if __name__ == "__main__":
    main()

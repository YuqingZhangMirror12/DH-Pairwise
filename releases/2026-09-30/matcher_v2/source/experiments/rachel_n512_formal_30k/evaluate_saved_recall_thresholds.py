"""Re-score cached predictions with thresholds chosen only from saved VAL."""
import argparse
import json
from pathlib import Path

from .recall_operating_points import fit_operating_points
from .run_layout_decoder_experiment import classification
from .train_realism_data_ablation import save_json


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def score(row, field):
    value = row
    for part in field.split("."):
        value = value[part]
    return float(value)


def run(args):
    validation = rows(args.validation)
    if len(validation) != 3000 or sum(bool(r["label"]) for r in validation) != 1500:
        raise ValueError("expected original balancedVAL3000")
    points = fit_operating_points([r["label"] for r in validation], [score(r, args.field) for r in validation])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "validation_freeze.json", dict(**points, source=str(Path(args.validation).resolve()),
        field=args.field, threshold_fit_split="val", real_or_test_opened_for_fit=False))
    # Held-out predictions first opened only after writing the VAL threshold freeze.
    result = dict(field=args.field, calibration_source="cleanVAL3000", methods={})
    for split in ("test", "real"):
        sample = rows(getattr(args, split))
        result[split] = {name: classification([r["label"] for r in sample], [score(r, args.field) for r in sample], threshold)
                         for name, threshold in points["thresholds"].items()}
    result.update(status="complete", model_retrained=False, pose_recomputed=False,
        real_or_test_used_for_fit=False, caveat="VAL target recall is not guaranteed on REAL")
    save_json(output / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--real", required=True)
    p.add_argument("--field", required=True)
    p.add_argument("--output", required=True)
    run(p.parse_args())

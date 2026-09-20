"""Original endpoint evaluator/decoder with the isolated depth-control schema."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.continuation import evaluate_continuation as base
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.continuation_depth_controls import continue_depth_controls as train

_load_frozen_model = train.private_function(base.load_frozen_model, train=train, __file__=__file__)


def inference_runtime(device):
    """Add adapter source to the existing PredictionReuse comparison fields."""
    runtime = base.evaluator.inference_runtime(device)
    root = Path(base.evaluator.__file__).resolve().parents[2]
    files = (Path(__file__), Path(train.__file__), Path(base.__file__),
             Path(train.base.__file__), Path(train.old.__file__))
    runtime["source_sha256"] = dict(runtime["source_sha256"])
    runtime["source_sha256"].update({str(p.resolve().relative_to(root)): train.old._sha256(p)
                                     for p in files})
    runtime["depth_control_runtime_binding"] = "depth-continuation-adapter/1"
    return runtime


def load_frozen_model(root, selection):
    freeze = json.loads((Path(root) / "classifier_freezes/freeze.json").read_text())
    identity = freeze.get("continuation_identity", {})
    train.validate_identity(identity)
    bindings = {"implementation_sha256": train.__file__,
                "reused_continuation_sha256": train.base.__file__,
                "reused_trainer_sha256": train.old.__file__}
    if any(identity.get(field) != train.old._sha256(path) for field, path in bindings.items()):
        raise ValueError("depth-control implementation changed since the training freeze")
    return _load_frozen_model(root, selection)


_run = train.private_function(base.evaluator.run, load_frozen_model=load_frozen_model,
                             inference_runtime=inference_runtime)


def run(args):
    if args.device != "cuda:0":
        raise ValueError("formal endpoint CLI uses cuda:0; CPU tests call loader directly")
    with train.gpu_lock():
        return _run(args)


if __name__ == "__main__":
    run(base.evaluator.parser().parse_args())

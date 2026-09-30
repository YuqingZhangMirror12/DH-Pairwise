"""C16-only held-out evaluation; use the existing frozen evaluation protocol."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace

import torch

from experiments.rachel_n512_formal_30k import evaluate_score_decoupled as original
from ..matched_only import cache, data, inference
from .model import make
from .train import SCHEMA
from .runtime import lease


class LocalEvidenceInference(inference.FrozenMatchedInference):
    def __init__(self, base, head):
        super().__init__(base, head)
        self.score_head.register_forward_hook(self.capture_scores)

    def capture_scores(self, module, arguments, output):
        self._latest_scores = output

    @torch.inference_mode()
    def forward(self, *inputs):
        result = super().forward(*inputs)
        details = dict(result.score_details, used_edge_count=self._latest_scores.used_edge_count)
        if self._latest_scores.candidate_logit is not None:
            details.update(candidate_quality_logit=self._latest_scores.candidate_logit,
                candidate_quality_probability=self._latest_scores.candidate_logit.sigmoid())
        return replace(result, score_details=details)


def load_frozen_model(root, selection):
    if selection != "fixed_epoch":
        raise ValueError("this round predeclares fixed C16, not test-driven checkpoint selection")
    root=Path(root)
    freeze=json.loads((root/"freeze.json").read_text())
    identity=freeze["identity"]
    if freeze["schema"]!=SCHEMA or freeze["status"]!="complete" or freeze["real_ood_used_for_fit"]:
        raise ValueError("not a frozen local-evidence C16 head")
    checkpoint=root/freeze["checkpoint"]
    if data.sha(checkpoint)!=freeze["checkpoint_sha256"]:
        raise ValueError("frozen head changed")
    saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if saved["completed_segments"]!=64 or saved["identity"]!=identity or saved["matcher_updated"]:
        raise ValueError("incomplete or mismatched training checkpoint")
    head=make(identity["arm"])
    head.load_state_dict(saved["model"],strict=True)
    base=cache.old.load_decoupled_checkpoint(cache.source_checkpoint()).base_model
    model=LocalEvidenceInference(base,head)
    operating=freeze["operating_points"]
    receipt=dict(training_run=str(root),selection=selection,budget=28,head_budget=16,epoch=28,head_epoch=16,
        seed=260913,freeze_path=str(root/"freeze.json"),freeze_sha256=data.sha(root/"freeze.json"),
        checkpoint_path=str(checkpoint),checkpoint_sha256=freeze["checkpoint_sha256"],
        source_matcher_checkpoint=str(cache.SOURCE),source_matcher_sha256=cache.SOURCE_SHA,
        model_config=asdict(base.config), architecture="local_evidence_v2:"+identity["arm"],
        sampling="original512",classifier_thresholds={name:operating["thresholds"]["max_f1"] for name in original.core.BRANCHES},
        operating_points=operating,winner_record=freeze,training_identity=identity,model_design=model.metadata(),
        classifier_only_pair_bce=not head.config_v2.joint_d,coarse_is_untrained_diagnostic=True,
        local_and_fused_are_same_single_classifier=True,test_or_real_used_for_fit=False,ood_used_for_fit=False,
        decoder_unchanged=True,source_code_sha256=data.sha(__file__))
    return model,receipt


def run(args):
    globals_copy=dict(original.run.__globals__,load_frozen_model=load_frozen_model)
    frozen=json.loads((Path(args.training_run)/"freeze.json").read_text())
    if frozen["identity"]["model"]["configuration"]["graph"]=="shredding":
        def set_seed(seed):
            original.core.sealed._set_determinism(seed)
            torch.use_deterministic_algorithms(True,warn_only=True)
        sealed=SimpleNamespace(**vars(original.core.sealed))
        sealed._set_determinism=set_seed
        globals_copy["core"]=SimpleNamespace(**dict(vars(original.core),sealed=sealed))
    execute=FunctionType(original.run.__code__,globals_copy,original.run.__name__,original.run.__defaults__,original.run.__closure__)
    return execute(args)


if __name__=="__main__":
    p=original.parser()
    p.add_argument("--gpu-uuid",required=True)
    p.add_argument("--lock-root",required=True)
    args=p.parse_args()
    with lease(args.gpu_uuid,args.lock_root):
        run(args)

from dataclasses import asdict, dataclass


@dataclass
class Config:
    experiment: str = 'seam_context_v3_scratch_pd'
    seed: int = 260921
    canvas_size: int = 800
    contour_cap: int = 512
    windows: tuple = (7., 16., 32., 64.)
    patch_size: int = 16
    dim: int = 96
    heads: int = 4
    arc_layers: int = 4
    correspondence_layers: int = 2
    verifier_layers: int = 2
    topk: int = 4
    arc_bins: int = 32
    edge_cap: int = 5120
    neighbors: int = 64
    sinkhorn_iterations: int = 100
    temperature: float = .25
    tolerance: float = .001
    max_candidates: int = 8
    max_gap_px: float = 128.
    context_extension_px: float = 64.
    mode_radius_px: float = 40.
    beam_width: int = 4
    activation_checkpointing: bool = True
    effective_batch: int = 32
    train_mirror_probability: float = .10
    stage_a_min: int = 12
    stage_a_max: int = 60
    stage_b_min: int = 20
    stage_b_max: int = 100
    validate_every: int = 2
    teacher_zero_epoch: int = 10
    native_only_min: int = 10
    lr_a: float = 1e-4
    lr_b: float = 5e-5
    lr_new: float = 1e-4
    weight_decay: float = 1e-4
    min_lr: float = 1e-6
    min_delta: float = .002
    threshold_min: float = .2
    threshold_max: float = .8
    threshold_step: float = .01
    layout_tolerance_px: float = 20.

    def record(self):
        return dict(asdict(self), initialization='scratch', pretrained_source=None,
                    candidate_score='sigmoid(quality_logit-null_logit)',
                    hard_dustbin_rejection=False, estimate_rotation=False,
                    same_candidate_score_and_pose=True,
                    threshold_objective='CAL mean(clean,hard) joint F1',
                    source_split_unit='connected manuscript source groups',
                    loss_weights=dict(ot=.5, support=.5, link=.25, candidate=1.,
                                      rank=.5, pose=.25, balance=.05))

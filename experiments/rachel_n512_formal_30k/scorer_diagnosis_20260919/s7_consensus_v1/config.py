"""Predeclared optimizer, pair denominators and simulation-only stop rules."""
from dataclasses import asdict,dataclass

from .losses import LossConfig
from .pose_consensus import ProposalConfig, MergeRepairPolicy, REVISION
from .scratch_matcher import MatcherLossConfig


@dataclass(frozen=True)
class TrainingConfig:
    schema: str = 's7-consensus-training/1'
    head_seed: int = 26092406
    matcher_seed: int = 26092407
    data_seed: int = 26092408
    microbatch: int = 8
    world_size: int = 2
    accumulate: int = 2
    workers_per_rank: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.
    validation_every_epochs: int = 2
    minimum_epochs: int = 16
    maximum_epochs: int = 48
    significant_improvement: float = .001
    plateau_validations: int = 3
    learning_rate_factor: float = .5
    maximum_lr_reductions: int = 2
    validations_after_reduction: int = 3
    extension_modulus: int = 4
    threshold_minimum: float = .20
    threshold_maximum: float = .80
    threshold_step: float = .01
    threshold_tie_preference: float = .30
    checkpoint_every_updates: int = 100

    @property
    def effective_batch(self):
        return self.microbatch*self.world_size*self.accumulate

    def record(self):
        return dict(**asdict(self),effective_batch=self.effective_batch,
            optimizer='AdamW',precision='fp32, no AMP/TF32',
            candidate_loss=asdict(LossConfig()),matcher_loss=asdict(MatcherLossConfig()),
            proposals=asdict(ProposalConfig()),head=dict(layers=2,dim=96,heads=4,length_scale_px=32.),
            proposal_revision=REVISION,merge_repair_policy=asdict(MergeRepairPolicy()),
            validation_checkpoint_archive='all CAL/SELECT validation epochs; immutable model weights, not TEST selection',
            train_count=24000,online_augmentation=False,
            candidate_class_reweighting='none; mean over known candidates then ALL pairs',
            extension_schedule='stable hash(pair_id, epoch) modulo4; only native correct candidates/known edges',
            primary='SELECT joint F1 under bound validation design; paired clean/hard mean or single mixed population; threshold from CAL only',
            scratch_matcher_selection='native candidate coverage under bound validation design, then top-proposal Layout20, then negative matching loss',
            real_epoch_or_geometry_selection=False,
            cached_proposals='optional bound full-Matcher-output geometric proposals; full FP32Q/F/H/unmatched recomputed online',
            saturation_note='budget limit while improving is NOT convergence')


@dataclass
class Plateau:
    best: float = float('-inf')
    bad: int = 0
    reductions: int = 0
    since_reduction: int = 0

    def observe(self,value,epoch,config):
        self.since_reduction+=1
        if value>self.best+config.significant_improvement:
            self.best=float(value);self.bad=0
        else:
            self.bad+=1
        if (epoch>=config.minimum_epochs and self.bad>=config.plateau_validations
                and self.since_reduction>=config.validations_after_reduction):
            if self.reductions>=config.maximum_lr_reductions:
                return 'simulation_plateau_after_lr_reductions'
            if epoch>=config.maximum_epochs:
                return 'budget_limit_not_claimed_converged'
            if self.reductions<config.maximum_lr_reductions:
                self.reductions+=1;self.bad=0;self.since_reduction=0
                return 'reduce_lr'
        if epoch>=config.maximum_epochs:
            return 'budget_limit_not_claimed_converged'
        return 'continue'

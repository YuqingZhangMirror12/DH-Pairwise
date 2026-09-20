"""E4/E5 equal-budget partial-seam data ablations; no architecture changes.

E4: partial source-seam truncation, original loss.
E5: exactly the same partial truncation plus the E1 weathering and target policy.
Both start independently from the same 350bf9 warm start used by E0/E1/E2.
"""
from __future__ import annotations

from dataclasses import asdict
from functools import partial
import json
from pathlib import Path

from experiments.rachel_n512_formal_30k import train_edge_weathering as trainer


PARTIAL_SEED = 260910
VARIANTS = {"e4": "partial_seam_e4", "e5": "partial_seam_weathering_e5"}


def read_pair_metadata(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "rachel-composite-training/1" or manifest.get("split") != "train":
        raise ValueError("partial ablation requires the existing composite TRAIN manifest")
    entries = manifest["entries"]
    if any(entry["row"].get("split") != "train" for entry in entries):
        raise ValueError("partial source contains a non-TRAIN row")
    return tuple((entry["row"]["pair_id"], int(entry["row"]["label"])) for entry in entries)


class PartialTrainingDataset:
    """Expose only ordinary student inputs plus loss/report sidecars.

    report.changed_pair continues to mean weathering, not truncation: surviving
    intact source-seam correspondences still supervise the original xy loss in
    E4, and in E5 whenever the subsequent weathering did not change the pair.
    """
    def __init__(self, base, *, seed, cache_dir, pair_metadata, arm):
        from staging.pairwise_v0_2.pairwise_data.rachel_partial_seam_dataset import PartialSeamDataset
        from staging.pairwise_v0_2.pairwise_data.rachel_weathered_dataset import RachelWeatheredDataset
        if arm not in VARIANTS:
            raise ValueError("unknown partial seam arm")
        self.arm = arm
        self.partial = PartialSeamDataset(base, seed=PARTIAL_SEED, epoch=0,
                                          pair_metadata=pair_metadata)
        self.weather = RachelWeatheredDataset(self.partial, seed=seed, epoch=0,
            cache_dir=cache_dir, clean_probability=1.0 if arm == "e4" else .70,
            mild_probability=0.0 if arm == "e4" else .25)
        self.split, self.root = "train", getattr(base, "root", None)

    def __len__(self):
        return len(self.partial)

    def set_epoch(self, epoch):
        self.partial.set_epoch(epoch)
        self.weather.set_epoch(epoch)

    def __getitem__(self, index):
        sample, weather_report = self.weather[index]
        report = dict(weather_report)
        report["partial_seam"] = self.partial.diagnostics(index)
        return sample, report


def build_recipe(arm, train_manifest):
    from staging.pairwise_v0_2.pairwise_data.rachel_partial_seam_dataset import PartialSeamConfig
    if arm not in VARIANTS:
        raise ValueError("unknown partial seam arm")
    metadata = read_pair_metadata(train_manifest)
    policy = dict(
        partial_seam_config=asdict(PartialSeamConfig()), partial_seed=PARTIAL_SEED,
        partial_seam_enabled=True, weathering_enabled=arm == "e5",
        requested_endpoint_tier_probabilities=(dict(clean=1., mild=0., moderate=0.)
            if arm == "e4" else dict(clean=.70, mild=.25, moderate=.05)),
        correspondence_policy="partial truncation retains only original source-supported seam targets; artificial cut-edge exclusion; E5 subsequently inherits these through E1 weathering source arcs",
        translation_loss_policy="original GT fixed; partial truncation alone retains raw xy supervision on surviving intact seam; only subsequent actual weathering disables raw xy auxiliary on positives",
        partial_pair_policy="positive and negative TRAIN groups share proposal draws and coupled acceptance; label and original GT unchanged; rejected proposals retain source sample",
        cache_policy="geometry content fingerprints distinguish partial shapes with shared fragment tokens; no token-only feature cache",
        target_rules_changed=True, original_gt_translation_preserved=True,
        inference_inputs_unchanged=True, architecture_changed=False,
        comparison="E4 vs E0 isolates partial-data package; E5 vs E1 adds partial-data package under E1 weathering; E4 vs E5 adds weathering package",
        target_distribution_used_for_design="previously inspected REAL research cohort; not a new blind holdout",
    )
    return dict(variant=VARIANTS[arm], policy=policy,
                build_dataset=partial(PartialTrainingDataset, pair_metadata=metadata, arm=arm))


def parser():
    p = trainer.parser()
    p.description = __doc__
    p.add_argument("--arm", choices=sorted(VARIANTS), required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    trainer.run(args, recipe=build_recipe(args.arm, args.train_manifest))


if __name__ == "__main__":
    main()

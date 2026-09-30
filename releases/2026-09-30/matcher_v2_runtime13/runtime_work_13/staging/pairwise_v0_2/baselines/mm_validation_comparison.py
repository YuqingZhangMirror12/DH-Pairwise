"""Post-training comparison on exactly the same frozen MM report pairs.

The formal LOCAL-Q1 runner stores aggregate validation metrics but deliberately
does not persist row-level probabilities or pair IDs.  This independent module
reloads each winning checkpoint, replays the already frozen
``validation_report`` batches, and aligns the four local arms with the
historical MM MobileNetV2 checkpoint by ``TrainingPairRecord.pair_id``.

No population is sampled here and no real-Dunhuang input is accepted.  The
primary comparison uses the intersection of valid rows across every selected
model, so every reported AUROC/AUPRC is calculated on identical pair IDs.  A
matched Route-A whole-mask Siamese checkpoint can be included as method six.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import torch
from torch import Tensor, nn

from staging.pairwise_v0_2.baselines.historical_mm_siamese import (
    HISTORICAL_MM_BASELINE_ID,
    load_historical_mm_checkpoint,
    score_historical_mm_validation,
)
from staging.pairwise_v0_2.baselines.matched_route_a_siamese import (
    MATCHED_ROUTE_A_SIAMESE_ID,
)
from staging.pairwise_v0_2.models.local_matcher import MatcherMode
from staging.pairwise_v0_2.pairwise_data.training_stream import TrainingPairRecord
from staging.pairwise_v0_2.training.checkpoint import (
    canonical_config_hash,
    load_trusted_checkpoint,
)
from staging.pairwise_v0_2.training.evaluation import evaluate_pairwise
from staging.pairwise_v0_2.training.local_q1_backend import (
    LocalQ1Backend,
    LocalQ1BackendMode,
)
from staging.pairwise_v0_2.training.local_q1_cache_builder import reopen_local_q1_cache
from staging.pairwise_v0_2.training.local_q1_provider import (
    FrozenLocalQ1BatchPlan,
    LocalQ1ReadOnlyBatchProvider,
    reopen_local_q1_batch_plan,
)
from staging.pairwise_v0_2.training.short_ablation import (
    AblationArm,
    AblationArmName,
    EvidenceMode,
    PredictionBatch,
    PreparedAblationBatch,
    record_sequence_fingerprint,
)


MM_VALIDATION_COMPARISON_VERSION = "dunhuang-mm-validation-comparison/0.1"
MM_DATASET_ID = "mm_augmented"
REPORT_PHASE = "validation_report"
_FOUR_ARMS = (
    AblationArmName.LOCAL_DUAL_SOFTMAX,
    AblationArmName.LOCAL_DUSTBIN_SINKHORN,
    AblationArmName.KEYPOINT_DUAL_SOFTMAX,
    AblationArmName.KEYPOINT_DUSTBIN_SINKHORN,
)


class MMValidationComparisonError(ValueError):
    """Raised when the post-training comparison ceases to be pair-identical."""


class _Plan(Protocol):
    def phase_batches(self, phase: str) -> Tuple[Mapping[str, Any], ...]: ...


class _Provider(Protocol):
    def planned_records(
        self, phase: str, batch_ordinal: int
    ) -> Tuple[TrainingPairRecord, ...]: ...

    def prepare(
        self,
        records: Sequence[TrainingPairRecord],
        *,
        arm: AblationArm,
        phase: str,
    ) -> PreparedAblationBatch: ...


class _Session(Protocol):
    def predict_batch(
        self, batch: PreparedAblationBatch, *, evidence: EvidenceMode
    ) -> PredictionBatch: ...


@dataclass(frozen=True)
class LoadedArmWinner:
    arm: AblationArm
    session: _Session
    runner_mm_report: Mapping[str, Any]


ArmWinnerLoader = Callable[[AblationArmName], LoadedArmWinner]


def _matcher_for(name: AblationArmName) -> str:
    if name in {
        AblationArmName.LOCAL_DUAL_SOFTMAX,
        AblationArmName.KEYPOINT_DUAL_SOFTMAX,
    }:
        return MatcherMode.DUAL_SOFTMAX.value
    if name in {
        AblationArmName.LOCAL_DUSTBIN_SINKHORN,
        AblationArmName.KEYPOINT_DUSTBIN_SINKHORN,
    }:
        return MatcherMode.DUSTBIN_SINKHORN.value
    raise MMValidationComparisonError("unsupported LOCAL-Q1 comparison arm")


def _arm_from_backend(backend: LocalQ1Backend, name: AblationArmName) -> AblationArm:
    return AblationArm(
        name=name,
        evidence=EvidenceMode.LOCAL,
        matcher_mode=_matcher_for(name),
        model_config=backend.model_config_for(name),
        optimizer_config=backend.optimizer_config,
        aggregation_config=backend.aggregation_config,
        arc_pooling=backend.model_template.arc_pooling,
    )


def _json_object(path: Path, name: str) -> Tuple[Mapping[str, Any], bytes]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    payload = target.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MMValidationComparisonError(name + " is not JSON") from exc
    if not isinstance(value, Mapping):
        raise MMValidationComparisonError(name + " root must be an object")
    return value, payload


class FourArmRunWinnerLoader:
    """Load one winner at a time from a completed four-arm run directory."""

    def __init__(
        self,
        run_directory: Path,
        plan: FrozenLocalQ1BatchPlan,
        *,
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        self.run_directory = Path(run_directory)
        self.plan = plan
        self.device = torch.device(device)
        receipt, _ = _json_object(
            self.run_directory / "local_q1_run_receipt.json", "LOCAL-Q1 run receipt"
        )
        if (
            receipt.get("status")
            != "complete_four_local_arms_validation_partitioned_no_test"
        ):
            raise MMValidationComparisonError(
                "LOCAL-Q1 run is not a completed four-arm run"
            )
        plan_row = receipt.get("batch_plan")
        if not isinstance(plan_row, Mapping) or (
            plan_row.get("file_sha256") != plan.canonical_file_sha256
            or plan_row.get("content_sha256") != plan.content_sha256
        ):
            raise MMValidationComparisonError("run and supplied batch plan differ")
        contract = receipt.get("contract")
        if (
            not isinstance(contract, Mapping)
            or type(contract.get("initialization_seed")) is not int
        ):
            raise MMValidationComparisonError("run initialization seed is missing")
        self.seed = int(contract["initialization_seed"])
        rows = receipt.get("arm_results")
        if not isinstance(rows, list):
            raise MMValidationComparisonError("run receipt lacks arm results")
        self._rows = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise MMValidationComparisonError("run arm result is invalid")
            try:
                name = AblationArmName(row["arm"]["name"])
            except (KeyError, TypeError, ValueError) as exc:
                raise MMValidationComparisonError("run arm name is invalid") from exc
            if name in self._rows:
                raise MMValidationComparisonError("run contains a duplicate arm")
            self._rows[name] = row
        if set(self._rows) != set(_FOUR_ARMS):
            raise MMValidationComparisonError("run does not contain exactly four arms")

    def __call__(self, name: AblationArmName) -> LoadedArmWinner:
        name = AblationArmName(getattr(name, "value", name))
        row = self._rows[name]
        winner = row.get("winner")
        if not isinstance(winner, Mapping) or type(winner.get("epoch")) is not int:
            raise MMValidationComparisonError("winner epoch is missing")
        epoch = int(winner["epoch"])
        checkpoint_path = self.run_directory / (
            name.value + "-epoch-{:02d}.pt".format(epoch)
        )
        authority_path = self.run_directory / (name.value + ".authority.json")
        authority, authority_payload = _json_object(
            authority_path, name.value + " authority"
        )
        authority_claim = row.get("checkpoint_authority")
        if not isinstance(authority_claim, Mapping) or (
            hashlib.sha256(authority_payload).hexdigest()
            != authority_claim.get("file_sha256")
            or authority.get("content_sha256") != authority_claim.get("content_sha256")
            or authority.get("arm") != name.value
        ):
            raise MMValidationComparisonError(
                "winner authority differs from run receipt"
            )
        checkpoint_claim = authority.get("checkpoint")
        checkpoint_config = authority.get("checkpoint_config")
        if (
            not isinstance(checkpoint_claim, Mapping)
            or not isinstance(checkpoint_config, Mapping)
            or checkpoint_claim != winner.get("checkpoint")
            or checkpoint_claim.get("epoch") != epoch
        ):
            raise MMValidationComparisonError("winner checkpoint claim is inconsistent")

        backend = LocalQ1Backend(device=self.device, mode=LocalQ1BackendMode.FORMAL)
        arm = _arm_from_backend(backend, name)
        if canonical_config_hash(checkpoint_config.get("arm")) != canonical_config_hash(
            arm.to_dict()
        ):
            raise MMValidationComparisonError("winner checkpoint arm semantics changed")
        session = backend.create_session(arm, seed=self.seed)
        load_trusted_checkpoint(
            checkpoint_path,
            session.model,
            expected_config=checkpoint_config,
            expected_file_sha256=str(checkpoint_claim["file_sha256"]),
            expected_canonical_content_sha256=str(
                checkpoint_claim["canonical_content_sha256"]
            ),
            map_location="cpu",
            trusted=True,
        )
        session.model.eval()
        report = row.get("validation_report")
        by_dataset = report.get("by_dataset") if isinstance(report, Mapping) else None
        mm_report = (
            by_dataset.get(MM_DATASET_ID) if isinstance(by_dataset, Mapping) else None
        )
        if not isinstance(mm_report, Mapping):
            raise MMValidationComparisonError("runner lacks the MM validation report")
        return LoadedArmWinner(
            arm=arm,
            session=session,
            runner_mm_report=mm_report,
        )


def _metrics(
    probability: Sequence[float],
    label: Sequence[bool],
    valid: Sequence[bool],
    cluster_id: Sequence[str],
) -> Mapping[str, Any]:
    return evaluate_pairwise(
        probability,
        label,
        valid,
        cluster_id,
        threshold=0.5,
    )


def _ranking_match(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    try:
        return all(
            math.isclose(
                float(first[view][metric]),
                float(second[view][metric]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for view in ("row", "cluster_balanced")
            for metric in ("auroc", "auprc")
        )
    except (KeyError, TypeError, ValueError):
        return False


def _serial_probability(probability: Tensor, valid: Tensor) -> list:
    return [
        float(score) if bool(is_valid) else None
        for score, is_valid in zip(probability.tolist(), valid.tolist())
    ]


def compare_mm_validation_report(
    *,
    plan: _Plan,
    provider: _Provider,
    arm_winner_loader: ArmWinnerLoader,
    historical_model: nn.Module,
    historical_mask_loader: Callable[[Any], Any],
    matched_model: Optional[nn.Module] = None,
    matched_batch_size: int = 256,
) -> Mapping[str, Any]:
    """Recompute five or six methods on one frozen MM report pair order."""

    if not callable(arm_winner_loader):
        raise TypeError("arm_winner_loader must be callable")
    entries = tuple(plan.phase_batches(REPORT_PHASE))
    if not entries:
        raise MMValidationComparisonError("validation_report has no frozen batches")
    batches = []
    mm_records = []
    mm_indices = []
    for expected_ordinal, entry in enumerate(entries):
        ordinal = int(entry.get("ordinal", -1))
        if ordinal != expected_ordinal:
            raise MMValidationComparisonError(
                "report batch ordinals are not contiguous"
            )
        records = tuple(provider.planned_records(REPORT_PHASE, ordinal))
        if not records:
            raise MMValidationComparisonError("report contains an empty batch")
        indices = tuple(
            index
            for index, record in enumerate(records)
            if record.dataset_id == MM_DATASET_ID
        )
        batches.append(records)
        mm_indices.append(indices)
        mm_records.extend(records[index] for index in indices)
    if not mm_records:
        raise MMValidationComparisonError("frozen report contains no MM records")
    if any(record.split != "val" for record in mm_records):
        raise MMValidationComparisonError("MM report contains a non-validation row")
    pair_ids = tuple(record.pair_id for record in mm_records)
    if len(set(pair_ids)) != len(pair_ids):
        raise MMValidationComparisonError("MM report pair IDs are not unique")
    labels = tuple(record.label for record in mm_records)
    clusters = tuple(record.component_id for record in mm_records)
    if not any(labels) or all(labels):
        raise MMValidationComparisonError("MM report must contain both classes")

    historical_parameter = next(historical_model.parameters(), None)
    historical_device = (
        torch.device("cpu")
        if historical_parameter is None
        else historical_parameter.device
    )
    historical = score_historical_mm_validation(
        historical_model,
        mm_records,
        historical_mask_loader,
        device=historical_device,
    )
    if historical.pair_ids != pair_ids:
        raise MMValidationComparisonError("historical model changed MM pair order")
    method_probability: Dict[str, Tensor] = {
        HISTORICAL_MM_BASELINE_ID: historical.probability
    }
    method_valid: Dict[str, Tensor] = {
        HISTORICAL_MM_BASELINE_ID: torch.ones(len(pair_ids), dtype=torch.bool)
    }
    if matched_model is not None:
        matched_parameter = next(matched_model.parameters(), None)
        matched_device = (
            torch.device("cpu")
            if matched_parameter is None
            else matched_parameter.device
        )
        matched_result = score_historical_mm_validation(
            matched_model,
            mm_records,
            historical_mask_loader,
            device=matched_device,
            batch_size=matched_batch_size,
        )
        if matched_result.pair_ids != pair_ids or not torch.equal(
            matched_result.label, historical.label
        ):
            raise MMValidationComparisonError(
                "matched Siamese changed MM pair order or labels"
            )
        method_probability[MATCHED_ROUTE_A_SIAMESE_ID] = matched_result.probability
        method_valid[MATCHED_ROUTE_A_SIAMESE_ID] = torch.ones(
            len(pair_ids), dtype=torch.bool
        )
    runner_reference: Dict[str, Mapping[str, Any]] = {}
    prepared_identity: Dict[Tuple[str, int], Tuple[str, str, str]] = {}

    for name in _FOUR_ARMS:
        loaded = arm_winner_loader(name)
        if loaded.arm.name is not name:
            raise MMValidationComparisonError("winner loader returned the wrong arm")
        arm_probability = []
        arm_valid = []
        for ordinal, (records, indices) in enumerate(zip(batches, mm_indices)):
            prepared = provider.prepare(records, arm=loaded.arm, phase=REPORT_PHASE)
            if not isinstance(prepared, PreparedAblationBatch):
                raise MMValidationComparisonError("provider returned an invalid batch")
            if prepared.record_sequence_sha256 != record_sequence_fingerprint(records):
                raise MMValidationComparisonError(
                    "provider changed report record order"
                )
            identity = (
                prepared.record_sequence_sha256,
                prepared.prepared_input_sha256,
                str(prepared.local_candidate_sha256),
            )
            key = (prepared.candidate_representation, ordinal)
            previous = prepared_identity.setdefault(key, identity)
            if previous != identity:
                raise MMValidationComparisonError(
                    "dual/Sinkhorn inputs differ within one representation"
                )
            prediction = loaded.session.predict_batch(
                prepared, evidence=EvidenceMode.LOCAL
            )
            if not isinstance(prediction, PredictionBatch) or tuple(
                prediction.probability.shape
            ) != (len(records),):
                raise MMValidationComparisonError("winner prediction shape changed")
            arm_probability.extend(
                float(prediction.probability[index].detach().cpu()) for index in indices
            )
            arm_valid.extend(
                bool(prediction.valid[index].detach().cpu()) for index in indices
            )
        probability = torch.tensor(arm_probability, dtype=torch.float64)
        valid = torch.tensor(arm_valid, dtype=torch.bool)
        if tuple(probability.shape) != (len(pair_ids),):
            raise MMValidationComparisonError("winner MM prediction count changed")
        method_probability[name.value] = probability
        method_valid[name.value] = valid
        runner_reference[name.value] = loaded.runner_mm_report
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    common_valid = torch.ones(len(pair_ids), dtype=torch.bool)
    for method, valid in method_valid.items():
        common_valid &= valid & torch.isfinite(method_probability[method])
    if not common_valid.any().item():
        raise MMValidationComparisonError(
            "selected methods have no common valid MM row"
        )
    common_labels = torch.tensor(labels, dtype=torch.bool)[common_valid]
    if not common_labels.any().item() or common_labels.all().item():
        raise MMValidationComparisonError("common-valid MM rows lack both classes")

    methods = {}
    for method, probability in method_probability.items():
        valid = method_valid[method]
        native = _metrics(probability.tolist(), labels, valid.tolist(), clusters)
        common = _metrics(probability.tolist(), labels, common_valid.tolist(), clusters)
        row = {
            "probability": _serial_probability(probability, valid),
            "native_valid": valid.tolist(),
            "native_valid_count": int(valid.sum().item()),
            "native_metrics": native,
            "common_valid_metrics": common,
        }
        if method in runner_reference:
            reference = runner_reference[method]
            matched = _ranking_match(native, reference)
            if not matched:
                raise MMValidationComparisonError(
                    method + " replayed AUROC/AUPRC differs from runner report"
                )
            row["runner_ranking_metrics_reproduced"] = True
        methods[method] = row

    return {
        "schema_version": MM_VALIDATION_COMPARISON_VERSION,
        "status": (
            "complete_same_mm_pair_ids_six_method_comparison"
            if matched_model is not None
            else "complete_same_mm_pair_ids_five_method_comparison"
        ),
        "phase": REPORT_PHASE,
        "dataset_id": MM_DATASET_ID,
        "pair_count": len(pair_ids),
        "common_valid_count": int(common_valid.sum().item()),
        "same_pair_ids_all_methods": True,
        "primary_metric_population": (
            "intersection_valid_across_all_six_methods"
            if matched_model is not None
            else "intersection_valid_across_all_five_methods"
        ),
        "matched_route_a_siamese_checkpoint_evaluated": matched_model is not None,
        "pair_ids": list(pair_ids),
        "labels": list(labels),
        "common_valid": common_valid.tolist(),
        "methods": methods,
    }


def compare_mm_validation_report_from_run(
    *,
    plan: FrozenLocalQ1BatchPlan,
    provider: _Provider,
    run_directory: Path,
    historical_model: nn.Module,
    historical_mask_loader: Callable[[Any], Any],
    matched_model: Optional[nn.Module] = None,
    matched_batch_size: int = 256,
    device: Union[str, torch.device] = "cuda",
) -> Mapping[str, Any]:
    """Convenience wrapper for a completed formal run directory."""

    loader = FourArmRunWinnerLoader(run_directory, plan, device=device)
    return compare_mm_validation_report(
        plan=plan,
        provider=provider,
        arm_winner_loader=loader,
        historical_model=historical_model,
        historical_mask_loader=historical_mask_loader,
        matched_model=matched_model,
        matched_batch_size=matched_batch_size,
    )


def write_mm_validation_comparison(path: Path, result: Mapping[str, Any]) -> None:
    """Write row-level scores once so AUROC/AUPRC can be independently rerun."""

    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def compare_route_a_mm_validation_report(
    *,
    inputs: Any,
    cache_dir: Path,
    batch_plan_path: Path,
    run_directory: Path,
    historical_checkpoint_path: Path,
    matched_checkpoint_path: Optional[Path] = None,
    matched_batch_size: int = 256,
    max_local_tensor_elements: Optional[int] = None,
    output_path: Path,
    device: Union[str, torch.device] = "cuda",
) -> Mapping[str, Any]:
    """Reopen an existing Route-A cache/plan and write the comparison result.

    This convenience path uses the same context constructors as the research
    command.  Call :func:`compare_mm_validation_report_from_run` directly when
    an already-open provider is available, which avoids reopening the cache.
    """

    # Imported lazily to keep the core evaluator independent and easy to test.
    from staging.pairwise_v0_2.training import local_q1_route_a_research as research

    if not isinstance(inputs, research.RouteAResearchInputs):
        raise TypeError("inputs must be RouteAResearchInputs")
    population = research.rebuild_route_a_population(inputs)
    population = research.population_for_existing_cache(population, cache_dir)
    config = research.planning_config_with_local_tensor_bound(
        research._planning_config_from_cache(cache_dir),  # noqa: SLF001
        max_local_tensor_elements,
    )
    trust = research.automatic_cache_trust(population, cache_dir)
    loader_factory = research.research_loader_factory(inputs)
    opened = reopen_local_q1_cache(
        population=population,
        loader_factory=loader_factory,
        output_dir=cache_dir,
        trust=trust,
        inventory_config=config.inventory_config,
        cache_limits=config.cache_limits,
        replay_source_masks=False,
    )
    _plan_value, plan_file, plan_content = research._load_receipt(  # noqa: SLF001
        batch_plan_path
    )
    external_locks = research._external_locks(  # noqa: SLF001
        trust,
        config,
        plan_file_sha256=plan_file,
        plan_content_sha256=plan_content,
    )
    plan = reopen_local_q1_batch_plan(batch_plan_path, external_locks=external_locks)
    provider = LocalQ1ReadOnlyBatchProvider(
        opened=opened,
        loader_factory=loader_factory,
        plan=plan,
        geometry_config=config.geometry_batch_config,
        inventory_config=config.inventory_config,
        external_locks=external_locks,
    )
    historical_model = load_historical_mm_checkpoint(
        historical_checkpoint_path, device=device
    )
    matched_model = (
        None
        if matched_checkpoint_path is None
        else load_historical_mm_checkpoint(matched_checkpoint_path, device=device)
    )
    historical_loader = loader_factory()
    try:
        result = compare_mm_validation_report_from_run(
            plan=plan,
            provider=provider,
            run_directory=run_directory,
            historical_model=historical_model,
            historical_mask_loader=historical_loader,
            matched_model=matched_model,
            matched_batch_size=matched_batch_size,
            device=device,
        )
    finally:
        close = getattr(historical_loader, "close", None)
        if callable(close):
            close()
    write_mm_validation_comparison(output_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "route_a_freeze",
        "predecessor_freeze",
        "eligibility_index",
        "eligibility_receipt",
        "route_policy",
        "route_config",
        "mm_archive",
        "eccv_archive",
        "mm_fingerprint_cache",
        "eccv_fingerprint_cache",
        "historical_split",
        "synthetic_manifest",
        "synthetic_archive",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--mm-extracted-root", type=Path)
    parser.add_argument("--eccv-extracted-root", type=Path)
    parser.add_argument("--synthetic-extracted-root", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--batch-plan", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--historical-checkpoint", type=Path, required=True)
    parser.add_argument("--matched-checkpoint", type=Path)
    parser.add_argument("--matched-batch-size", type=int, default=256)
    parser.add_argument("--max-local-tensor-elements", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    from staging.pairwise_v0_2.training.local_q1_route_a_research import (
        RouteAResearchInputs,
    )

    args = _parser().parse_args(argv)
    inputs = RouteAResearchInputs(
        route_a_freeze=args.route_a_freeze,
        predecessor_freeze=args.predecessor_freeze,
        eligibility_index=args.eligibility_index,
        eligibility_receipt=args.eligibility_receipt,
        route_policy=args.route_policy,
        route_config=args.route_config,
        mm_archive=args.mm_archive,
        eccv_archive=args.eccv_archive,
        mm_fingerprint_cache=args.mm_fingerprint_cache,
        eccv_fingerprint_cache=args.eccv_fingerprint_cache,
        historical_split=args.historical_split,
        synthetic_manifest=args.synthetic_manifest,
        synthetic_archive=args.synthetic_archive,
        mm_extracted_root=args.mm_extracted_root,
        eccv_extracted_root=args.eccv_extracted_root,
        synthetic_extracted_root=args.synthetic_extracted_root,
    )
    result = compare_route_a_mm_validation_report(
        inputs=inputs,
        cache_dir=args.cache_dir,
        batch_plan_path=args.batch_plan,
        run_directory=args.run_directory,
        historical_checkpoint_path=args.historical_checkpoint,
        matched_checkpoint_path=args.matched_checkpoint,
        matched_batch_size=args.matched_batch_size,
        max_local_tensor_elements=args.max_local_tensor_elements,
        output_path=args.output,
        device=args.device,
    )
    compact = {
        "status": result["status"],
        "pair_count": result["pair_count"],
        "common_valid_count": result["common_valid_count"],
        "output": str(args.output),
        "common_valid_ranking": {
            method: {
                metric: row["common_valid_metrics"]["row"][metric]
                for metric in ("auroc", "auprc")
            }
            for method, row in result["methods"].items()
        },
    }
    print(json.dumps(compact, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


__all__ = [
    "FourArmRunWinnerLoader",
    "LoadedArmWinner",
    "MMValidationComparisonError",
    "MM_VALIDATION_COMPARISON_VERSION",
    "compare_mm_validation_report",
    "compare_mm_validation_report_from_run",
    "compare_route_a_mm_validation_report",
    "write_mm_validation_comparison",
]


if __name__ == "__main__":
    raise SystemExit(main())

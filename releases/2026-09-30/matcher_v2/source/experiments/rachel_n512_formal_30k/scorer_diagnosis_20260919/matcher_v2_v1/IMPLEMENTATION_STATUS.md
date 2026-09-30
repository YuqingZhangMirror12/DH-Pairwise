# Matcher v2 implementation and verified boundaries — 2026-09-30

## Current verified update (supersedes the archived checklist below)

The Matcher v2 implementation and bounded CUDA feasibility check are complete.
**Formal B1–B3 training, new-head training and accuracy claims are not complete.**

`runtime_work_11` passed **183/183 CPU tests locally and remotely**, no skips,
errors or failures. Its source binding is
`7ecbe67e262fee81043d985026f632b028ff0cb800ba2ffbc9e66c414175f71b`.
The original659 production-baseline Python files remain byte-identical.

The actual two-GPU test used800×800 masks/N512, FP32, micro8 per GPU,
accumulate2/effective32. It ran12 updates/384 exposures, then a separate update1
checkpoint resumed through update12. All three children returned0; all111
trainable tensors received finite gradients on each rank. Every enabled module
had positive aggregate gradient from update2. Idle Scorer weights stayed unchanged.
The full shared state and both rank-specific states matched exactly.

- Shared state SHA: `3c4fd48078e42961afe3d78d59e1d3bae677b9b09e209d9d4685f1a15c2d5195`.
- Verified GPU gate SHA: `882467507c110009eb4781703e321cbad404f46331f00f6100ba9781f1acf5c8`.
- Peak allocated6769.80MiB; peak reserved8638MiB; device capacity24080.875MiB.
- Gate weights are not formal weights; actual new formal updates remain0.

The user-authorized temporary pause preserved the original joint run's complete
model, AdamW and two-rank RNG checkpoint. The same joint source was resumed once
after each bounded attempt. Other running jobs were not modified.

Two preceding failed tests remain archived. The first exposed CUDA cumsum's
strict-determinism limitation, fixed with an on-device fixed-order float64 prefix
scan and one cast back. Strict deterministic algorithms stayed enabled. The
second exposed a missing gradient-receipt directory, fixed with rank-safe
creation and a real-disk-write regression test. Neither was OOM; neither failed
attempt is counted as a passed gate.

Full v4.2 TRAIN6000/SELECT900/TEST900 generation and independent pixel/GT/target
audits have completed (7800pairs; all six full-split children returned0).
Generation terminal SHA:
`ccc25e1b5c89f52cdee6202fe93d2cb9901474601c27511d63604f4ac9555d25`.
Cross-split source-family/model-input overlaps are0. **Training admission remains
false** pending the canonical-loader/cross-catalog audit, population geometry
acceptance, E32 SELECT-only calibration and remaining source-lineage qualification.
Generation completion is not training admission; TEST is not used for tuning.

The public code capsule excludes private colleague/vendor code, masks, weights,
human labels and fixed-case identities. Included production Python is byte-identical
to the tested source; one test-only fixture now uses synthetic identities instead
of the private case plan. The public inventory has a separate binding and183-test
CPU receipt, not the server's full inventory hash. Reference test arrays and the
three externally supplied v4.2 generator modules remain external dependencies.

The new-head/other-arm catalog-bound GPU gates remain required. Short-test success
establishes execution, gradients, memory and recovery only, not better accuracy.

## Archived pre-CUDA checklist

The following is the earlier preparation record. Its pending-state statements
are historical; the verified update above takes precedence.

The two user-provided `CODEX_PROMPT_*_20260930.md` specifications and later
direct clarifications are authoritative. This checklist is not a completion
or GPU-training receipt. The user's requested GitHub upload and archive are
conditional on code completion; no completed release has been published yet.

| Requirement | Implementation | Verification / remaining work |
|---|---|---|
| U1 full-contour self attention and valid-length circular RoPE | `network.py` | CPU tests pass, including cyclic rotation, integer harmonics and padding; actual CUDA gate pending |
| U3 synchronous full point-to-point cross attention | `network.py` | CPU symmetry/padding/long-range tests pass; CUDA/DDP pending |
| U4 gain initialized at 1, cap 5; optional matchability disabled | `network.py` | CPU neutral-output, cap and gradient tests pass |
| U2 concat residual and four-scale late logits | `adapter.py`, `network.py` | CPU single-Sinkhorn, zero-step equivalence and gradients pass |
| Preserve legacy convolutions/landmarks, FP32 Sinkhorn, T16 and light-head interfaces | `adapter.py` | Default-disabled compatibility and six-input evidence tests pass; full pipeline gate pending |
| Explicit architecture in exports | `spec.py`, `runtime_io.py`, `terminal.py`, `evaluate.py` | Trainer/export/evaluator integration and strict new-architecture reconstruction pass CPU tests; formal trained v2 results pending |
| Keep all original data exposures; add straight updates from start | `additive_exposure.py`, `runtime_schedule.py`, `compile_execution.py` | Exact original-clock LR/cursor and additive-ledger reconstruction implemented/tested; actual straight catalog admission pending |
| Approved v4.2 geometry with new seeds | sibling `straight_seam_v42_reuse` | 30-image review approved; 900-pair reproduction complete |
| Explicit mismatch ignore and healthy correspondence minimum | `supervision.py`, `full_generate.py`, `audit_full.py` | 15 CPU tests local/remote; 70-pair actual end-to-end gate passed; full batch pending |
| TRAIN6000 / SELECT900 / TEST900 and balanced M/J/R quotas | `full_generate.py`, `full_pipeline.py` | Full immutable CPU pipeline running; all-population acceptance/calibration pending |
| Avoid real evaluation donors | synthetic allowlist, source hashes and known-alias exclusion | Direct donor restriction enforced; complete manuscript-level alias mapping remains unproved |
| B1–B3 training and both retrained Scorers | `execution.py`, `launcher.py`, `pipeline.py`, `evaluation_controller.py` | Native loss, optimizer/RNG recovery, own-arm frozen Matcher, fresh heads, mandatory final evaluations and failure-first lifecycle tested on CPU; CUDA gate and formal results pending |
| Compare supplied in-tree implementation | `compare_intree_reference.py` | Ten full-size pairs: zero-init exactly matches legacy in both versions; mapped nonzero output maximum difference2.861e-6, synthetic-only gradient maximum difference1.043e-7; not an accuracy comparison |
| Verification without accidental formal start | `launcher.py --gate-only`, `joint_priority.py` | Gate-only lifecycle tested; checkpoint-preserving authorized two-GPU pause/continuation controller prepared |
| GitHub completed code release + archive | not published | Only after remaining code integration and verification; do not label a partial snapshot complete |

Expanded runtime preparation:177 tests passed locally and remotely for
`runtime_work_08` (`runtime_cpu_{local,remote}_06.json`). The next immutable
snapshot adds two checkpoint/identity tests for the explicit pause controller.
Original production-baseline659 Python files remain byte-identical.
Those tests do not replace CUDA, distributed-gradient, optimizer/RNG recovery,
memory or formal end-to-end training validation.

Full-data pipeline (CPU only):
`/root/autodl-tmp/matcher_v2_20260930/v42_full_dataset_01`.
Immutable code: sibling remote `v42_full_runtime_01`.
The 70-pair gate's independent audit SHA-256 is
`bed5cc1868969e51ff52826a7f9005d9e92726036a1901d43e0572bf6e6d8ea6`.
Every production split retains `training_admitted=false` until all admission
requirements are met. Existing GPU source snapshots are not modified. The user
has separately authorized a temporary joint-training pause for actual CUDA
verification; only a verified pause receipt establishes that it occurred.

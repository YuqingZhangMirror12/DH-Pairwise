# Matcher v2 and additive straight-seam training

## Scope and precedence

The two 2026-09-30 user-supplied specifications under
`deliverables/research_handoff_20260929_published/claude_analysis_20260930/`
define the network and M/J/R data. Direct user clarifications take precedence:

1. Introduce straight-seam data from the start and an early subsequent window,
   not only at the end of the curriculum.
2. **Preserve every original exposure and add straight-seam optimizer updates.**
   Do not replace original examples to keep the total at 24,000 updates.
3. The latest review supersedes the initial independent-generator suggestion:
   reuse the supplied v4.2 geometry with independent seeds and synthetic-only
   allowed donors. The user has approved the 30-image review and full generation.
   Use TRAIN M1200/J2400/R2400, each balanced, plus SELECT/TEST 900 each.
   The 8-positive/2-negative review ratio is NOT the training label ratio.

This is a NEW experiment package, not a modification of the running curriculum,
simple, joint, pooling, residual, or frozen-evaluation jobs. The user-authorized
bounded two-GPU v2 gate is complete; no new formal v2 job has started. The joint
task was temporarily paused at a full checkpoint and resumed by its dedicated
continuation supervisor. Legacy source snapshots remain immutable.

## Registered budget (compiled from the original ledger; added-data admission required)

Original ledger: 24,000 updates × effective batch 32 = 768,000 exposures.
Original stage lengths remain 15,000 / 6,000 / 3,000 updates; their individual
sample identities, repetitions, label balance and within-base ordering are kept.

| Window of original updates | Original updates | Added straight updates | Straight fraction of updates | Total |
|---|---:|---:|---:|---:|
| 0–1,499: initial window | 1,500 | 167 | 10.02% | 1,667 |
| 1,500–14,999: early follow-up | 13,500 | 4,500 | 25% | 18,000 |
| 15,000–20,999 | 6,000 | 2,000 | 25% | 8,000 |
| 21,000–23,999 | 3,000 | 1,000 | 25% | 4,000 |
| **Total** | **24,000** | **7,667** | **24.21%** | **31,667** |

The added stream contains 245,344 exposures, for 1,013,344 combined exposures.
These are training presentations, NOT independent physical manuscripts.
Each added global batch is balanced 16 positive + 16 negative, like the base.
Extra batches are deterministically interleaved starting at base update 0 of
each window. This is interleaving of optimizer batches, not a claim that every
single microbatch contains both sources. Original three-stage proportions are
unchanged after removing the inserted straight-seam batches.

The learning-rate clock is the number of **completed original updates**:
original knots 0 / 15,000 / 21,000 retain 1e-4 / 5e-5 / 2.5e-5. Added updates use
that current LR but do not skip a scheduled original update. Optimizer time
and recovery cursors count ALL actual updates. This changes the optimizer
trajectory intentionally; it is not equivalent to the old run.

| Arm | Matcher | Training presentations | Actual updates |
|---|---|---|---:|
| B0 | Original | Existing original ledger | 24,000; existing result |
| B1 | Original | Original ledger + added straight stream | 31,667 |
| B2 | v2 | Same original ledger | 24,000 |
| B3 | v2 | Identical admitted ledger to B1 | 31,667 |

B1 and B3 must bind the SAME added catalog, RNG, sample order and update count;
B0/B2 share the original exposure budget. Different architectures need not
have identical FLOPs or wall time. B1/B0 and B3/B2 change BOTH data and training
amount: those contrasts cannot establish a pure data-only effect. No additional
budget-matched replay control is silently authorized by this plan. B4–B6 remain
conditional on B3, not automatically launched. Each selected Matcher needs new
Scorer heads per the original specification. Their compiler requires the same
arm's completed SIM-selected Matcher and reconstructs the registered module
ledger; it never substitutes E32 or another arm's checkpoint.

The B2 ledger can be compiled now from the admitted original data. B1/B3 require
complete generated-data admission before an executable ledger is emitted. The total above is
documented up front; do not silently extend it when data is merged.

## Implemented network boundary

- Default `MatcherV2Config.enabled=False` delegates to the unmodified adapter,
  retaining checkpoint keys and legacy output/RNG behaviour.
- When enabled: two pre-LN full-contour self/cross blocks, integer-harmonic
  circular RoPE for self attention only, four heads × 24 dimensions, FF 192.
  Both cross directions read the same pre-cross pair, not an in-place update.
- Concat multiscale residual and per-scale primal/dual cosine late logits use
  one shared patch encoding per side, with zero initial residual contributions.
- Learned log sharpness initializes to gain 1 and is capped at 5. Optional
  matchability is OFF by default and neutral if explicitly enabled.
- One existing FP32 Sinkhorn; original dustbin/temperature, six input fields,
  `MatcherEvidence`, T16 builder, final-8 budget, and downstream Scorer interface
  are unchanged. Existing legacy contour-context padding behaviour is not
  retroactively claimed to be fixed by the new module's padding tests.
- Non-reentrant activation checkpointing covers new attention; full training
  still requires actual CUDA/DDP determinism, memory and resume gates.
- Explicit serialized architecture/config is mandatory for exports; no guessing
  the new architecture from state-dict tensor shapes.

## Data quality and source admission

The companion source audit checks all referenced v14 source-family identifiers,
then the original Rachel masks admitted to the new generator: paths, decoded
pixel hashes and translation-normalized cropped pixel hashes across splits.
Derived union masks and gen5 composite groups without original-mask paths are
reported as exclusions from this NEW base-mask pool. They are NOT removed from
the original training ledger. Unknown identifiers, missing files, ambiguous
families and cross-split collisions fail closed.

The initial audit rejected the two actual derived-token spellings; the revised
auditor recognizes them explicitly and still checks their declared families.
The failed first report is retained, never overwritten or labelled passed.

Correspondence supervision uses the original mutual-nearest-contour algorithm
but additionally requires both tokens to lie on the intended seam and outside
the UNION of either side's mismatch intervals (including shoulders and a
smoothing margin). Those damaged seam tokens are ignored (`-2`), not forced
matches or false dustbin targets; other unmatched points are `-1`. The tolerance
may add the two documented R-type uniform-wear depths, but must NOT absorb
localized gaps/overlaps. No GT, interval or provenance mask enters model inputs.
These functions do not by themselves certify a generated image: an independent
pixel/projection audit must check the eventual generator's eligibility masks.

The approved v4.2 geometry is implemented in the sibling
`straight_seam_v42_reuse` package. The first independent 900-pair reproduction
and pixel/GT audit are complete. A shadow supervision audit found four J
positives below eight healthy correspondences after explicit damage exclusion;
that reproduction batch remains non-admitted. Its original files are unchanged.

The full-generation wrapper preserves shape functions, uses exact J base quotas
(80/12/8 within each label), filters damaged targets before saving, and rejects
attempts with fewer than eight healthy correspondences without changing class.
All attempts and rejection reasons are retained. A 70-pair end-to-end preflight
has passed; TRAIN6000, SELECT900 and TEST900 generation and independent audits
are complete in their immutable CPU pipeline, using separate source pools/seeds.
TEST receives integrity checks only, not calibration or model-based selection.

Full 6K/900/900 source/seed isolation, survival-bias tables, geometry acceptance,
slide/competing-edge checks and frozen E32 SELECT-only calibration remain
required before dataset admission. The successful preflight does not certify
the full dataset or complete cross-library manuscript lineage.

## Verification and remaining execution gates

The local/remote immutable snapshot receipts are stored under
`artifacts/matcher_v2_20260930` and `/root/autodl-tmp/matcher_v2_20260930`.
CPU checks cover neutral initialization, gradients, padding, module freeze,
checkpoint parity, model-spec round trip, source collisions, mismatch targets,
and preservation/resumption of the original-plus-extra exposure sequence.

Training, new-architecture export/reload, separate final populations, fresh
Patch/Stats heads and mandatory evaluation controllers are integrated and CPU
tested. Still required before B1/B3 formal training: full generated-data admission
and their actual catalog-bound ledgers. B2's full-shape FP32 CUDA/DDP12-update
memory/gradient/resume gate passed on `runtime_work_11`; the other arm/head
catalog-bound gates remain mandatory. Discard gate weights and begin formal
training from the specified new random seed. Do not claim all-arm gates or
accuracy improvement from this one Matcher feasibility gate.

Model selection, preprocessing and parameter choices never use held-out TEST.
Preserve original real CAL/SELECT/TEST source folds and distinguish development
results from a fresh blind evaluation. Keep Turufan Layout/Joint metrics null.

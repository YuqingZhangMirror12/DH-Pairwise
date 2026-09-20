# Rachel PairingNet benchmark: adaptation contract

This benchmark is a **PairingNet-derived Rachel adaptation**, not a claim that
the unmodified official repository was reproduced.  Its reference is the
official PairingNet repository at commit
`e878b781b2b2065a4b7da09d2f639e8f0a35e97a`, together with the ECCV 2024
paper *PairingNet: A Learning-based Pair-searching and -matching Network for
Image Fragments*.

## Official method audited

The official pair-matching path uses:

- ordered contour points, padded to at most 2,900 points;
- a 7 x 7 edge-only contour patch and a 7 x 7 RGB texture patch at every
  contour point;
- two 64-dimensional, 14-layer ResGCN branches, with each point linked to the
  eight ordered neighbours on either side;
- a learned 64-dimensional gate that fuses contour and texture features;
- feature inner products divided by `sqrt(64)`, followed by the product of a
  row softmax and a column softmax (dual-softmax, not Sinkhorn);
- focal correspondence loss with alpha 0.55 and gamma 8;
- at inference, fixed epsilon 0.006, one diagonal erosion, one diagonal
  dilation, and SE(2) RANSAC.

The official configuration declares 128 epochs, Adam at 1e-3, weight decay
5e-4, and cosine annealing.  The paper reports matching batch size 20, whereas
the released config defaults to 25.  This adapter follows the paper's batch
size 20 and records the discrepancy.

The released `run.py` invokes, in order, matching training, matching testing,
stage-one feature export, four-GPU distributed searching training, and
searching testing.  The checkout is not a turnkey immutable reproduction:
the stage-two trainer reads an argparse field named `batch_size` that the
released config does not declare, and the final command names a script that is
not present in the repository.  This is another reason to keep the audited
architecture while using an independent Rachel adapter instead of silently
patching the official tree.

Formal preflight requires an explicit official-source checkout.  The runner
verifies that its Git top level is clean and exactly at commit
`e878b781b2b2065a4b7da09d2f639e8f0a35e97a`, then checks pinned SHA-256 values
for `run.py`, the model pipeline and encoder, configuration, focal loss,
evaluation, RANSAC, matching trainer, and matching postprocessor.  The full
source audit is included in `run_config.json`, checkpoints, and the completion
receipt; a constant commit string alone is not treated as provenance.

The official matching stage is trained on known adjacent pairs.  Pair-searching
is a separate second stage: it freezes the matching backbone, applies linear
transformers and an InfoNCE loss, and produces fragment-level retrieval
vectors.  Therefore the Rachel yes/no pair head below is an explicit task
adapter, not an original PairingNet component.

## Required Rachel adaptations

1. **Mask-only input.** Rachel deliberately exposes binary masks rather than
   RGB.  The contour branch uses boundary patches derived from the binary mask.
   The nominal texture branch receives three replicated mask-occupancy channels;
   consequently it is an occupancy/context branch and must not be described as
   an RGB texture encoder.
2. **N = 512.** The frozen Rachel release supplies at most 512 ordered contour
   points, not PairingNet's maximum 2,900.  No denser contour is reconstructed
   from hidden RGB or parent-canvas artefacts.  Rachel padding nodes are masked
   explicitly instead of being encoded at coordinate zero as can happen in the
   released preprocessing path.  Patch encoders gather valid nodes before any
   convolution or batch normalization, and every ResGCN normalization likewise
   operates on valid nodes only.  Ring neighbourhoods wrap at each sample's
   valid contour length, never at the tensor cap.  Thus changing only the
   padding cap cannot change valid features or batch-normalization state.
3. **Known orientation.** The primary estimator is deterministic
   translation-only correspondence consensus.  Rotation error is reported as
   conditioned/not-applicable for this primary path, never as a learned zero.
   An official-style SE(2) RANSAC diagnostic is retained as secondary output.
4. **Pair negatives.** Rachel contains an exact 1:1 positive/negative pair
   population.  Negative pairs receive an all-negative correspondence target,
   and a small pair-evidence head is trained for the yes/no decision.  This is
   necessary to score the Dunhuang task but extends the official matching
   stage.  The optimized objective is PairingNet's focal correspondence loss
   plus binary cross-entropy for this non-official pair head.  Ground-truth
   translation is used for validation, not as a direct differentiable
   regression loss; translation is recovered from the learned
   correspondences.
5. **Split protocol.** Only `pairs/train.jsonl` and `pairs/val.jsonl` are
   accepted.  Their full populations, seed 260831, and deterministic epoch
   ordering match the Rachel experiment.  Train and validation
   `split_unit_id` sets must be disjoint.  The runner never discovers or opens
   a test or real-data manifest.

The released inference threshold is 0.006.  Its custom SE(2) RANSAC first
requires at least 20 candidate correspondences, samples five at a time, uses a
10-pixel correspondence threshold, and permits up to 4,000 iterations and 20
validations.  The adapter preserves 0.006 and 10 px, but the secondary solver
uses OpenCV partial-affine RANSAC and is therefore labelled official-style,
not official-exact.

## Primary metrics

The benchmark's main geometry is directly interpretable in pixels:

- translation error (TE), with coverage and mean/median/p95;
- positive translation success at 2, 5, 8, and 10 pixels;
- assembly-edge precision, recall, and F1 at the same tolerances.

The primary pair operating point is fitted once, after the winner checkpoint
is selected, by the repository's common Rachel validation-only
`fit_pairwise_threshold` routine.  It maximizes equal-lineage-cluster-weighted
F1 on all 3,000 validation rows.  Its artifact binds the ordered validation
scores and labels, winner checkpoint SHA-256, model configuration, and
aggregation rule.  The adapter-native 0.5 threshold for the explicitly
non-official pair head is reported separately as a secondary view; it is not
the main-table operating point and is not attributed to original PairingNet.

For assembly-edge metrics, a ground-truth positive is a true positive only if
the pair is accepted, a pose is available, and TE is within tolerance.  An
accepted positive with the wrong translation is both a false positive (a
wrong placed edge) and a false negative
(the true edge was missed).  A rejected or pose-invalid positive is a false
negative.  A negative becomes an assembly false positive only when it is both
accepted and assigned a valid pose; pair-only false positives remain visible in
the separate pair-decision metrics.  Primary assembly uses the frozen
validation-fitted threshold; the secondary adapter-native view uses 0.5.

## PairingNet-compatible secondary metrics

For positive validation pairs, the runner also reports the repository's metric
definitions:

- RR: fraction with `e_rmse < 4`, where the released code defines
  `e_rmse = sqrt(mean(per-point Euclidean error))`;
- HD: mean symmetric Hausdorff distance between ground-truth seam point sets
  after applying the predicted transform;
- NTE: Euclidean translation error divided by the sum of the two contour
  polygon areas;
- RE: only for successful secondary SE(2) RANSAC estimates.  The primary
  translation-only path records RE as not applicable.

These metrics are retained for method-family comparison, but RR is not used as
a substitute for the primary assembly-edge metrics.

The released postprocessor obtains the paper's single RR number by placing
`e_rmse < 2` and `2 <= e_rmse < 4` in exclusive bins and adding the bins, hence
RR is cumulative `e_rmse < 4`.  RE is the absolute difference of the two
`atan2` rotation angles in radians.  NTE divides Euclidean translation error by
the sum of OpenCV contour polygon areas.  The paper's generated-data result for
the full RGB method is RR 0.835, HD 43.116, RE 0.352, and NTE 7.735e-4.  Those
numbers are not directly comparable to Rachel because the data, resolution,
point density, and modality differ.  This warning is especially important:
the paper's binarized contour+texture ablation falls to RR 0.553, confirming
that mask-only adaptation is a materially different setting rather than a
cosmetic input conversion.

## Selection and stopping

Checkpoint selection never fits or uses a pair threshold.  Its primary key is
the geometric mean of cluster-balanced pair AUROC, cluster-balanced pair
AUPRC, unconditional positive translation success at 8 px, and
PairingNet-compatible RR.  Ties use translation success, RR, AUROC, AUPRC,
lower median TE, then the earlier epoch.  Training runs for at least 20 and at
most 128 epochs, stopping after 12 consecutive validation epochs without a
relative improvement of at least 0.5%.  Reaching the hard cap is reported
explicitly.  No test/real result can affect selection or threshold fitting.

The seed, population, and epoch ordering are frozen.  As in the official graph
implementation, CUDA graph scatter/gather kernels are not claimed to be
bitwise deterministic across hardware/runtime builds; the receipt states this
explicitly.

`last.pt` is the epoch-granular atomic recovery authority.  It contains the
current model, Adam and cosine-scheduler states, stopping/plateau state, exact
winner snapshot, canonical epoch history and its SHA-256, and CPU/CUDA RNG
states.  Explicit `--resume-from` accepts only the exact `.partial` or `.failed`
directory belonging to the requested output.  A process interrupted within an
epoch discards that incomplete work and replays it from the last completed
epoch with the same frozen order and restored RNG state.  Resume also rechecks
the manifests, source code, adaptation
contract, and official checkout before restoring state.

Run-state paths are compared as normalized lexical absolute paths, without
resolving aliases.  Every existing component of the output, `.partial`,
`.failed`, and explicit resume paths is checked with `lstat`; symbolic links
are rejected.  The same guard is reapplied before directory creation and each
atomic run-state move, so an alias cannot write through to another directory.

## Entry points

Audit the formal 24,000/3,000 train/validation release without constructing a
model:

```bash
python -m staging.pairwise_v0_2.baselines.rachel_pairingnet_benchmark \
  --dataset-root /root/autodl-tmp/dataset_rachel_pairwise_n512_v1 \
  --official-source-root /root/autodl-tmp/benchmarks/PairingNet \
  --audit-only
```

Run the train/validation benchmark on the remote GPU:

```bash
python -m staging.pairwise_v0_2.baselines.rachel_pairingnet_benchmark \
  --dataset-root /root/autodl-tmp/dataset_rachel_pairwise_n512_v1 \
  --official-source-root /root/autodl-tmp/benchmarks/PairingNet \
  --output-root /root/autodl-tmp/rachel_pairingnet_benchmark_<run-id> \
  --device cuda:0 --precision fp32
```

Resume that same output after an interruption:

```bash
python -m staging.pairwise_v0_2.baselines.rachel_pairingnet_benchmark \
  --dataset-root /root/autodl-tmp/dataset_rachel_pairwise_n512_v1 \
  --official-source-root /root/autodl-tmp/benchmarks/PairingNet \
  --output-root /root/autodl-tmp/rachel_pairingnet_benchmark_<run-id> \
  --resume-from /root/autodl-tmp/.rachel_pairingnet_benchmark_<run-id>.failed \
  --device cuda:0 --precision fp32
```

The no-clobber output contains `winner.pt`, resumable `last.pt`, both validation
operating-point reports, ordered predictions, `validation_threshold.json`, a
completion receipt, and `inference_contract.json`.  The public restore and
inference functions are `load_frozen_pairingnet_checkpoint`,
`load_frozen_validation_threshold`, and `infer_pairingnet_batch`; their
`method_id` is
`pairingnet_rachel_mask_n512_upright_translation_v1`.

`winner.pt` has the dedicated checkpoint kind `frozen_train_val_winner` and is
bound to the SHA-256 of both the adapter source and this adaptation contract.
The frozen loader verifies all three fields and rejects `last.pt` even though
that recovery payload also contains model weights.  Conversely, strict resume
accepts only checkpoint kind `last_resumable_completed_epoch` and validates its
nested winner against the same source and contract hashes.

`completion_receipt.json` records the method/official commit and source audit,
full population audit, adaptation disclosure, stop reason and convergence
claim, completed/winner epochs, threshold-free selection key, resume count and
policy, frozen validation-threshold artifact, and SHA-256 values for the last
checkpoint, winner checkpoint, threshold file, inference contract, validation
report, and ordered prediction file.  Both sealed-synthetic and real-data
access flags must remain false at this train/validation stage.

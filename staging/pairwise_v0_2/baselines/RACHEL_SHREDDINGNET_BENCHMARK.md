# Rachel ShreddingNet benchmark adaptation

Status: executable train/validation-only runner with CPU-focused contract
tests. No sealed synthetic test or real Dunhuang pair is opened by this
runner. A CUDA smoke command is provided, but no GPU run was started while
preparing this adapter.

## What this benchmark is

This is a same-population Rachel benchmark derived from the official
ShreddingNet release at commit
`0ae3b544ca4e910732f3f459b39aa15cdc62dbcb`.  It retains the released method's
three independently trained stages:

1. coarse contour/support feature learning with two 14-layer ResGCNs,
   bidirectional cross-modal attention, flattened fragment descriptors, and
   InfoNCE;
2. fine contour-point matching with two stacked bidirectional cross-attention
   blocks and dual-softmax similarity; and
3. the frozen fine matcher followed by the released three-layer CNN pair-score
   evaluator.

It is **not** a bitwise reproduction of the paper and must not be reported as
one.  It is a mask-only, N=512, upright Rachel adaptation evaluated on the
same frozen Rachel train/validation pair manifests used by our Sliding Patch +
Sinkhorn arm.

## Executable authority and disclosed conflicts

The executable recipe is the checked-out release and its `art_2192` YAML files:

- coarse: 128 epochs, batch 54, Adam at `1e-4`, weight decay `5e-4`;
- fine: 128 epochs, batch 36, Adam at `1e-3`, weight decay `5e-4`;
- classifier: 128 epochs, batch 20, Adam at `1e-4`, weight decay `5e-4`;
- all stages use cosine annealing.  The release seed is 1024, while the primary
  same-exposure Rachel benchmark uses the already frozen formal comparison seed
  260831.  A 1024 run, if later requested, is an official-seed secondary and
  cannot be presented as the same-seed comparison.

The supplement instead states batch sizes 75/54/20 and epochs 128/128/10.
It also says testing uses the lowest validation loss, while release code selects
coarse `infor_loss` minimum, fine `positive_loss` minimum, and classifier
validation accuracy maximum.  The runner follows the release code's metric
semantics, fixes its epoch-zero checkpoint omission, and records both the
conflict and the fix in the freeze receipt.

The release has another stage-specific validation detail: `PairingTrainer`
constructs both train and validation loaders with `shuffle=True`, whereas
`MatchingTrainer` and `ClassifyTrainer` use `shuffle=False` for validation.
This matters because coarse InfoNCE uses the other members of each validation
batch as negatives, so a different permutation can change validation loss and
the selected winner. The adapter therefore shuffles only coarse validation;
matching and classifier validation remain in manifest order.

For restart-independent reproducibility, coarse validation uses the release
loader's `torch.randperm` permutation mechanism with an isolated generator per
epoch. Its seed is
`formal_seed + 2_000_033 * epoch_one_based`. This preserves the released
shuffle and batch-grouping semantics but deliberately decouples validation
order from unrelated model and training RNG consumption. It is therefore a
deterministic Rachel adaptation, not a claim of bitwise reproduction of the
release's global-RNG stream.

The primary same-data comparison threshold is **not** silently fixed at 0.5.
After all three stage winners are selected, one deterministic replay of the
validation manifest fits the shared equal-cluster-weight maximum-F1 threshold
used by the main Rachel arm. The threshold artifact is bound to the classifier
winner SHA-256 and exact validation pair order. The released 0.5 threshold is
kept as a separately named native secondary result. Threshold fitting never
participates in stage checkpoint selection.

## Rachel-only adaptations

- Only `pairs/train.jsonl` and `pairs/val.jsonl` are accepted.  The runner
  audits that all source-manuscript lineages are disjoint across these two
  manifests.  There is no CLI split selector and no code path that constructs
  the sealed-test or real-data loaders.
- Rachel's independently generated binary 800x800 masks and ordered N=512
  contours are reused.  RGB is never loaded.  To preserve the released
  contour/texture two-branch tensor contract, the support mask is repeated to
  three channels for the former RGB-patch branch; this branch is therefore a
  second geometry branch, not texture evidence.
- The released 7x7 contour sampling, cycle graph with +/-8 neighbours,
  ResGCNs, cross-attention equations, dual-softmax, anti-diagonal morphology,
  and CNN scorer are retained. Patch Conv-BN and every ResGCN block gather only
  valid contour-prefix points; each fragment's ring is constructed from its own
  valid prefix length and the results are scattered back afterwards. Padding
  therefore cannot change valid features or BatchNorm statistics. Non-prefix
  validity masks fail closed.
- Dual-softmax promotes AMP FP16/BF16 logits to FP32 before masking and both
  softmax operations and returns FP32 probabilities. This avoids the FP16
  `-1e9` mask overflow while retaining gradient flow back to AMP logits.
- Fine-stage focal loss retains the release's additive `1e-9` inside
  `log(p + eps)` and `log(1 - p + eps)`. It does not substitute an FP32 upper
  clamp, because `1 - 1e-9` rounds to one and would leave exact-one negative
  probabilities with an infinite loss and gradient.
- Orientation is known.  The primary pose estimator is a deterministic robust
  translation-only consensus over predicted contour correspondences.  A
  deterministic port of the release's five-point, 10-pixel RANSAC is retained
  only as an SE(2) compatibility diagnostic.
- K=2 and K=20 views over the frozen balanced pair list are retained only as
  explicitly named selected-pair-list diagnostics. They are not the paper's
  exhaustive same-parent candidate graph and are never labeled native module
  metrics; K=20 can degenerate to selecting every listed pair.
- Stage three follows the executable release's subtle state behavior: matcher
  parameters are frozen, but the full wrapper is placed in train mode, so its
  BatchNorm running buffers evolve. The complete matcher-plus-classifier stage
  winner is saved; inference restores stage two and then the hash-bound full
  stage-three state.

## Effective batch, microbatch, and resume contract

The release effective batches remain 54/36/20. They are distinct from the GPU
microbatch, which defaults to one and is explicitly recorded together with
AMP and accumulation counts. Coarse training uses a two-pass gradient cache so
all 54 effective examples remain in the InfoNCE denominator before one
optimizer step. Fine and classifier losses are weighted across microbatches
before their one optimizer step.

This preserves population exposure, the full coarse negative set, and
optimizer-step semantics. It is not numerically identical to putting all 54,
36, or 20 examples through the model at once because BatchNorm uses
microbatch statistics. The receipt states this directly; it must not be called
a bitwise release reproduction.

Every training epoch uses the formal Rachel seed-260831 order algorithm. Every
stage history records the training pair-order SHA-256 and the validation
pair-order and positional-permutation SHA-256. Progress checkpoints additionally
bind the stage validation-order contract and the current best epoch's validation
order; winners bind their own epoch's order; completion receipts bind the full
history and winner order. Resume recomputes every persisted train and validation
permutation from the frozen manifest and seed derivation before restoring any
winner. A fresh run accepts only an absent output
path and rejects even a pre-existing empty directory. `--resume` normally
requires and verifies an existing matching run contract; it may also recover
the narrowly controlled empty directory left by a failure between creating the
run root and publishing that contract. Unknown or non-empty contractless states
fail closed. Completed stages are hash-verified rather than overwritten. Progress is atomically
committed before a newly improved winner; resume repairs the narrow intervening
crash window only when that progress state is itself the recorded best epoch.
Checkpoint metadata is decoded through PyTorch's restricted weights-only
loader, including on older runtimes that cannot natively decode raw floats.

Validation-threshold scores are a recoverable part of the freeze transaction.
If a resumed run finds the score JSONL but no freeze, it recomputes every row
from the frozen three-stage winners and reuses the file only when its canonical
bytes, SHA-256, row order, and decoded row contents all match exactly. Any
tamper or nondeterministic difference fails closed instead of overwriting the
artifact.

Every progress checkpoint, winner, completion receipt, and freeze binds a
dedicated artifact kind plus the adapter-source SHA-256, this document's
SHA-256, both train/validation manifest-content SHA-256 values, and both
canonical ordered-pair-ID SHA-256 values. Resume and frozen loading compare all
six provenance hashes strictly. Output paths remain lexical rather than being
resolved through aliases; every existing ancestor and artifact component is
checked with `lstat`, symlinks are rejected, and create-only artifacts use
atomic no-clobber publication.

Before the first output directory or file is created, training and GPU smoke
decode one train positive and negative, recheck both manifest identities,
import the formal PyTorch Geometric dependency, construct all three real stage
models on the target device, and perform finite minimal forwards. Any failed
dependency, decode, construction, or forward leaves no successful output
artifact. GPU smoke likewise publishes its receipt only after all three stage
steps succeed.

## Metrics that answer "did it join correctly?"

The benchmark headline is pairwise assembly-edge precision/recall/F1 at 2, 5,
8, and 10 pixels. An edge is correct only when the pair is truly adjacent,
passes the frozen pair-score threshold, yields a valid translation, and its
translation error is within the stated tolerance. Translation error is the
per-pair `L2(t_hat_rc - t_GT_rc)` in pixels, not eRMSE. Invalid pose estimates
count as unrecovered GT edges; because they emit no transformed edge, they are
not themselves false transformed edges. An accepted pose-valid true edge with
the wrong translation counts both as a false predicted transformed edge and an
unrecovered GT edge.

The available validation manifest is a preselected 1:1 balanced pair list, not
the exhaustive same-parent candidate graph required by the paper's native
coarse/fine/score-evaluator protocol. Consequently the K=2/K=20 outputs live
under `balanced_selected_pair_list_module_like_diagnostics` with descriptive
names such as `coarse_topk_like_selection`; they are non-headline diagnostics
and must not be reported as native ShreddingNet module results. No exhaustive
parent-local graph is claimed or synthesized without sufficient ground truth.

The paper's GA recall criterion (rotation below 5 degrees and translation below
100 pixels) is reported only under
`secondary_official_se2_threshold_pairwise_edge`.  It is explicitly **not GA**:
ShreddingNet GA requires maximum-spanning-tree selection and multi-fragment
global placement, which are outside this Pairwise experiment.

PairingNet's RR criterion based on eRMSE below 4 belongs only to the separately
adapted PairingNet benchmark. It is not relabeled as a ShreddingNet metric.

## Frozen inference contract

Training writes `train_val_freeze.json` and three hash-bound winner
checkpoints.  The method identifier is
`rachel_shreddingnet_maskonly_n512_upright_release0ae3b544_v1`.
`load_frozen_inference(...)` restores all three winners, verifies their hashes,
artifact kinds, adapter/document provenance, and train/validation manifest and
order identities, then exposes `predict_batch(...)` with model-input-only
fields. Its outputs are:

- `method_id`;
- `coarse_score`;
- `pair_score`;
- optional dense `correspondence_probability` and post-morphology
  `correspondence_binary`;
- correspondence/inlier counts;
- `translation_hat_rc` and `translation_valid` (primary, no rotation); and
- `se2_translation_hat_rc`, `se2_rotation_degrees`, and `se2_valid`
  (secondary RANSAC diagnostic).

The forward path never reads labels, correspondence targets, translations, or
lineage metadata.  A later sealed evaluator can therefore make one frozen
forward pass using the existing `RachelBatch` model-facing arrays.

## Commands

Run from the repository root. Replace the dataset and output paths with the
server locations.

```bash
python3 -m staging.pairwise_v0_2.baselines.rachel_shreddingnet_benchmark audit \
  --dataset-root /root/autodl-tmp/dataset_rachel/PREPARED_RELEASE \
  --official-repo tmp/benchmarks/shreddingnet \
  --output /root/autodl-tmp/shreddingnet_audit.json
```

Before a 128-epoch run, execute one release-effective optimizer step per stage
and retain the measured memory/throughput receipt:

```bash
python3 -m staging.pairwise_v0_2.baselines.rachel_shreddingnet_benchmark gpu-smoke \
  --dataset-root /root/autodl-tmp/dataset_rachel/PREPARED_RELEASE \
  --official-repo tmp/benchmarks/shreddingnet \
  --device cuda --num-workers 0 \
  --coarse-microbatch 1 --matching-microbatch 1 --classify-microbatch 1 \
  --output /root/autodl-tmp/shreddingnet_gpu_smoke.json
```

Only after that receipt and this adapter receive review GO, start a fresh run:

```bash
python3 -m staging.pairwise_v0_2.baselines.rachel_shreddingnet_benchmark train \
  --dataset-root /root/autodl-tmp/dataset_rachel/PREPARED_RELEASE \
  --official-repo tmp/benchmarks/shreddingnet \
  --output-root /root/autodl-tmp/rachel_shreddingnet_formal \
  --device cuda --num-workers 0 \
  --coarse-microbatch 1 --matching-microbatch 1 --classify-microbatch 1
```

Resume uses the identical command plus `--resume`. A frozen bundle can be
replayed on validation without any split selector:

```bash
python3 -m staging.pairwise_v0_2.baselines.rachel_shreddingnet_benchmark evaluate-val \
  --dataset-root /root/autodl-tmp/dataset_rachel/PREPARED_RELEASE \
  --freeze /root/autodl-tmp/rachel_shreddingnet_formal/train_val_freeze.json \
  --output /root/autodl-tmp/rachel_shreddingnet_formal/validation_report_replay.json \
  --device cuda
```

The pre-smoke resource envelope is deliberately low-confidence: with AMP and
microbatch one it budgets roughly 4--10 GiB for coarse, 6--16 GiB for fine,
and 7--18 GiB for classifier, with an initial planning range of 0.2--3
pairs/second depending on stage and GPU. Formal ETA must be recomputed from the
GPU smoke receipt; these ranges are not reported measurements.

# S7 dustbin decoding diagnostic — 2026-09-21

Completed in the private research environment. CPU only; about 4.89 s
of script runtime. Reused all 40 previously selected S7 heatmap matrix caches.
No training, new model selection, score recomputation, threshold fitting, or
production decoder change. Existing matrix entries include actual unmatched
masses; no dustbin values were inferred from the real block.

The four variants were declared before running: original Q; an edge retained
only when Qij exceeds both endpoint dustbins; endpoints retained only when
their normalized real-match mass exceeds 0.5; and Q softly weighted by the
geometric mean of the two endpoints' real-match mass fractions. All use the
same Top-2-union/cap512/radius10/min3 translation solver. All 40 baseline poses
reproduced the saved layout to within 0.001 px. Labels and GT enter evaluation
only after all four predictions are fixed.

| Selected population | Original | Edge hard gate | Node hard gate | Soft weight |
|---|---:|---:|---:|---:|
| REAL correct layout <=20px / 12 positive GT pairs | 8 | 0 | 4 | 7 |
| REAL numerical layouts / 8 negative pairs | 8 | 0 | 7 | 8 |
| SIM TEST correct layout <=20px / 4 positive GT pairs | 4 | 0 | 2 | 4 |
| SIM TEST numerical layouts / 4 negative pairs | 4 | 0 | 2 | 4 |
| OOD numerical layouts / 12 positive pairs, no pose GT | 12 | 0 | 5 | 12 |

These are selected diagnostic examples, not representative test metrics.
Numerical layout validity is NOT a pairwise classification decision. In
particular, the OOD row says nothing about placement accuracy. No variant
rescued a baseline-wrong positive pose in this selection. The simple gates
lose useful proposals, and soft weighting did not demonstrate improvement.
This does not rule out learning to use dustbin features or recalibrating
matchability on a separately defined synthetic validation population.

The existing Matcher already supervises unmatched tokens using dustbin NLL.
The unresolved experiment is how to use that information in downstream
candidate decoding/scoring without equating a large unmatched fraction with
a negative pair. Short true seams naturally leave most contour points
unmatched. A future fair Scorer comparison needs the same candidate set,
data, architecture and budget with/without local dustbin evidence; it was
NOT trained in this diagnostic.

The private `results.json` is not published: it contains per-case predictions and source metadata. The executable is [probe.py](../experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/dustbin_probe_v1/probe.py).

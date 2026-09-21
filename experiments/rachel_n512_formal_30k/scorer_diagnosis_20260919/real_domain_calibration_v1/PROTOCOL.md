# Real-domain threshold CV v1 — 2026-09-21

User authorizes cross-source negatives and real-domain threshold calibration.
This is not retraining, changing a checkpoint, or selecting fusion weights.

- The same eight completed local_evidence_v2 C16 models remain frozen.
- Five source-group folds, seed260921. Fit threshold on four folds; evaluate
  exclusively on the fifth, cycling all five. All models share identical folds.
- Two predeclared policies: calibration maximumF1 (ties: precision, then higher
  threshold), and highest threshold achieving calibration recall>=95%.
- Keep Dunhuang295 reviewed positives and39 original within-case negatives.
  Its original469 cross-source negatives link791/803 rows in one component,
  so rebuild469 random cross-source negatives *within* assigned folds. Compare
  original SIMVAL thresholds and real-CV thresholds on this same new803 set.
- Turufan301 known positives +301 cross-source constructed negatives. Negative
  labels follow the user's provenance assumption; they are not manually
  verified geometric negatives. No Layout GT exists for Turufan.
- Group original Dunhuang case IDs; merge aliases sharing existing mask hashes.
  Group Turufan recto/verso/seite suffixes to the same manuscript prefix and
  merge existing identical-mask hashes. No known source or fragment may cross
  calibration/test in an iteration. Metadata cannot prove absence of unknown
  aliases; do not promise a universally contamination-free historical benchmark.
- Split sources before sampling negatives. Negative sampler sees no scores or
  Layout predictions; no score-based selection, hard-negative mining or test
  threshold optimization. Keep every positive and every sampled negative.
- Prepared fragment representations are unchanged and normalization stays at
  original-case level, as in the earlier cross-source Dunhuang benchmark.
  Reuse original positive/strict-negative scores; infer only new negatives.
- Report pooled out-of-fold predictions (each pair once), per-fold thresholds,
  confusion counts, F1/Recall/Accuracy, unchanged-score AUROC and DH correct
  Layout acceptance. Standard fold spread is not an independence-based CI.
- Any all-real refit threshold is for future unseen pairs only; its in-sample
  metric is not reported as CV performance and no deployed threshold is changed.
- These datasets already informed earlier model design and human exclusions.
  This experiment isolates threshold fitting from its held-out fold; it does
  not restore untouched external-test status or prove overfitting from a high
  numeric threshold alone. Construction changes class priors and difficulty;
  conclusions apply to this benchmark, not arbitrary deployment prevalence.

No GPU training, no new HTML, no changes to previous results.

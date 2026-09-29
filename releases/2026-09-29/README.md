# 2026-09-29 research code release

Start with [the self-contained research handoff](../../docs/RESEARCH_HANDOFF_20260929.md).

The root `experiments/` tree contains development modules and their Python dependency closure. It is **not** a universal replacement for the source bound to every historical checkpoint. The six code capsules below preserve the relevant original Python files byte-for-byte, at their original import paths; `manifests/` records each SHA256. They omit unrelated scripts, datasets, weights, machine credentials, annotation stores and run outputs.

| Capsule | Meaning |
|---|---|
| `mergefix_complex` | Common-pose merge repair; directional evidence and complex head |
| `threshold_complex` | Fixed 16px diameter; exact union Q; complex head |
| `simple_complex` | Independent simple builder (16px radius), still the complex neural head |
| `binary_e32_micro32` | Frozen v14 E32; fresh Patch/Context or Q/statistics MLP; single-GPU effective batch32 |
| `v17_matcher_binary` | v17 random-start Matcher then frozen-Matcher Patch/Context head source |
| `pooling_controls` | Fresh mean-only and bounded-sum heads; results pending at the handoff cutoff |

The base directory is `threshold_complex`. Other capsules contain only changed/new files in readable overlays. The materializer selects exactly the manifest's file set, so files present only in the base cannot leak into another variant.

```bash
python tools/materialize_research_snapshot.py \
  --variant binary_e32_micro32 --output /tmp/dh-binary-e32-source
cd /tmp/dh-binary-e32-source
python -m unittest experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.binary_scorer_v1.test_binary
```

Use Python3.10/3.11, the repository requirements and a suitable PyTorch build. CPU synthetic tests require no private data. A full training/evaluation run additionally requires the matching **data contract, prepared data, geometry calibration, checkpoint, real split and external launcher**, which are not bundled. Source hashes must be checked against the target checkpoint/run binding; an arbitrary prepared snapshot is not permission to bypass that check. Historical inherited `launch.py` files are not recommended entrypoints. Never point a new experiment at an existing output directory.

The source capsules and their unit tests reproduce code behavior, not the published performance figures by themselves. Aggregate, reviewed numbers and provenance hashes are in [the evidence extract](../../docs/evidence/research_handoff_20260929.json). Real masks, per-case trajectories and editable HTML remain in the user's local workspace, not in this public repository.

Release verification: all six capsules passed manifest SHA256 and Python syntax checks, plus 102 selected synthetic CPU tests; the safe materializer passed 4 additional tests. See [the 106-test receipt](verification.json), including exact module names and the local dependency versions. For the historical `decoder_readout_v1.test_controls` module, also put its own directory on `PYTHONPATH` because it uses script-local imports. These are packaging smoke checks, not a rerun of every historical test or any GPU experiment.

# Third-party sources and publication boundaries

## ShreddingNet

Upstream: <https://github.com/tqychy/shreddingnet>, pinned reference commit
`0ae3b544ca4e910732f3f459b39aa15cdc62dbcb`.

The project-specific architecture port and benchmark adapter live at
`staging/pairwise_v0_2/baselines/rachel_shreddingnet_benchmark.py`.
The upstream MIT license is preserved in [licenses/ShreddingNet-MIT.txt](licenses/ShreddingNet-MIT.txt).
The complete upstream checkout is not redistributed here. Training provenance
checks may require a separate checkout at the pinned revision.

## PairingNet

Upstream: <https://github.com/zhourixin/PairingNet>, pinned reference commit
`e878b781b2b2065a4b7da09d2f639e8f0a35e97a`.

`staging/pairwise_v0_2/baselines/rachel_pairingnet_benchmark.py` is the
project's explicitly adapted mask-only benchmark implementation. The official
source tree and pretrained weights are not bundled. No project-wide license
here grants rights to upstream code. Consult the authors' terms before
redistributing their materials.

Both adapters change the input/task/supervision; their results must not be
presented as native-paper reproductions. Detailed adaptation notes accompany
the source files.

## Colleague materials and research data

The privately supplied `geo_attn` and training packages, vendor trees, trained
weights, raw fragment images, human-review labels, and per-case result files
are intentionally excluded. G0/G1 in this repository are project experiment
implementations, not redistribution or a claimed reproduction of the
colleague's original training package.

No new license for the project's original code is selected in this snapshot.
Repository publication by itself must not be described as an MIT/Apache grant.


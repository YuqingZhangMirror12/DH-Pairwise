"""New TEST manifest, unchanged real populations, GT joined after prediction."""
from consensus_binary_eval_common import evaluate as common
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.aggressive_binary_v1.admission import REVISION
from .contracts import SIM_SPLIT, read, sha


def load_population(split, contract, batch_size):
    if split in ('dunhuang_cv', 'turufan'):
        return common.load_population(split, contract, batch_size)
    if split != SIM_SPLIT or contract.get('augmentation_revision') != REVISION:
        raise ValueError('only the bound new TEST or registered real populations are allowed')
    spec = contract['test']['mixed']
    manifest = read(spec['path'])
    if (sha(spec['path']) != spec['sha256'] or spec.get('pair_count') != 3000
            or manifest.get('split') != 'test' or manifest.get('augmentation_revision') != REVISION):
        raise ValueError('new aggressive TEST identity differs')
    dataset = common.Dataset(spec['path'], spec['sha256'])
    if len(dataset) != 3000:
        raise ValueError('new aggressive TEST requires all 3000 pairs')

    def batches():
        for start in range(0, len(dataset), batch_size):
            ids = range(start, min(start + batch_size, len(dataset)))
            yield [dataset.entries[i] for i in ids], common.collate([dataset[i] for i in ids])

    return dict(pairs=dataset.entries), batches(), dict(manifest=spec['path'], manifest_sha256=spec['sha256'],
        # These descriptive counts are not required by the admitted v17
        # contract. Missing metadata must not become zero or an invented count
        # of independent manuscripts, and must not prevent frozen inference.
        source_count=spec.get('source_count'), base_pair_count=spec.get('base_pair_count'),
        optional_count_metadata_available=all(k in spec for k in ('source_count', 'base_pair_count')),
        preprocessing='immutable human-approved aggressive TEST archives',
        augmentation_revision=REVISION), dataset


def attach_targets(predictions, meta, split, dataset=None, ground_truth=None):
    if split not in (SIM_SPLIT, 'dunhuang_cv', 'turufan'):
        raise ValueError('unregistered target population')
    # The common helper's legacy SIM name chooses only its generic archive-GT
    # join. It neither opens the old v14 manifest nor chooses model inputs.
    return common.attach_targets(predictions, meta,
        'sim_test_v14' if split == SIM_SPLIT else split, dataset, ground_truth)

"""Match canonical preparation's per-entry seeds without changing any input.

The archived population loader uses manifest.seed in a negative provenance key;
prepare_data uses entry.generation_seed. The strict generator has per-entry
seeds. Override only this identity reconstruction; retain all hashes, official
loading, item checks, groupings and prediction semantics. Never patch source13.
"""


def canonical_rows(api, entries, audited, role, seed, loader):
    return [api.canonical_entry(e, audited[e['pair_id']], role,
                               e.get('generation_seed', seed), loader) for e in entries]


def load_population(api, split, plan, source_root):
    api.require(split in api.STRAIGHT, 'bridge applies only to registered straight heldouts')
    canonical, manifest, audit, _ = api.straight_manifest(plan['canonical_straight'], split)
    view = plan['straight'][split]; role = split[len('sim_straight_'):]
    api.require(api.digest([r['pair_id'] for r in manifest['entries']]) == view['pair_ids_sha256'],
                'heldout membership changed')

    class PerEntrySeedPopulation(api.StraightPopulation):
        # __getitem__ is inherited byte-for-byte, including every file SHA check.
        def __init__(self):
            self.entries = manifest['entries']; self.audited = {r['pair_id']:r for r in audit['records']}
            api.require(len(self.audited) == len(self.entries), 'duplicate/missing audited IDs')
            self.loader = api.bound_module('staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset',
                                          source_root).load_sample
            rows = canonical_rows(api, self.entries, self.audited, role, manifest['seed'], self.loader)
            api.require(api.digest(rows) == canonical['populations'][role]['canonical_row_identity_sha256'],
                        'actual heldout canonical inputs/targets differ')

    dataset = PerEntrySeedPopulation()
    data = api.bound_module(api.BASE+'s7_consensus_v1.data', source_root)
    def batches():
        for start in range(0, len(dataset), 8):
            ids = range(start, min(start+8, len(dataset)))
            yield [dataset.entries[i] for i in ids], data.collate([dataset[i] for i in ids])
    source = dict(manifest=view['manifest_path'], manifest_sha256=view['manifest_sha256'],
        canonical_rows_sha256=view['canonical_rows_sha256'], preprocessing='unchanged official materialized heldout',
        simulation_revision='v4.2-reuse-strict-MD', actual_real_donor_used=False,
        provenance_seed_policy='entry.generation_seed with manifest.seed fallback, identical to canonical preparation',
        pixel_or_supervision_changes=False)
    return dict(pairs=dataset.entries), batches(), source, dataset

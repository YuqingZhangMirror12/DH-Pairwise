"""Insert straight-seam optimizer batches without replacing a base exposure.

This builds a deterministic plan only. Pixel/source admission and a dedicated
v2 training runner are mandatory before execution. Existing immutable ledgers
are never edited. A longer plan is NOT a same-compute comparison with B0/B2.
"""
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

from ..curriculum_training_v1.exposure import (
    ORDERS, digest, integer, is_sha, permutation, repeated_draws,
)


@dataclass(frozen=True)
class StraightSampleRef:
    pair_id: str
    source_base_key: str
    model_input_sha256: str
    sample_path: str
    sample_sha256: str
    label: bool
    split: str = 'train'
    stage: str = 'straight_seam'


@dataclass(frozen=True)
class AdditiveLedger:
    catalog: tuple
    effective_batch: int
    seed: int
    curriculum: tuple
    mixed: tuple
    # Slot metadata is common to both orderings. A tuple is (kind, base_index
    # or extra_index, base_updates_completed_before_this_optimizer_update).
    slots: tuple
    base_sha256: str
    base_catalog_size: int
    windows: tuple
    sha256: str

    @property
    def total_updates(self):
        return len(self.slots)

    def sequence(self, order):
        if order not in ORDERS:
            raise ValueError('unknown presentation order')
        return getattr(self, order)

    def base_completed(self, completed_updates):
        integer(completed_updates, 'completed_updates')
        if completed_updates > self.total_updates:
            raise ValueError('completed cursor exceeds registered total')
        if completed_updates == self.total_updates:
            return sum(kind == 'base' for kind, _, _ in self.slots)
        return self.slots[completed_updates][2]

    def summary(self):
        base = sum(kind == 'base' for kind, _, _ in self.slots)
        extra = self.total_updates - base
        return dict(schema='straight-seam-additive-ledger/1', sha256=self.sha256,
            base_ledger_sha256=self.base_sha256, original_updates=base,
            added_straight_updates=extra, total_updates=self.total_updates,
            original_exposures=base * self.effective_batch,
            added_straight_exposures=extra * self.effective_batch,
            total_exposures=self.total_updates * self.effective_batch,
            original_exposures_preserved=True, original_batch_order_preserved=True,
            same_budget_as_base=False,
            learning_rate_clock='completed original updates; extra steps do not advance it',
            windows=[dict(original_start=a, original_stop=b, added_updates=e,
                          fraction_added=e / (b - a + e)) for a, b, e in self.windows],
            gpu_started=False)


def build_additive_ledger(base, straight_samples, windows, *, seed):
    """Each window explicitly gives (base_start, base_stop, added_updates).

    Extra batches start at base position zero within EVERY window, including
    the first one. All added examples are TRAIN; neither CAL nor SELECT can
    enter through this adapter. Actual source-family disjointness is admission's
    responsibility, not something sample IDs alone can establish.
    """
    integer(seed, 'seed')
    samples = tuple(sorted(straight_samples, key=lambda r: r.pair_id))
    if not samples or any(not isinstance(r, StraightSampleRef) for r in samples):
        raise ValueError('admitted straight sample references required')
    seen_ids = {r.pair_id for r in base.catalog}
    seen_inputs = {r.model_input_sha256 for r in base.catalog}
    for row in samples:
        if (row.stage != 'straight_seam' or row.split != 'train' or type(row.label) is not bool
                or not row.pair_id or not row.source_base_key
                or not is_sha(row.model_input_sha256) or not is_sha(row.sample_sha256)
                or not PurePosixPath(row.sample_path).is_absolute()):
            raise ValueError('invalid or held-out straight reference')
        if row.pair_id in seen_ids or row.model_input_sha256 in seen_inputs:
            raise ValueError('duplicate identity or actual six-input hash')
        seen_ids.add(row.pair_id)
        seen_inputs.add(row.model_input_sha256)
    windows = tuple(tuple(window) for window in windows)
    previous = 0
    for window in windows:
        if len(window) != 3:
            raise ValueError('explicit start/stop/extra window required')
        start, stop, extra = window
        integer(start, 'window start'); integer(stop, 'window stop', 1)
        integer(extra, 'extra updates', 1)
        if start != previous or stop <= start or stop > base.total_updates:
            raise ValueError('windows must exactly partition all original updates')
        previous = stop
    if not windows or previous != base.total_updates:
        raise ValueError('windows do not cover the original budget')
    slots = []
    extra_index = 0
    for start, stop, count in windows:
        length = stop - start
        by_position = Counter((j * length) // count for j in range(count))
        for original in range(start, stop):
            for _ in range(by_position[original - start]):
                slots.append(('straight', extra_index, original))
                extra_index += 1
            slots.append(('base', original, original))
    if slots[0][0] != 'straight':
        raise AssertionError('straight examples were postponed until a later stage')
    half = base.effective_batch // 2
    by_label = {label: [len(base.catalog) + i for i, r in enumerate(samples) if r.label == label]
                for label in (False, True)}
    if extra_index * half < max(map(len, by_label.values())):
        raise ValueError('extra budget would silently omit admitted straight samples')
    streams = {label: repeated_draws(ids, extra_index * half, seed, 'straight', label, half)
               for label, ids in by_label.items()}
    extra_batches = []
    for step in range(extra_index):
        batch = streams[False][step * half:(step + 1) * half] + streams[True][step * half:(step + 1) * half]
        extra_batches.append(tuple(batch[j] for j in permutation(range(base.effective_batch), seed, 'straight', step)))
    sequences = {}
    for order in ORDERS:
        original_sequence = base.sequence(order)
        sequence = []
        recovered = []
        for kind, index, _ in slots:
            if kind == 'base':
                batch = original_sequence[index * base.effective_batch:(index + 1) * base.effective_batch]
                recovered.extend(batch)
            else:
                batch = extra_batches[index]
            sequence.extend(batch)
        if tuple(recovered) != tuple(original_sequence):
            raise AssertionError('an original exposure/order changed')
        sequences[order] = tuple(sequence)
    catalog = base.catalog + samples
    record = dict(schema='straight-seam-additive-ledger/1', base_sha256=base.sha256,
                  catalog=[asdict(r) for r in catalog], windows=windows, seed=seed,
                  effective_batch=base.effective_batch, slots=slots, **sequences)
    return AdditiveLedger(catalog, base.effective_batch, seed, sequences['curriculum'],
                          sequences['mixed'], tuple(slots), base.sha256,
                          len(base.catalog), windows, digest(record))

"""One exposure ledger, two presentation orders, exact update-boundary cursors.

This module does not certify pixels, choose stage budgets, or start training.
It consumes explicitly admitted sample references and explicitly chosen budgets.
The same global ledger can drive a two-GPU Matcher or a single-GPU Scorer.
"""
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math

STAGES = ('v17_filtered', 'v17.5', 'v18')
ORDERS = ('curriculum', 'mixed')
MAX_UPDATES = 36000


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + ' must be an integer >= ' + str(minimum))
    return value


def is_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


@dataclass(frozen=True)
class SampleRef:
    stage: str
    pair_id: str
    source_base_key: str
    model_input_sha256: str
    sample_path: str
    sample_sha256: str
    label: bool


def canonical_catalog(samples):
    """Reject cross-stage identities and duplicate actual inputs, not rare rows."""
    samples = tuple(samples)
    if any(not isinstance(r, SampleRef) or r.stage not in STAGES for r in samples):
        raise ValueError('unknown stage or invalid sample reference')
    rows = tuple(sorted(samples, key=lambda r: (STAGES.index(r.stage), r.pair_id)))
    if not rows or set(r.stage for r in rows) != set(STAGES):
        raise ValueError('all three explicit stages are required')
    ids, inputs, bases = set(), set(), {}
    for row in rows:
        if (not isinstance(row, SampleRef) or type(row.label) is not bool
                or not all(isinstance(v, str) and v for v in
                           (row.pair_id, row.source_base_key, row.sample_path))
                or not is_sha(row.model_input_sha256) or not is_sha(row.sample_sha256)):
            raise ValueError('invalid admitted sample reference')
        if row.pair_id in ids or row.model_input_sha256 in inputs:
            raise ValueError('duplicate Pair ID or actual model input')
        previous = bases.get(row.source_base_key)
        if previous is not None and (previous != row.stage or row.stage != 'v17_filtered'):
            raise ValueError('original pair identity reused across stages or hard examples')
        ids.add(row.pair_id); inputs.add(row.model_input_sha256)
        bases[row.source_base_key] = row.stage
    for stage in STAGES:
        if {r.label for r in rows if r.stage == stage} != {False, True}:
            raise ValueError('each stage must contain positives and negatives')
    return rows


def permutation(values, *key):
    """Hash ordering is independent of Python/NumPy RNG implementation versions."""
    prefix = json.dumps(key, ensure_ascii=False, separators=(',', ':')).encode() + b'\0'
    return sorted(values, key=lambda n: (hashlib.sha256(prefix + str(n).encode()).digest(), n))


def repeated_draws(indices, count, seed, stage, label, half_batch):
    result = []
    cycle = 0
    if len(indices) < half_batch:
        raise ValueError('too few unique class samples for one global batch')
    while len(result) < count:
        order = permutation(indices, seed, stage, label, cycle)
        remainder = len(result) % half_batch
        if remainder:
            # A cycle boundary may fall inside a batch. Move enough not-yet-used
            # IDs to its front, while retaining every ID exactly once per cycle.
            used = set(result[-remainder:])
            prefix = [i for i in order if i not in used][:half_batch - remainder]
            selected = set(prefix)
            order = prefix + [i for i in order if i not in selected]
        result.extend(order)
        cycle += 1
    return result[:count]


@dataclass(frozen=True)
class ExposureLedger:
    catalog: tuple
    stage_updates: tuple
    effective_batch: int
    seed: int
    curriculum: tuple
    mixed: tuple
    sha256: str

    @property
    def total_updates(self):
        return sum(self.stage_updates)

    def sequence(self, order):
        if order not in ORDERS:
            raise ValueError('unknown presentation order')
        return getattr(self, order)

    def summary(self):
        counts = Counter(self.curriculum)
        cutpoints = []
        completed = 0
        for stage, updates in zip(STAGES, self.stage_updates):
            completed += updates
            stage_rows = [i for i, r in enumerate(self.catalog) if r.stage == stage]
            cutpoints.append(dict(stage=stage, updates=updates, end_update=completed,
                unique_pairs=len(stage_rows), unique_original_pairs=len({self.catalog[i].source_base_key for i in stage_rows}),
                positives=sum(self.catalog[i].label for i in stage_rows),
                exposures=sum(counts[i] for i in stage_rows),
                minimum_per_sample_exposures=min(counts[i] for i in stage_rows),
                maximum_per_sample_exposures=max(counts[i] for i in stage_rows)))
        return dict(schema='curriculum-exposure-ledger/1', sha256=self.sha256,
            seed=self.seed, effective_batch=self.effective_batch,
            total_updates=self.total_updates, total_exposures=len(self.curriculum),
            stage_plan=cutpoints, per_sample_counts_identical=Counter(self.mixed) == counts,
            both_orders_balanced_per_update=True, same_global_batches=True,
            mixing_unit='global optimizer batch; batch contents unchanged', gpu_started=False)


def build_ledger(samples, stage_updates, seed, effective_batch=32, max_updates=MAX_UPDATES):
    """Choose counts externally; mixed permutes the same balanced global batches."""
    rows = canonical_catalog(samples)
    integer(seed, 'seed'); integer(effective_batch, 'effective_batch', 2)
    integer(max_updates, 'max_updates', 1)
    if effective_batch % 2 or max_updates > MAX_UPDATES:
        raise ValueError('even balanced batch and registered total update cap required')
    if set(stage_updates) != set(STAGES):
        raise ValueError('explicit update budget for each stage required')
    steps = tuple(integer(stage_updates[s], 'stage updates', 1) for s in STAGES)
    if sum(steps) > max_updates:
        raise ValueError('total budget exceeds single-module cap, not a cap per stage')
    half = effective_batch // 2
    curriculum = []
    for stage, updates in zip(STAGES, steps):
        by_label = {label: [i for i, r in enumerate(rows) if r.stage == stage and r.label == label]
                    for label in (False, True)}
        if updates * half < max(map(len, by_label.values())):
            raise ValueError('budget would silently leave some admitted samples unseen')
        streams = {label: repeated_draws(indices, updates * half, seed, stage, label, half)
                   for label, indices in by_label.items()}
        for step in range(updates):
            batch = streams[False][step * half:(step + 1) * half] + streams[True][step * half:(step + 1) * half]
            curriculum.extend(batch[j] for j in permutation(range(effective_batch), seed, stage, step, 'batch'))
    # Holding the global batches identical isolates their order from composition.
    mixed = []
    for step in permutation(range(sum(steps)), seed, 'mixed_global_batches'):
        mixed.extend(curriculum[step * effective_batch:(step + 1) * effective_batch])
    if Counter(curriculum) != Counter(mixed):
        raise AssertionError('per-sample exposure counts changed')
    binding = dict(schema='curriculum-exposure-ledger/1', catalog=[asdict(r) for r in rows],
        stage_updates=steps, seed=seed, effective_batch=effective_batch,
        curriculum=curriculum, mixed=mixed)
    return ExposureLedger(rows, steps, effective_batch, seed, tuple(curriculum), tuple(mixed), digest(binding))


class RankMicrobatches:
    """DataLoader batch_sampler, without DistributedSampler padding or reshuffle.

    Cursor units are *completed optimizer updates*, never prefetched batches.
    Saving model/optimizer/RNG states at the same boundary remains the trainer's job.
    """
    def __init__(self, ledger, order, completed_updates, rank, world_size, microbatch, accumulate):
        self.ledger = ledger; self.order = order
        self.sequence = ledger.sequence(order)
        self.completed = integer(completed_updates, 'completed_updates')
        self.rank = integer(rank, 'rank')
        self.world = integer(world_size, 'world_size', 1)
        self.micro = integer(microbatch, 'microbatch', 1)
        self.accumulate = integer(accumulate, 'accumulate', 1)
        if (self.completed > ledger.total_updates or self.rank >= self.world
                or self.world * self.micro * self.accumulate != ledger.effective_batch):
            raise ValueError('cursor or effective batch/topology mismatch')

    def __len__(self):
        return (self.ledger.total_updates - self.completed) * self.accumulate

    def __iter__(self):
        for update in range(self.completed, self.ledger.total_updates):
            start = update * self.ledger.effective_batch
            for accumulation in range(self.accumulate):
                offset = start + (accumulation * self.world + self.rank) * self.micro
                yield list(self.sequence[offset:offset + self.micro])

    def cursor(self, completed_updates):
        integer(completed_updates, 'completed_updates')
        if not self.completed <= completed_updates <= self.ledger.total_updates:
            raise ValueError('cannot record a backward or out-of-budget cursor')
        return dict(schema='curriculum-sampling-cursor/1', ledger_sha256=self.ledger.sha256,
            order=self.order, completed_updates=completed_updates,
            completed_exposures=completed_updates * self.ledger.effective_batch,
            world_size=self.world, microbatch=self.micro, accumulate=self.accumulate)

    @classmethod
    def from_cursor(cls, ledger, cursor, rank, order, world_size, microbatch, accumulate):
        expected = dict(schema='curriculum-sampling-cursor/1', ledger_sha256=ledger.sha256,
                        order=order, world_size=world_size, microbatch=microbatch, accumulate=accumulate)
        if any(cursor.get(k) != v for k, v in expected.items()):
            raise ValueError('resume ledger/order/topology changed')
        result = cls(ledger, order, cursor['completed_updates'], rank, world_size, microbatch, accumulate)
        if cursor != result.cursor(result.completed):
            raise ValueError('cursor has inconsistent exposure count or unexpected fields')
        return result


def learning_rate_at(update, knots):
    """Explicit common global schedule; no metric-dependent per-arm changes."""
    integer(update, 'global update')
    if not knots or knots[0][0] != 0:
        raise ValueError('learning-rate plan must start at global update0')
    previous = -1
    result = None
    for at, value in knots:
        integer(at, 'learning-rate knot')
        if at <= previous or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError('invalid common learning-rate schedule')
        if at <= update:
            result = float(value)
        previous = at
    return result

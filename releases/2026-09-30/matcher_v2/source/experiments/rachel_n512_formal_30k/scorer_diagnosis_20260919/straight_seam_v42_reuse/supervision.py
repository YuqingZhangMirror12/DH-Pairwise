"""Reconstruct reference damage intervals without altering its pixel generator.

These quantities are supervision-only. They must never become model inputs.
"""
import numpy as np
from scipy import ndimage


class RecordingRNG:
    """Record the exact scalar draws consumed by frozen v4.2 misfit()."""
    def __init__(self, rng):
        self.rng = rng
        self.events = []

    def uniform(self, low=0., high=1., size=None):
        if size is not None:
            raise ValueError('frozen reference misfit expects scalar draws')
        value = self.rng.uniform(low, high)
        self.events.append(('uniform', float(low), float(high), float(value)))
        return value

    def random(self, size=None):
        if size is not None:
            raise ValueError('frozen reference misfit expects scalar draws')
        value = self.rng.random()
        self.events.append(('random', float(value)))
        return value


def parse_events(events, parameters, smin, smax):
    """Parse draws in frozen misfit's control-flow order, not from output depths."""
    index = 0

    def uniform(low, high):
        nonlocal index
        event = events[index]; index += 1
        if event[0] != 'uniform' or not np.allclose(event[1:3], [low, high], rtol=0, atol=1e-8):
            raise ValueError('reference RNG protocol changed')
        return event[3]

    def random():
        nonlocal index
        event = events[index]; index += 1
        if event[0] != 'random':
            raise ValueError('reference RNG protocol changed')
        return event[1]

    def segments(coverage):
        result = []; covered = 0.
        while covered < coverage*(smax-smin):
            width = uniform(20, 90)
            start = uniform(smin, max(smin, smax-width))
            result.append((start, start+width)); covered += width
        return result

    wear = [uniform(*parameters['wear']), uniform(*parameters['wear'])] if parameters['wear'] else [0., 0.]
    intervals = []
    gap_coverage = uniform(*parameters['gap_cover'])
    for start, stop in segments(gap_coverage):
        depth = uniform(*parameters['gap_depth']); side = int(random() >= .5)
        intervals.append(dict(kind='gap', start=start, stop=stop, depth=depth, side=side))
    overlap_active = random() < parameters['ov_p']
    overlap_coverage = None
    if overlap_active:
        overlap_coverage = uniform(*parameters['ov_cover'])
        for start, stop in segments(overlap_coverage):
            depth = uniform(*parameters['ov_depth']); side = int(random() >= .5)
            intervals.append(dict(kind='overlap', start=start, stop=stop, depth=depth, side=side))
    if index != len(events):
        raise ValueError('unconsumed reference RNG draws')
    return dict(intervals=intervals, wear=wear, gap_coverage_draw=gap_coverage,
                overlap_active=overlap_active, overlap_coverage_draw=overlap_coverage)


def replay_final_attempt(reference, entry, seed):
    """Replay only the stored accepted attempt, verify its pixels separately."""
    kind = entry['recipe'][-1]
    if not entry['label']:
        raise ValueError('negative pairs have no common seam')
    idx = int(entry['id'].rsplit('_', 1)[1])
    rng = np.random.default_rng([seed, reference.SPLIT_SEED[entry['split']], 'MJR'.index(kind), 1, idx, entry['tries']-1])
    original_misfit = reference.misfit
    trace = {}

    def record(a, b, geom, rng, kind):
        seam = a & ndimage.binary_dilation(b)
        recorder = RecordingRNG(rng)
        result = original_misfit(a, b, geom, recorder, kind)
        if seam.sum() >= 50:
            along = (np.column_stack(np.nonzero(seam))-geom[0]) @ geom[1]
            trace.update(parse_events(recorder.events, reference.MIS[kind], float(along.min()), float(along.max())))
        else:
            assert not recorder.events
            trace.update(intervals=[], wear=[0., 0.], gap_coverage_draw=None, overlap_active=False, overlap_coverage_draw=None)
        return result

    reference.misfit = record
    try:
        a, b, meta = reference.piece_pair(kind, entry['meta']['base'], rng, reference.SRCS)
    finally:
        reference.misfit = original_misfit
    if a is None:
        raise ValueError('accepted reference attempt no longer reproduces')
    for key in ('source_mask', 'source_family', 'scale', 'rough_std_px', 'rough_corr_px', 'bend_amp_px', 'misfit'):
        if meta.get(key) != entry['meta'].get(key):
            raise ValueError('accepted metadata differs: '+key)
    return a, b, trace


def unpack(proof, name):
    shape = tuple(int(v) for v in proof['shape'])
    value = np.unpackbits(proof[name], axis=-1)[:, :shape[1]].astype(bool)
    if value.shape != shape:
        raise ValueError('malformed packed pixel provenance')
    return value


def observed_nick_intervals(proof, centre, axis, distance_to_seam, band_px=10.):
    removed = np.zeros(distance_to_seam.shape, bool)
    for side in 'ab':
        removed |= unpack(proof, 'post_misfit_'+side) & ~unpack(proof, 'final_parent_'+side)
    removed &= distance_to_seam <= band_px
    if not removed.any():
        return []
    bins = np.unique(np.floor((np.column_stack(np.nonzero(removed))-centre) @ axis).astype(int))
    groups = np.split(bins, np.flatnonzero(np.diff(bins) > 1)+1)
    return [(float(g[0]), float(g[-1]+1)) for g in groups]


def build_supervision(sample, proof, trace, projected_interval_damage):
    """Exclude the union of both sides' damaged intervals from both targets."""
    ca, cb = unpack(proof, 'cut_a'), unpack(proof, 'cut_b')
    seam = (ca & ndimage.binary_dilation(cb)) | (cb & ndimage.binary_dilation(ca))
    if not seam.any():
        raise ValueError('no original common cutting interface')
    distance = ndimage.distance_transform_edt(~seam)
    centre, axis = proof['cut_centre'], proof['cut_axis']
    intervals = [(x['start'], x['stop']) for x in trace['intervals']]
    nick_intervals = observed_nick_intervals(proof, centre, axis, distance)
    intervals += nick_intervals
    parents = []; eligible = []; damaged = []
    for side in 'ab':
        parent = np.asarray(getattr(sample, 'points_rc_'+side), np.float64)-proof['shift_'+side]
        valid = np.asarray(getattr(sample, 'contour_valid_'+side), bool)
        nearest = ndimage.map_coordinates(distance, parent.T, order=1, mode='constant', cval=1e9)
        on_seam = valid & (nearest <= 10.)
        along = (parent-centre) @ axis
        damage = on_seam & projected_interval_damage(along, intervals, margin_px=3.)
        parents.append(parent); eligible.append(on_seam); damaged.append(damage)
    # Filter the existing official correspondences rather than rerunning NN
    # after float32 serialization and potentially changing a numerical tie.
    ta = np.full(len(parents[0]), -1, np.int64)
    tb = np.full(len(parents[1]), -1, np.int64)
    ta[damaged[0]] = -2; tb[damaged[1]] = -2
    old_i = np.flatnonzero(sample.target_a >= 0); old_j = sample.target_a[old_i]
    np.testing.assert_array_equal(sample.target_b[old_j], old_i)
    for i, j in zip(old_i, old_j):
        if not (eligible[0][i] and eligible[1][j]):
            continue
        if damaged[0][i] or damaged[1][j]:
            ta[i] = tb[j] = -2
            continue
        ta[i], tb[j] = j, i
    ia = np.flatnonzero(ta >= 0); ib = ta[ia]
    if np.any(damaged[0][ia]) or np.any(damaged[1][ib]):
        raise ValueError('damaged point retained as a target')
    np.testing.assert_array_equal(tb[ib], ia)
    tolerance = 3.+sum(trace['wear'])
    if len(ia) and np.linalg.norm(parents[0][ia]-parents[1][ib],axis=1).max() > tolerance+1e-4:
        raise ValueError('retained correspondence exceeds reference tolerance')
    affected = damaged[0][old_i] | damaged[1][old_j]
    outside = ~eligible[0][old_i] | ~eligible[1][old_j]
    report = dict(correspondence_count=len(ia), distance_tolerance_px=tolerance,
                  original_correspondences=len(old_i), original_matches_touching_damage=int(affected.sum()),
                  original_matches_outside_common_seam=int(outside.sum()),
                  ignored_tokens_a=int((ta == -2).sum()), ignored_tokens_b=int((tb == -2).sum()),
                  sufficient_correspondences=len(ia) >= 8, main_intervals=trace['intervals'], nick_intervals=nick_intervals,
                  band_px=10., transition_and_contour_smoothing_margin_px=3.,
                  target_semantics='only filter original reciprocal matches; damage -2; other unmatched -1',
                  six_model_inputs_unchanged=True, training_admitted=False)
    return dict(target_a=ta, target_b=tb, eligible_a=eligible[0], eligible_b=eligible[1],
                damaged_a=damaged[0], damaged_b=damaged[1]), report

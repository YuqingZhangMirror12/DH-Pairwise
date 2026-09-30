"""GT-aligned geometry proxies, measured without Matcher Q or proposals.

An observed proximity band is NOT a semantic/original seam annotation.
Pixels are model-input coordinates, not a common physical millimetre scale.
"""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


def outline(mask):
    contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError('empty mask')
    p = max(contours, key=cv2.contourArea)[:, 0, ::-1].astype(float)
    signed = np.sum(p[:, 1] * np.roll(p[:, 0], -1) - np.roll(p[:, 1], -1) * p[:, 0])
    if signed < 0:
        p = p[::-1]
    lengths = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    p = p[lengths > 1e-9]
    lengths = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    n = max(4, int(np.ceil(cumulative[-1])))
    positions = np.arange(n) * cumulative[-1] / n
    ix = np.minimum(np.searchsorted(cumulative, positions, side='right') - 1, len(p) - 1)
    points = p[ix] + ((positions - cumulative[ix]) / lengths[ix])[:, None] * (np.roll(p, -1, axis=0)[ix] - p[ix])
    smooth = gaussian_filter1d(points, 3., axis=0, mode='wrap')
    tangent = np.roll(smooth, -4, axis=0) - np.roll(smooth, 4, axis=0)
    normal = np.c_[-tangent[:, 1], tangent[:, 0]]
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    return dict(points=points, normal=normal, step=float(cumulative[-1] / n),
                perimeter=float(cumulative[-1]), components=len(contours), area=int(np.count_nonzero(mask)))


def runs(bits):
    bits = np.asarray(bits, bool)
    if not bits.any():
        return []
    if bits.all():
        return [np.arange(len(bits))]
    output = []
    for start in np.flatnonzero(bits & ~np.roll(bits, 1)):
        indices = []
        for j in range(len(bits)):
            i = (int(start) + j) % len(bits)
            if not bits[i]:
                break
            indices.append(i)
        output.append(np.asarray(indices, int))
    return output


def side(a, b, shift):
    distance, index = cKDTree(b['points'] + shift).query(a['points'], workers=1)
    delta = b['points'][index] + shift - a['points']
    opposed = (a['normal'] * b['normal'][index]).sum(1) <= -.5
    exterior = ((delta * a['normal']).sum(1) >= -3) & ((-delta * b['normal'][index]).sum(1) >= -3)
    return distance, opposed & exterior


def measure(mask_a, mask_b, translation, model_a, model_b):
    shapes = [outline(mask_a), outline(mask_b)]
    t = np.asarray(translation)
    distances = [side(shapes[0], shapes[1], -t), side(shapes[1], shapes[0], t)]
    record = dict(perimeter_mean_px=float(np.mean([s['perimeter'] for s in shapes])),
                  area_mean_px=float(np.mean([s['area'] for s in shapes])),
                  external_components=[s['components'] for s in shapes])
    lengths = {}
    for limit in (4, 10, 20, 40, 64):
        selected = [ok & (d <= limit) for d, ok in distances]
        lengths[limit] = float(np.mean([bits.sum() * s['step'] for bits, s in zip(selected, shapes)]))
        record['length_%d_px' % limit] = lengths[limit]
        record['segments_%d_ge8' % limit] = float(np.mean([
            sum(len(r) * s['step'] >= 8 for r in runs(bits)) for bits, s in zip(selected, shapes)]))
        record['longest_%d_px' % limit] = float(np.mean([
            max((len(r) * s['step'] for r in runs(bits)), default=0.) for bits, s in zip(selected, shapes)]))
        record['unfiltered_length_%d_px' % limit] = float(np.mean([
            (d <= limit).sum() * s['step'] for (d, _), s in zip(distances, shapes)]))
    record['length20_over40'] = lengths[20] / lengths[40] if lengths[40] else None
    record['length10_over40'] = lengths[10] / lengths[40] if lengths[40] else None
    record['length40_over64'] = lengths[40] / lengths[64] if lengths[64] else None
    for cutoff in (40, 64):
        values = np.concatenate([d[ok & (d <= cutoff)] for d, ok in distances])
        record['gap%d' % cutoff] = quant(values)
        record['gap%d_over10_share' % cutoff] = float(np.mean(values > 10)) if len(values) else None
        record['gap%d_over20_share' % cutoff] = float(np.mean(values > 20)) if len(values) else None
    points = [np.asarray(model_a), np.asarray(model_b)]
    nearest = [cKDTree(points[1] - t).query(points[0])[0], cKDTree(points[0] + t).query(points[1])[0]]
    record['tokens_mean'] = float(np.mean([len(p) for p in points]))
    record['model_spacing_mean_px'] = float(np.mean([
        np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1).mean() for p in points]))
    for limit in (10, 20, 40):
        record['potential_tokens_%d_mean' % limit] = float(np.mean([(d <= limit).sum() for d in nearest]))
    return record


def quant(values):
    a = np.asarray([v for v in values if v is not None], float)
    if not np.isfinite(a).all():
        raise ValueError('nonfinite observations')
    return dict(n=len(a), mean=float(a.mean()) if len(a) else None,
                **{'p%d' % p: float(np.percentile(a, p)) if len(a) else None for p in (0, 10, 25, 50, 75, 90, 100)})

"""Read-only geometric certificate for a residual largest-outer-loop seam.

This is not a data producer or target adapter. It proves source-edge support
only; it does not certify global boundary arc indices or projection labels.
No GT pose is accepted. Existing v3 normalization and budgets are untouched.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .rachel_preprocess import _pixel_cell_boundary_edges
from .rachel_source_density import source_boundary


class OuterCertificateRejected(ValueError):
    pass


CROSS = ndimage.generate_binary_structure(2, 1)
BACKGROUND = np.ones((3, 3), bool)  # boundary tracer: foreground4 / background8


def _key(edge):
    return tuple(sorted(tuple(int(x) for x in p) for p in edge))


def _incident(edge):
    (r, c), (rr, cc) = edge
    if r == rr and cc == c + 1:
        return (r - 1, c), (r, c)
    if c == cc and rr == r + 1:
        return (r, c - 1), (r, c)
    raise OuterCertificateRejected('not a unit source-cell edge')


def _at(mask, cell):
    r, c = cell
    return bool(0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and mask[r, c])


def _side_certificate(lower, upper, retained):
    if np.any(lower & ~upper):
        raise OuterCertificateRejected('source lower is not a subset of upper')
    labels, component_count = ndimage.label(lower, structure=CROSS)
    inner, outer = [], []
    for edge in sorted(retained):
        cells = _incident(edge)
        flags = [_at(lower, cell) for cell in cells]
        if sum(flags) != 1:
            raise OuterCertificateRejected('candidate is not a lower source boundary')
        inner.append(cells[flags.index(True)])
        outer.append(cells[flags.index(False)])
    core_ids = {int(labels[r, c]) for r, c in inner}
    if len(core_ids) != 1 or 0 in core_ids:
        raise OuterCertificateRejected('retained edge cells do not share one 4-connected lower core')
    core = labels == next(iter(core_ids))
    complement = ~np.pad(upper, 1, constant_values=False)
    seed = np.zeros_like(complement); seed[0, 0] = True
    exterior = ndimage.binary_propagation(seed, structure=BACKGROUND, mask=complement)
    exterior_ok = [bool(exterior[r + 1, c + 1]) for r, c in outer]
    if not all(exterior_ok):
        raise OuterCertificateRejected('retained outside cell lacks upper-complement exterior path')
    filled_core_area = int(ndimage.binary_fill_holes(core, structure=BACKGROUND).sum())
    competitors, n = ndimage.label(upper & ~core, structure=CROSS)
    areas = [int(ndimage.binary_fill_holes(competitors == k, structure=BACKGROUND).sum())
             for k in range(1, n + 1)]
    maximum = max(areas, default=0)
    if filled_core_area <= maximum:
        raise OuterCertificateRejected('largest outer loop could switch to a non-core upper component')
    return dict(lower_area=int(lower.sum()), upper_area=int(upper.sum()),
        uncertain_support_cells=int((upper & ~lower).sum()),
        lower_component_count=int(component_count), core_area=int(core.sum()),
        filled_core_area=filled_core_area, competitor_component_count=int(n),
        maximum_filled_competitor_area=maximum,
        filled_area_strict_margin=filled_core_area - maximum,
        retained_edges_checked=len(retained), all_inner_cells_same_lower_core=True,
        all_outer_cells_reach_padded_upper_complement_exterior=True,
        foreground_connectivity=4, background_connectivity=8,
        largest_core_outer_loop_stable=True)


def certify_residual_outer_interface(lower, upper, unknown_coordinates_rc, trim_radius_px=15.):
    """Prove residual edge-set invariance without assigning unknown owners.

For each side L <= M <= H. Every retained edge has its material cell in one
fixed 4-connected L core and its other cell in exterior(~H). That edge remains
on the core outer loop for every M. fill(core) exceeds every potential
competitor's fill area, so source_boundary's largest loop cannot switch.
Changes to raw edges are confined to cells incident to unknown support.
"""
    lower = {s: np.asarray(lower[s], bool) for s in 'ab'}
    upper = {s: np.asarray(upper[s], bool) for s in 'ab'}
    shapes = {m.shape for m in list(lower.values()) + list(upper.values())}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise OuterCertificateRejected('source shape mismatch')
    try:
        trim_radius_px = float(trim_radius_px)
    except (ValueError, TypeError):
        raise OuterCertificateRejected('trim must be finite and >=15px') from None
    if not np.isfinite(trim_radius_px) or trim_radius_px < 15:
        raise OuterCertificateRejected('trim must be finite and >=15px')
    coords = np.asarray(unknown_coordinates_rc)
    if (coords.ndim != 2 or coords.shape[1] != 2 or len(coords) == 0
            or coords.dtype.kind not in 'iuf' or not np.isfinite(coords).all()
            or not np.equal(coords, np.floor(coords)).all()):
        raise OuterCertificateRejected('unknown coordinates must be a nonempty Nx2 integer array')
    shape = next(iter(shapes))
    if np.any(coords < 0) or np.any(coords >= np.asarray(shape)):
        raise OuterCertificateRejected('unknown coordinates out of source bounds')
    coords = coords.astype(np.int64)
    if len(np.unique(coords, axis=0)) != len(coords):
        raise OuterCertificateRejected('unknown coordinates must be unique')
    if np.any(lower['a'] & lower['b']):
        raise OuterCertificateRejected('lower source A/B ownership must be mutually exclusive')
    unknown = np.zeros(shape, bool)
    unknown[coords[:, 0], coords[:, 1]] = True
    if any(np.any(lower[s] & unknown) for s in 'ab'):
        raise OuterCertificateRejected('declared unknown cells must be absent from both lower sources')
    for s in 'ab':
        if np.any((upper[s] & ~lower[s]) & ~unknown):
            raise OuterCertificateRejected('source uncertainty extends outside declared unknown cells')
    tree = cKDTree(coords + .5)
    def trimmed(keys):
        keys = sorted(keys)
        if not keys:
            return set()
        centers = np.asarray([(np.asarray(a) + np.asarray(b)) * .5 for a, b in keys])
        return {edge for edge, distance in zip(keys, tree.query(centers)[0]) if distance > trim_radius_px}
    raw = {s: {_key(edge) for edge in _pixel_cell_boundary_edges(lower[s])} for s in 'ab'}
    all_shared = raw['a'] & raw['b']
    retained = trimmed(all_shared)
    if not retained:
        raise OuterCertificateRejected('zero raw residual shared interface')
    known_outer = {s: source_boundary(lower[s]) for s in 'ab'}
    outer_retained = trimmed(set(known_outer['a'].keys) & set(known_outer['b'].keys))
    if outer_retained != retained:
        raise OuterCertificateRejected('all raw residual shared edges differ from selected outer shared edges')
    incident = set()
    for r, c in coords:
        corners = [(r,c),(r+1,c),(r+1,c+1),(r,c+1)]
        incident.update(_key((corners[i], corners[(i+1) % 4])) for i in range(4))
    if retained & incident:
        raise OuterCertificateRejected('retained edge is incident to unknown support')
    sides = {s: _side_certificate(lower[s], upper[s], retained) for s in 'ab'}
    return dict(schema_version='density-residual-outer-loop-certificate/1', status='proved',
        proof_method='lower_core_upper_exterior_strict_filled_area_bound',
        unknown_pixels=len(coords), trim_radius_px=float(trim_radius_px),
        raw_shared_edges=len(all_shared), retained_raw_shared_edges=len(retained),
        selected_outer_shared_trim_exactly_matches_all_raw_shared_trim=True,
        unknown_incident_edges=len(incident), retained_unknown_incident_intersection=0,
        both_actual_masks_independent_of_unknown=all(np.array_equal(lower[s], upper[s]) for s in 'ab'),
        sides=sides, global_assignment_enumeration_used=False, unknown_owner_assigned=False,
        GT_used=False, model_masks_modified=False, source_edge_support_only=True,
        projection_arc_indices_or_targets_certified=False)

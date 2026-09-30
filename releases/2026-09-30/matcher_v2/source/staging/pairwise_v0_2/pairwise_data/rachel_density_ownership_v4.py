"""V4 residual-source recovery with certified local geometric projection.

Successful v3 samples use v3 verbatim. Only explicit native-positive unknown
ownership refusals can use the new branch. No GT pose is used for matching.
The certificate proves residual selected-outer source edge support, NOT that
old full-loop arc indices or all hypothetical-owner targets are identical.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .rachel_density_ownership_v3 import (
    SourceDensityRecordResolverV3, CleanSourceDensityV3Dataset,
    resample_source_density_v3, _segmented_seams, SOURCE_TRIM_RADIUS, TOKEN_IGNORE_RADIUS)
from .rachel_density_ownership_v2 import MAX_PROJECTION_RADIUS
from .rachel_density_outer_certificate import (
    certify_residual_outer_interface, OuterCertificateRejected, CROSS, BACKGROUND,
    _incident, _at, _key)
from .rachel_source_density import (
    SCHEMA as DENSITY_SCHEMA, source_boundary, confirm_fixed_e1_mask, inherit_dense_source_targets)
from .rachel_preprocess import extract_ordered_outer_contour, _pixel_cell_boundary_edges
from .rachel_source_density_records import _mask
from .rachel_union_augmentation import UnionGeometryRejected
from .rachel_clean_density import SCHEMA as CLEAN_REPORT_SCHEMA
from .rachel_materialized_dataset import load_sample, save_sample
from .rachel_paired_density_dataset import file_sha256, validate_density_sample
from .rachel_training_dataset import _readonly

OWNERSHIP_PROTOCOL = 'density-local-unknown-ownership/4'
TRAIN_PIPELINE = 'paired-source-cell-density-materialization/4'
CLEAN_SCHEMA = 'rachel-clean-source-density-eval/4'
RECOVERABLE = frozenset({
    'density_unknown_owner_region_exceeds_bounded_protocol',
    'density_unknown_owner_proof_budget_exceeded',
    'density_unknown_owner_too_close_to_actual_interface',
    'density_unknown_owner_changes_actual_interface',
    'density_v3_unknown_owner_proof_budget_exceeded',
    'density_v3_retained_interface_not_owner_invariant'})
PROJECTION_PROTOCOL = 'certified-local-geometry-fixed8-hop/1'


def _digest(mask):
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=bool).tobytes()).hexdigest()


def certified_unknown_ownership(parents, ab):
    """Keep all >3px source cells unknown, prove the unaffected largest seam."""
    ids = sorted(parents)
    if len(ab) != 2 or ab[0] == ab[1] or any(k not in ids for k in ab):
        raise ValueError('two distinct existing source members required')
    stack = np.stack([np.asarray(parents[k], bool) for k in ids])
    overlap = stack.sum(0) > 1
    fraction = float(overlap.sum() / max(1, stack.any(0).sum()))
    if fraction > .02:
        raise UnionGeometryRejected('ownership_overlap_fraction_exceeds_2_percent')
    depths = np.stack([ndimage.distance_transform_edt(m) for m in stack])
    second = np.sort(depths, axis=0)[-2]
    unknown = overlap & (second > 3.)
    coords = np.argwhere(unknown)
    if not len(coords):
        raise UnionGeometryRejected('density_v4_requires_explicit_unknown_support')
    known = stack.copy(); safe = overlap & ~unknown
    known[:,safe] = np.arange(len(ids))[:,None] == depths[:,safe].argmax(0)[None,:]
    known[:,unknown] = False
    if (not np.array_equal(known.any(0), stack.any(0) & ~unknown)
            or not np.array_equal(known[:,~overlap], stack[:,~overlap])):
        raise ValueError('known ownership altered exclusive source support')
    lower = {s:known[ids.index(k)] for s,k in zip('ab',ab)}
    claims = {s:unknown & stack[ids.index(k)] for s,k in zip('ab',ab)}
    upper = {s:lower[s] | claims[s] for s in 'ab'}
    try:
        certificate = certify_residual_outer_interface(lower, upper, coords, SOURCE_TRIM_RADIUS)
    except OuterCertificateRejected as error:
        raise UnionGeometryRejected('density_v4_outer_certificate_refused:' + str(error)) from error
    boundaries = {s:source_boundary(lower[s]) for s in 'ab'}
    common = set(boundaries['a'].keys) & set(boundaries['b'].keys)
    tree = cKDTree(coords + .5)
    retained = {key for key in common if tree.query((np.asarray(key[0])+np.asarray(key[1]))*.5)[0] > SOURCE_TRIM_RADIUS}
    seams, components, topology = _segmented_seams(boundaries, retained)
    if not components:
        raise UnionGeometryRejected('density_v4_zero_residual_continuous_seam')
    proof = dict(schema_version=OWNERSHIP_PROTOCOL, actual_pair_members=list(ab),
        unknown_coordinates_parent_rc=coords.tolist(), unknown_pixels=len(coords),
        claimant_coordinates_by_side={s:np.argwhere(claims[s]).tolist() for s in 'ab'},
        lower_mask_sha256={s:_digest(lower[s]) for s in 'ab'},
        upper_mask_sha256={s:_digest(upper[s]) for s in 'ab'},
        source_shape=list(unknown.shape), maximum_group_overlap_fraction=.02,
        maximum_second_owner_depth_px=3., observed_overlap_fraction=fraction,
        observed_second_depth_max_px=float(second[unknown].max()),
        source_trim_radius_px=SOURCE_TRIM_RADIUS, symmetric_token_ignore_radius_px=TOKEN_IGNORE_RADIUS,
        outer_certificate=certificate,
        retained_source_edges=[[list(a),list(b)] for a,b in sorted(seams)],
        residual_components=components, topology=topology,
        projection_protocol=PROJECTION_PROTOCOL, unknown_owner_assigned=False,
        physical_mask_modified=False, GT_used=False,
        old_v3_enumeration_budget_modified=False,
        proof_scope='residual largest-outer source-edge support; new conservative geometric projection protocol, not old global arc-index invariance')
    metadata = dict(method=OWNERSHIP_PROTOCOL, applied=True,
        exclusive_pixels_unchanged=True, source_reference_has_explicit_unknowns=True,
        total_physical_support_unchanged=True, reference_unknown_support_pixels=len(coords))
    return {k:known[i] for i,k in enumerate(ids)}, metadata, proof


def _projection_membership(lower, upper, boundary, seam_keys):
    """Certify each full-known-outer edge locally, not just seam edges."""
    labels, _ = ndimage.label(lower, structure=CROSS)
    example = min(seam_keys); cells = _incident(example)
    inside = next(cell for cell in cells if _at(lower, cell))
    core = labels == labels[inside]
    complement = ~np.pad(upper, 1, constant_values=False)
    seed = np.zeros_like(complement); seed[0,0] = True
    exterior = ndimage.binary_propagation(seed, structure=BACKGROUND, mask=complement)
    stable = []
    for edge in boundary.keys:
        cells = _incident(edge); flags = [_at(lower, cell) for cell in cells]
        if sum(flags) != 1:
            stable.append(False); continue
        inner, outer = cells[flags.index(True)], cells[flags.index(False)]
        stable.append(_at(core, inner) and exterior[outer[0]+1,outer[1]+1])
    return np.asarray(stable, bool)


def _project_certified(points, offset, boundary, seams, lower, upper, unknown_coords, *, radius):
    """Use full known outer geometry; accept only an invariant local window.

    All nearest candidates within radius+.5 are contained by midpoint radius+1.
    Raw non-outer competitors and uncertified outer members cause ignore. Local
    geometric ties must lie within 8 consecutive certified full-boundary edges;
    this fixed rule never scales with a hypothetical full-loop length. We make
    no claim that it reproduces v3's perimeter-dependent ambiguity rule.
    """
    if radius > MAX_PROJECTION_RADIUS:
        raise ValueError('v4 certificate does not cover this damage depth')
    pp = np.asarray(points, float) + .5 - np.asarray(offset, float)
    starts, ends = boundary.starts, boundary.ends; vectors = ends - starts
    centers = (starts + ends) * .5; tree = cKDTree(centers)
    stable = _projection_membership(lower, upper, boundary, set(seams))
    raw = {_key(e) for e in _pixel_cell_boundary_edges(lower)}
    competitor = sorted(raw - set(boundary.keys))
    competitor_tree = cKDTree(np.asarray([(np.asarray(a)+np.asarray(b))*.5 for a,b in competitor])) if competitor else None
    unknown_tree = cKDTree(np.asarray(unknown_coords, float)+.5)
    n = len(points); ids = np.full(n, -1, int); distances = np.full(n, np.inf)
    trusted = np.zeros(n, bool); nonseam = np.zeros(n, bool)
    component = np.full(n, -1, np.int64); seam_s = np.full(n, np.nan)
    reasons = {k:np.zeros(n, bool) for k in ('unknown_neighborhood','source_trim','raw_nonouter_competitor',
        'unstable_outer_member','disconnected_geometric_tie','outside_projection_radius','seam_endpoint_guard')}
    window = radius + 1.
    # Follow the genuine full closed known loop, never a trimmed arc as a loop.
    def connected_local(first, second, steps):
        if first == second:return bool(stable[first])
        for sign in (-1,1):
            path = [(first + sign*j) % len(starts) for j in range(steps+1)]
            if second in path and all(stable[x] for x in path[:path.index(second)+1]):return True
        return False
    for i, point in enumerate(pp):
        if unknown_tree.query(point)[0] <= TOKEN_IGNORE_RADIUS:
            reasons['unknown_neighborhood'][i] = True; continue
        neighbors = tree.query_ball_point(point, window)
        if not neighbors:
            reasons['outside_projection_radius'][i] = True; continue
        if competitor_tree is not None and competitor_tree.query(point)[0] <= window:
            reasons['raw_nonouter_competitor'][i] = True; continue
        if not stable[neighbors].all():
            reasons['unstable_outer_member'][i] = True; continue
        neighbors = np.asarray(sorted(neighbors, key=lambda j:boundary.keys[j]), int)
        alpha = np.clip(np.sum((point - starts[neighbors])*vectors[neighbors], axis=1), 0, 1)
        projected = starts[neighbors] + alpha[:,None]*vectors[neighbors]
        ds = np.linalg.norm(point - projected, axis=1)
        winner = int(ds.argmin()); edge_id = int(neighbors[winner]); best = float(ds[winner])
        ids[i] = edge_id; distances[i] = best
        if best > radius:
            reasons['outside_projection_radius'][i] = True; continue
        if unknown_tree.query(centers[edge_id])[0] <= SOURCE_TRIM_RADIUS:
            reasons['source_trim'][i] = True; continue
        ties = neighbors[ds <= best + .5]
        if not all(connected_local(edge_id, int(j), 8) for j in ties):
            reasons['disconnected_geometric_tie'][i] = True; continue
        trusted[i] = True
        source = seams.get(boundary.keys[edge_id])
        if source is not None:
            cid, position, begin, end = source
            component[i] = cid
            seam_s[i] = position + np.clip(np.dot(projected[winner]-begin, end-begin), 0, 1)
        else:
            guard = [(edge_id + j) % len(starts) for j in range(-2,3)]
            endpoint = not all(stable[j] for j in guard) or any(boundary.keys[j] in seams for j in guard)
            reasons['seam_endpoint_guard'][i] = endpoint
            nonseam[i] = not endpoint
    step = float(np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1).sum()/n)
    return dict(component=component, seam_s=seam_s, trusted=trusted, nonseam=nonseam,
        source_edge_index=ids, distance_px=distances, mean_step_px=step,
        projection_radius_px=radius, ambiguous_count=int(reasons['disconnected_geometric_tie'].sum()),
        projection_protocol=PROJECTION_PROTOCOL, reason_masks=reasons,
        full_known_outer_edges=len(starts), stable_full_outer_edges=int(stable.sum()),
        raw_nonouter_competitor_edges=len(competitor), midpoint_window_radius_px=window)


def resample_source_density_v4(sample, report, source, offsets, clean, *, cap,
                               ownership_allowance_px=0., origin_receipts=None):
    proof = report.get('source_ownership_v4')
    if proof is None:
        return resample_source_density_v3(sample, report, source, offsets, clean, cap=cap,
            ownership_allowance_px=ownership_allowance_px, origin_receipts=origin_receipts)
    if (not sample.label or proof.get('schema_version') != OWNERSHIP_PROTOCOL
            or any(report.get('source_ownership_v'+v) is not None for v in ('2','3'))):
        raise ValueError('v4 requires an exclusive verified native-positive recovery proof')
    if cap not in (512,1024) or report.get('schema_version') not in ('rachel-weathered-source-arc-e1/v1',CLEAN_REPORT_SCHEMA):
        raise ValueError('unsupported cap or missing augmentation ancestry')
    upper = {s:np.asarray(source[s], bool).copy() for s in 'ab'}
    coords = np.asarray(proof['unknown_coordinates_parent_rc'], int)
    for s in 'ab':
        if _digest(source[s]) != proof['lower_mask_sha256'][s]:raise ValueError('lower source identity differs')
        for r,c in proof['claimant_coordinates_by_side'][s]:upper[s][r,c] = True
        if _digest(upper[s]) != proof['upper_mask_sha256'][s]:raise ValueError('upper source identity differs')
    certificate = certify_residual_outer_interface(source, upper, coords, SOURCE_TRIM_RADIUS)
    if certificate != proof['outer_certificate']:raise ValueError('outer proof differs from current source')
    replay = {s:confirm_fixed_e1_mask(clean[s], getattr(sample,'mask_'+s), report['side_'+s],
        (origin_receipts or {}).get(s)) for s in 'ab'}
    boundaries = {s:source_boundary(source[s]) for s in 'ab'}
    retained = {tuple(tuple(int(v) for v in p) for p in edge) for edge in proof['retained_source_edges']}
    seams, components, topology = _segmented_seams(boundaries, retained)
    if set(seams) != retained or not components:raise ValueError('residual source edge identity differs')
    raw_shared = set(boundaries['a'].keys) & set(boundaries['b'].keys)
    updates, views, suppression = {}, {}, {}
    for s in 'ab':
        points, valid = extract_ordered_outer_contour(np.asarray(getattr(sample,'mask_'+s)).squeeze().astype(bool),
            cap=cap, smoothing_sigma=3.)
        if len(np.unique(points, axis=0)) != len(points):raise ValueError('duplicate density contour points')
        view = _project_certified(points, offsets[s], boundaries[s], seams, source[s], upper[s], coords,
            radius=6. + replay[s]['max_depth_px'] + ownership_allowance_px)
        # A trimmed/rejected original shared edge can never become dustbin.
        dropped = np.asarray([j >= 0 and boundaries[s].keys[j] in raw_shared - set(seams)
                              for j in view['source_edge_index']], bool)
        view['component'][dropped] = -1;view['seam_s'][dropped] = np.nan
        view['trusted'][dropped] = False;view['nonseam'][dropped] = False
        view['reason_masks']['dropped_shared_edge'] = dropped
        suppression[s] = {k:int(v.sum()) for k,v in view['reason_masks'].items()}
        suppression[s].update(symmetric_unknown_radius_px=TOKEN_IGNORE_RADIUS,
            source_trim_radius_px=SOURCE_TRIM_RADIUS, full_known_outer_edges=view['full_known_outer_edges'],
            stable_full_outer_edges=view['stable_full_outer_edges'],
            raw_nonouter_competitor_edges=view['raw_nonouter_competitor_edges'],
            midpoint_window_radius_px=view['midpoint_window_radius_px'])
        views[s] = view;updates['points_rc_'+s] = points;updates['contour_valid_'+s] = valid
    ta, tb, chosen = inherit_dense_source_targets(views['a'], views['b'], components)
    if not chosen:raise UnionGeometryRejected('density_v4_zero_residual_positive_correspondence')
    updates.update(target_a=_readonly(ta,np.int64), target_b=_readonly(tb,np.int64))
    result = replace(sample, **updates)
    detail = dict(schema_version=DENSITY_SCHEMA, cap=cap, physical_masks_unchanged=True,
        gt_translation_used_for_matching=False, weathered_cross_fragment_nearest_neighbor=False,
        target_source='certified residual source-cell segments with conservative own-fragment geometric projection',
        topology=topology, components=components, replay=replay, new_match_count=len(chosen),
        old_match_count=int(np.count_nonzero(sample.target_a>=0)), chosen_source_cells=[list(x) for x in chosen],
        source_ownership_v4=proof, projection_protocol=PROJECTION_PROTOCOL,
        uncertain_token_exclusion=suppression,
        sides={s:dict(points=len(views[s]['component']), trusted=int(views[s]['trusted'].sum()),
            ignored=int(np.count_nonzero(getattr(result,'target_'+s)==-2)),
            source_seam_tokens=int(np.count_nonzero(views[s]['component']>=0)),
            mean_step_px=views[s]['mean_step_px'], projection_radius_px=views[s]['projection_radius_px'],
            ambiguous_count=views[s]['ambiguous_count']) for s in 'ab'})
    fields = ('effective_supervised_match_count','inherited_match_count','ignored_token_count','inheritance_rule')
    derived = dict(report, density=detail, original_e1_target_summary={k:report[k] for k in fields},
        effective_supervised_match_count=len(chosen), inherited_match_count=len(chosen),
        ignored_token_count=int(np.count_nonzero(ta==-2)+np.count_nonzero(tb==-2)),
        inheritance_rule='certified residual source-cell IDs; own-fragment fixed-local-geometry projection; uncertain positive/negative tokens ignored')
    return result, derived, views


def _recover_native_v4(resolver, row, report, root):
    parts = row['fragment_a']['fragment_token'].split('/');group,path = resolver._group(parts[1],parts[2])
    fragments = {s:next(f for f in group['fragments'] if f['fragment_token']==row['fragment_'+s]['fragment_token']) for s in 'ab'}
    parents = resolver._parents(group);ab = [fragments[s]['fragment_id'] for s in 'ab']
    masks, metadata, proof = certified_unknown_ownership(parents, ab)
    offsets = {s:fragments[s]['target_audit']['parent_to_model_offset_rc'] for s in 'ab'}
    clean = {s:_mask(Path(root)/row['fragment_'+s]['model_mask_path']) for s in 'ab'}
    for s in 'ab':resolver._verify_centerpad(parents[fragments[s]['fragment_id']],clean[s],offsets[s])
    return dict(report,source_ownership_v4=proof), {s:masks[fragments[s]['fragment_id']] for s in 'ab'}, offsets,clean,metadata,str(path)


class SourceDensityRecordResolverV4(SourceDensityRecordResolverV3):
    def resolve(self,index):
        try:return super().resolve(index)
        except UnionGeometryRejected as error:
            entry=self.entries[index]
            if error.reason not in RECOVERABLE or entry['source_stratum']!='native_positive' or not entry['label']:raise
        entry=self.entries[index];sample,report=load_sample(self.artifact_root/entry['artifact_path'])
        report,source,offsets,clean,metadata,path=_recover_native_v4(self,entry['source_row'],report,entry['source_root'])
        provenance=dict(index=index,pair_id=sample.pair_id,label=True,source_stratum=entry['source_stratum'],
            fixed_artifact=str(self.artifact_root/entry['artifact_path']),source_root=entry['source_root'],
            groups=[path,path],ownership_reference_only=True,ownership_normalization=metadata,
            source_ownership_v4=report['source_ownership_v4'],parent_to_model_offsets=offsets)
        return sample,report,source,offsets,clean,3.,provenance

    def sample_at(self,index,cap):
        sample,report,source,offsets,clean,allowance,provenance=self.resolve(index)
        result,derived,views=resample_source_density_v4(sample,report,source,offsets,clean,cap=cap,ownership_allowance_px=allowance)
        derived['source_resolution']=provenance
        return result,derived,views


class CleanSourceDensityV4Dataset(CleanSourceDensityV3Dataset):
    def __init__(self,root,split,contour_cap=512,cache_dir=None):
        super().__init__(root,split,contour_cap,cache_dir=None)
        from . import rachel_density_outer_certificate
        protocol={k:v for k,v in self.protocol.items() if k!='identity_sha256'}
        protocol.update(schema_version=CLEAN_SCHEMA,source_density_version='v4',ownership_protocol=OWNERSHIP_PROTOCOL,
            density_v4_module_sha256=file_sha256(__file__),outer_certificate_sha256=file_sha256(rachel_density_outer_certificate.__file__),
            recovery_projection_protocol=PROJECTION_PROTOCOL)
        self.identity=hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.protocol=dict(protocol,identity_sha256=self.identity);self.cache_dir=Path(cache_dir).resolve() if cache_dir else None
        if self.cache_dir:
            if self.cache_dir==self.root or self.root in self.cache_dir.parents:raise ValueError('v4 cache must not modify original release')
            self.cache_dir.mkdir(parents=True,exist_ok=True);path=self.cache_dir/'cache_identity.json'
            if path.exists():
                if json.loads(path.read_text())!=self.protocol:raise ValueError('v4 cache split/cap/source identity differs')
            else:
                temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(self.protocol,indent=2)+'\n');os.replace(temp,path)

    def _resolve_clean(self,index):
        try:return super()._resolve_clean(index)
        except UnionGeometryRejected as error:
            row=self.rows[index]
            if error.reason not in RECOVERABLE or not row['label']:raise
        sample=self.base[index];count=int(np.count_nonzero(sample.target_a>=0))
        report=dict(schema_version=CLEAN_REPORT_SCHEMA,pair_id=sample.pair_id,source_split=self.split,
            changed_pair=False,changed_a=False,changed_b=False,weathering_applied=False,
            side_a={'effective_applied':False},side_b={'effective_applied':False},pose_supervision_enabled=True,
            original_gt_translation_preserved=True,effective_supervised_match_count=count,inherited_match_count=count,
            ignored_token_count=int(np.count_nonzero(sample.target_a==-2)+np.count_nonzero(sample.target_b==-2)),
            inheritance_rule='original clean512 targets used only as an audit count; never reindexed')
        report,source,offsets,clean,metadata,path=_recover_native_v4(self,row,report,self.root)
        report['source_resolution']=dict(groups=[path,path],parent_to_model_offsets=offsets,ownership_reference_only=True,
            ownership_normalization=metadata,label_origin=row['label_origin'],negative_origin=row.get('negative_origin'),
            source_ownership_v4=report['source_ownership_v4'])
        return sample,report,source,offsets,clean,3.

    def weathered(self,index):
        row=self.rows[index];path=self.cache_dir/(hashlib.sha256(row['pair_id'].encode()).hexdigest()+'.npz') if self.cache_dir else None
        if path and path.exists():
            sample,report=load_sample(path)
            if sample.pair_id!=row['pair_id'] or bool(sample.label)!=bool(row['label']):raise ValueError('v4 cached labels differ')
            validate_density_sample(sample,report,self.contour_cap,self.identity);return sample,report
        original,report,source,offsets,clean,allowance=self._resolve_clean(index)
        sample,derived,_=resample_source_density_v4(original,report,source,offsets,clean,cap=self.contour_cap,ownership_allowance_px=allowance)
        derived['paired_density']=dict(identity_sha256=self.identity,source_pair_id=sample.pair_id,
            source_manifest=str(self.manifest_path),source_split=self.split)
        for field in ('mask_a','mask_b','coarse_mask_a','coarse_mask_b','label','translation_a_to_b_rc','translation_a_to_b_xy_cartesian','translation_valid'):
            if not np.array_equal(getattr(original,field),getattr(sample,field),equal_nan=True):raise ValueError('v4 original field changed: '+field)
        validate_density_sample(sample,derived,self.contour_cap,self.identity)
        if path:save_sample(path,sample,derived)
        return sample,derived

"""Conservative, source-anchored correspondence overlays. No image generation.

The policy is a new TRAIN-label candidate, not permission to train. Existing
datasets, source code, contours, masks, poses, negatives and losses are immutable.
"""
from dataclasses import asdict, dataclass
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import extract_ordered_outer_contour

class EvidenceError(ValueError):
    pass

@dataclass(frozen=True)
class Policy:
    version: str = 'source-anchored-correspondence-review/2'
    source_contact_px: float = 3.
    source_run_min_px: float = 8.
    final_main_px: float = 5.
    final_max_px: float = 8.
    certain_nonmatch_px: float = 15.
    projection_residual_px: float = 3.
    projection_nonlocal_arc_px: float = 8.
    projection_tie_margin_px: float = .5
    smooth_gap_delta_px: float = 1.5
    endpoint_tokens: int = 4
    cut_guard_px: float = 3.
    destructive_guard_px: float = 8.
    strict_shallow_single_side_peak_px: float = 5.
    ray_max_px: float = 64.
    ray_step_px: float = .5

POLICY=Policy()

def need(condition, message):
    if not condition: raise EvidenceError(message)

def unpack(z, key, shape=None):
    a=np.unpackbits(z[key],axis=-1).astype(bool)
    if shape is not None: a=a[..., :shape[-1]].reshape(shape)
    return a

def sample_mask(z, side):
    return unpack(z, 'mask_'+side+'_packed', tuple(z['mask_'+side+'_shape']))[0]

def dense(mask):
    p,_=extract_ordered_outer_contour(mask,cap=int(mask.size),smoothing_sigma=0.)
    p=np.asarray(p,float); e=np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1)
    return p, np.r_[0.,np.cumsum(e[:-1])], e

def runs(bits):
    bits=np.asarray(bits,bool); n=len(bits)
    if not n or not bits.any(): return []
    if bits.all(): return [np.arange(n)]
    result=[]
    for first in np.flatnonzero(bits&~np.roll(bits,1)):
        ids=[]; k=int(first)
        while bits[k%n] and len(ids)<n:
            ids.append(k%n); k+=1
        result.append(np.asarray(ids,int))
    return result

def arc_expand(bits, arc, perimeter, radius):
    if not bits.any(): return bits.copy()
    locations=arc[bits]
    d=cKDTree(np.r_[locations-perimeter,locations,locations+perimeter][:,None]).query(arc[:,None])[0]
    return d<=radius+1e-8

def inside(mask, points, order=0):
    return ndimage.map_coordinates(np.asarray(mask,float),np.asarray(points).T,
                                  order=order,mode='constant',cval=0.)

def parent_to_canvas(mask, shift, shape):
    shift=np.asarray(shift)
    need(np.allclose(shift,np.rint(shift)), 'non-integral strict raster translation')
    shift=shift.astype(int); out=np.zeros(shape,bool)
    lo=np.maximum(shift,0); hi=np.minimum(np.asarray(mask.shape)+shift,shape)
    if np.all(hi>lo):
        out[lo[0]:hi[0],lo[1]:hi[1]]=mask[lo[0]-shift[0]:hi[0]-shift[0],lo[1]-shift[1]:hi[1]-shift[1]]
    return out

def frames(z, proof, record, preweather, latent):
    """Return source/final masks, final tokens and GT in one verified convention."""
    points={s:z['points_rc_'+s].astype(float).copy() for s in 'ab'}
    t=z['translation_a_to_b_rc'].astype(float).copy()
    final={s:sample_mask(z,s) for s in 'ab'}
    strict=record['recipe'].startswith('straight_')
    if strict:
        need('accepted_damage_trace' in record and 'target_audit' in record,'missing strict interval provenance')
        original={s:unpack(proof,'cut_'+s,tuple(proof['shape'])) for s in 'ab'}
        parents={s:unpack(proof,'final_parent_'+s,tuple(proof['shape'])) for s in 'ab'}
        for s in 'ab':
            # Frozen v4.2 finalize retains the largest 4-connected component
            # AFTER the stored final_parent stage and BEFORE integer centering.
            # Reproduce this read-only coordinate adapter, not a new generator.
            labels,n=ndimage.label(parents[s])
            need(n>0,'empty strict parent mask')
            sizes=np.bincount(labels.ravel())[1:]
            need(sizes.max()/parents[s].sum()>=.97,'strict finalization component gate differs')
            parents[s]=labels==(int(sizes.argmax())+1)
            shift=proof['shift_'+s]
            need(np.array_equal(final[s],parent_to_canvas(parents[s],shift,final[s].shape)), 'strict final proof differs from sample')
            points[s]-=shift
        need(np.allclose(t,proof['shift_b']-proof['shift_a'],atol=1e-5),'strict pose/parent frame mismatch')
        final=parents; t=np.zeros(2)
    else:
        need(preweather is not None and latent is not None, 'missing original or latent sidecar')
        original={s:unpack(proof,'packed_fragment_'+s) for s in 'ab'}
        need(np.allclose(t,preweather['translation_a_to_b_rc'],atol=1e-5),'preweather pose mismatch')
        for s in 'ab':
            need(np.array_equal(final[s],unpack(proof,'packed_final_'+s)),'final proof differs from sample')
            need(np.array_equal(original[s],unpack(preweather,'packed_preweather_'+s)),'original stage differs from preweather')
            need(not np.any(final[s]&~original[s]),'non-shrink curriculum mask')
            # Validate what latent actually records, not assume it covers the seam.
            shift=t if s=='a' else -t
            required=[s+'_'+k for k in ('source_points','partner_source_points','projected_points',
                     'partner_projected_points','gap','source_gap','valid','recession','partner_recession')]
            need(all(k in latent for k in required),'incomplete latent evidence')
            v=latent[s+'_valid']; src=latent[s+'_source_points']; src2=latent[s+'_partner_source_points']
            p=latent[s+'_projected_points']; q=latent[s+'_partner_projected_points']
            need(np.allclose(np.linalg.norm(p+shift-q,axis=1),latent[s+'_gap'],atol=2e-4),'latent gap/frame mismatch')
            need(np.allclose(np.linalg.norm(src+shift-src2,axis=1),latent[s+'_source_gap'],atol=2e-4),'latent source/frame mismatch')
            need(np.allclose(np.linalg.norm(p[v]-src[v],axis=1),latent[s+'_recession'][v],atol=2e-4),'latent recession mismatch')
            # First-hit rays are at/on foreground pixels. Invalid rays remain unknown.
            hits=inside(final[s],p[v])>.5
            # The archived ray coordinates are float32. A half-pixel can round
            # onto the other pixel on reload; permit only its quantified
            # floating-point envelope, never a whole-pixel geometry tolerance.
            for dr,dc in ((2e-4,0),(-2e-4,0),(0,2e-4),(0,-2e-4)):
                hits|=inside(final[s],p[v].astype(float)+[dr,dc])>.5
            need(np.all(hits),'latent projected pixels not retained')
    return original, final, points, t, strict

def ray_map(old, new, source, tokens, arc, edges, policy):
    smooth=ndimage.gaussian_filter(np.asarray(old,float),2.)
    gradient=np.stack(np.gradient(smooth),axis=-1)
    ij=np.clip(np.rint(source).astype(int),0,np.array(old.shape)-1)
    normals=gradient[ij[:,0],ij[:,1]]; norm=np.linalg.norm(normals,axis=1)
    normals/=np.maximum(norm[:,None],1e-9)
    depths=np.arange(0.,policy.ray_max_px+1e-8,policy.ray_step_px)
    rays=source[:,None]+depths[None,:,None]*normals[:,None]
    hit=ndimage.map_coordinates(np.asarray(new,float),[rays[:,:,0],rays[:,:,1]],order=0,mode='constant',cval=0)>.5
    valid=hit.any(1)&(norm>1e-6)
    first=hit.argmax(1); projected=rays[np.arange(len(source)),first]
    source_ids=np.flatnonzero(valid)
    need(len(source_ids)>=4,'no reliable original inward rays')
    distances, idx=cKDTree(projected[source_ids]).query(tokens,k=min(16,len(source_ids)))
    distances=np.asarray(distances).reshape(len(tokens),-1); idx=source_ids[np.asarray(idx).reshape(len(tokens),-1)]
    ancestor=idx[:,0]; chosen_arc=arc[ancestor]; perimeter=float(edges.sum())
    delta=np.abs(arc[idx]-chosen_arc[:,None]); delta=np.minimum(delta,perimeter-delta)
    ambiguous=np.any((delta>policy.projection_nonlocal_arc_px)&(distances<=distances[:,0,None]+policy.projection_tie_margin_px),axis=1)
    trusted=(distances[:,0]<=policy.projection_residual_px)&~ambiguous
    return dict(ancestor=ancestor, trusted=trusted, residual=distances[:,0],
                source=source[ancestor], ray_depth=depths[first[ancestor]],
                source_arc=chosen_arc, normals=normals[ancestor], projected=projected[ancestor])

def damage_on_source(source,arc,edges,original,proof,record,preweather,side,strict,policy):
    destructive=np.zeros(len(source),bool); overlap=np.zeros(len(source),bool)
    structural=np.zeros(len(source),bool); recipe=record['recipe']
    if strict:
        along=(source-proof['cut_centre'])@proof['cut_axis']
        trace=record['accepted_damage_trace']; intervals=trace['intervals']
        shallow=[]
        for x in intervals:
            lo,hi=float(x['start']),float(x['stop'])
            need(np.isfinite([lo,hi,x['depth']]).all() and hi>=lo, 'invalid strict interval')
            band=(along>=lo-policy.cut_guard_px)&(along<=hi+policy.cut_guard_px)
            if x['kind']=='overlap': overlap|=band; destructive|=band
            elif x['kind']=='gap':
                if x['depth']<=policy.strict_shallow_single_side_peak_px:
                    shallow.append(x)
                    # Abrupt interval shoulders are not the smooth interior.
                    destructive|=(np.abs(along-lo)<=policy.cut_guard_px)|(np.abs(along-hi)<=policy.cut_guard_px)
                else: destructive|=band
            else: raise EvidenceError('unknown strict interval kind')
        for x in shallow:
            for y in shallow:
                if x['side']==y['side']: continue
                lo=max(x['start'],y['start']); hi=min(x['stop'],y['stop'])
                if hi>=lo: destructive|=(along>=lo-policy.cut_guard_px)&(along<=hi+policy.cut_guard_px)
        for lo,hi in record['target_audit'].get('nick_intervals',[]):
            destructive|=(along>=lo-policy.destructive_guard_px)&(along<=hi+policy.destructive_guard_px)
    else:
        trimmed=unpack(proof,'packed_trim_'+side)
        structural|=inside(trimmed,source)<.5
        if recipe=='partial': structural|=inside(unpack(proof,'packed_primary_'+side),source)<.5
        structural=arc_expand(structural,arc,edges.sum(),policy.cut_guard_px)
        if recipe.startswith(('local','gaps')):
            prefix='primary_'+side+'_'
            field=proof
            if prefix+'points' not in field and record.get('v14_fallback'):
                prefix=side+'_'; field=preweather
            if prefix+'points' in field and prefix+'major' in field:
                p=field[prefix+'points']; major=field[prefix+'major']
                dist,ids=cKDTree(p).query(source)
                destructive|=(dist<=3.5)&(major[ids]>0)
            elif np.array_equal(unpack(proof,'packed_primary_'+side),trimmed):
                # This side received no primary damage. Field absence here is
                # intentional and demonstrated by the two immutable rasters.
                pass
            else:
                # Explicit source regions are an independently recorded fallback.
                details=record.get('detail',{}).get('primary_damage',{}).get(side,{})
                regions=details.get('ignore_source_regions',[])
                if not regions:
                    # Do not silently accept absence of a destructive field.
                    destructive[:]=True
                for r in regions:
                    p=np.asarray(r.get('source_points_rc',[]),float).reshape(-1,2)
                    if len(p): destructive|=cKDTree(p).query(source)[0]<=1.5
            destructive=arc_expand(destructive,arc,edges.sum(),policy.destructive_guard_px)
    return destructive|structural, overlap

def monotone_subset(candidates, side, dense_a, arc_a, run_id, run_pos):
    """Conservative subsequence in the SAME original A-seam arc; never rematch."""
    chosen=[]
    if not candidates:return chosen
    for rid in sorted({int(side['a']['component'][i]) for i,j in candidates}):
        group=[(i,j) for i,j in candidates if side['a']['component'][i]==rid]
        group.sort(key=lambda x:(side['a']['along'][x[0]],x[0],x[1]))
        best_end=[1]*len(group); prev=[-1]*len(group)
        vals=[side['b']['along_on_a'][j] for i,j in group]
        for k in range(len(group)):
            for q in range(k):
                if vals[q]+1e-6<vals[k] and side['a']['along'][group[q][0]]+1e-6<side['a']['along'][group[k][0]]:
                    if best_end[q]+1>best_end[k]:best_end[k]=best_end[q]+1;prev[k]=q
        if not group:continue
        at=max(range(len(group)),key=lambda k:(best_end[k],-k)); part=[]
        while at>=0: part.append(group[at]);at=prev[at]
        chosen.extend(reversed(part))
    return chosen

def state_counts(t,v):
    return dict(match=int(((t>=0)&v).sum()),ignore=int(((t==-2)&v).sum()),
                unmatched=int(((t==-1)&v).sum()),padding=int((~v).sum()))

def winding(p):
    return np.sign(np.sum(p[:,1]*np.roll(-p[:,0],-1)-np.roll(p[:,1],-1)*(-p[:,0])))

def added_overlap_on_tokens(original,final,points,shift,side,other):
    """Detect new material intersection, not a smoothed token inside other mask.

    Raw crops have fractional GT placement and can share a narrow raster band
    before any augmentation. Its existence is not evidence of added overlap.
    The explicit strict overlap intervals remain protected independently.
    """
    aligned_final=ndimage.shift(final[other].astype(np.uint8),-np.asarray(shift),order=0,mode='constant')>0
    aligned_original=ndimage.shift(original[other].astype(np.uint8),-np.asarray(shift),order=0,mode='constant')>0
    need(aligned_final.shape==final[side].shape,'overlap frame shape differs')
    added=(final[side]&aligned_final)&~(original[side]&aligned_original)
    if not added.any():return np.zeros(len(points),bool)
    distance=ndimage.distance_transform_edt(~added)
    return inside(distance,points,order=1)<=1.5

def run(z, proof, record, preweather=None, latent=None, policy=POLICY):
    old={s:np.array(z['target_'+s],copy=True) for s in 'ab'}
    valid={s:z['contour_valid_'+s].astype(bool) for s in 'ab'}
    if not bool(z['label']):
        return dict(full_a=old['a'],full_b=old['b'],tight_a=old['a'].copy(),tight_b=old['b'].copy()),dict(status='negative_unchanged',pairs=[])
    original, final, points, t, strict=frames(z,proof,record,preweather,latent)
    outlines={s:dense(original[s]) for s in 'ab'}
    final_outlines={s:dense(final[s])[0] for s in 'ab'}
    side={}
    for s,o,shift in [('a','b',t),('b','a',-t)]:
        src,arc,edges=outlines[s]
        dist,partner=cKDTree(outlines[o][0]).query(src+shift)
        shared=dist<=policy.source_contact_px
        ids=np.full(len(src),-1,int); pos=np.zeros(len(src))
        for rid,rr in enumerate(runs(shared)):
            if edges[rr].sum()<policy.source_run_min_px:shared[rr]=False;continue
            ids[rr]=rid; pos[rr]=np.r_[0.,np.cumsum(edges[rr][:-1])]
        mapping=ray_map(original[s],final[s],src,points[s],arc,edges,policy)
        destructive,overlap=damage_on_source(src,arc,edges,original[s],proof,record,preweather,s,strict,policy)
        anc=mapping['ancestor']; mapping.update(component=ids[anc],along=pos[anc],shared=shared[anc],
            destructive=destructive[anc],overlap=overlap[anc], original_destructive=destructive,
            original_overlap=overlap, original_run_ids=ids, original_run_pos=pos,
            original_shared=shared)
        # Only added material overlap is new damage. A smoothed contour token
        # inside the opposite unchanged mask does not prove such damage.
        overlap_actual=added_overlap_on_tokens(original,final,points[s],shift,s,o)
        overlap_all=mapping['overlap']|overlap_actual
        for k in range(-2,3): overlap_all|=np.roll(mapping['overlap']|overlap_actual,k)
        mapping['overlap_guard']=overlap_all
        mapping['destructive']|=overlap_all
        mapping['near_distance']=cKDTree(final_outlines[o]).query(points[s]+shift)[0]
        mapping['intact']=mapping['ray_depth']<=policy.ray_step_px
        endpoint=np.zeros(len(points[s]),bool)
        for rid in np.unique(ids[ids>=0]):
            mapped=np.flatnonzero(valid[s]&(ids[anc]==rid))
            mapped=mapped[np.argsort(pos[anc[mapped]],kind='stable')]
            endpoint[mapped[:policy.endpoint_tokens]]=True
            endpoint[mapped[-policy.endpoint_tokens:]]=True
        mapping['endpoint']=endpoint; mapping['eligible']=valid[s]&mapping['trusted']&mapping['shared']&~mapping['destructive']
        side[s]=mapping
    # Destruction invalidates both sides of the same original seam, including
    # the denser partner token just inside a protected shoulder.
    for s,o,shift in [('a','b',t),('b','a',-t)]:
        distance,idx=cKDTree(outlines[o][0]).query(side[s]['source']+shift)
        side[s]['destructive']|=(distance<=policy.source_contact_px)&side[o]['original_destructive'][idx]
        side[s]['eligible']&=~side[s]['destructive']
    # B descendants expressed on the same original A arc, with component identity.
    d,bona=cKDTree(outlines['a'][0]).query(side['b']['source']-t)
    side['b']['component_on_a']=side['a']['original_run_ids'][bona]
    side['b']['along_on_a']=side['a']['original_run_pos'][bona]
    # Distances to the fixed ancestral partner, not to a reselected final partner.
    ia=np.flatnonzero(valid['a']&side['a']['trusted']); ib=np.flatnonzero(valid['b']&side['b']['trusted'])
    need(len(ia)>0 and len(ib)>0,'no source-traceable tokens')
    ds=np.linalg.norm((side['a']['source'][ia]+t)[:,None]-side['b']['source'][ib][None],axis=2)
    jsrc=ib[ds.argmin(1)]; isrc=ia[ds.argmin(0)]
    df=np.linalg.norm((points['a']+t)[:,None]-points['b'][None],axis=2)
    df[~valid['a'],:]=np.inf;df[:,~valid['b']]=np.inf
    jfinal=df.argmin(1);ifinal=df.argmin(0)
    candidates=[]; modes={}
    for ai,i in enumerate(ia):
        j=int(jsrc[ai]); bj=int(np.searchsorted(ib,j))
        if isrc[bj]==i and not(side['a']['intact'][i] and side['b']['intact'][j]):
            candidates.append((int(i),j));modes[(int(i),j)]='pre_damage_source_mnn'
        j=int(jfinal[i])
        if ifinal[j]==i and side['a']['intact'][i] and side['b']['intact'][j]:
            candidates.append((int(i),j));modes[(int(i),j)]='intact_final_mnn'
    admitted=[]
    for i,j in candidates:
        if not(side['a']['eligible'][i] and side['b']['eligible'][j]):continue
        if side['a']['component'][i]!=side['b']['component_on_a'][j]:continue
        if np.linalg.norm(side['a']['source'][i]+t-side['b']['source'][j])>policy.source_contact_px:continue
        if df[i,j]>policy.final_max_px:continue
        if df[i,j]>policy.final_main_px and side['a']['intact'][i] and side['b']['intact'][j]:continue
        if (side['a']['endpoint'][i] or side['b']['endpoint'][j]) and not(
            df[i,j]<=3. and side['a']['intact'][i] and side['b']['intact'][j]):continue
        admitted.append((i,j))
    # A token may have one intact and one receded proposal. Keep only uncontested
    # identities; do not improve coverage by choosing a favourable alternative.
    from collections import Counter
    ca=Counter(i for i,j in admitted); cb=Counter(j for i,j in admitted)
    admitted=[(i,j) for i,j in admitted if ca[i]==cb[j]==1]
    admitted=monotone_subset(admitted,side,*outlines['a'][:2],side['a']['original_run_ids'],side['a']['original_run_pos'])
    ordered=[]
    direction_a=int(winding(points['a'][valid['a']])*winding(outlines['a'][0]))
    direction_b=-int(winding(points['b'][valid['b']])*winding(outlines['a'][0]))
    need(direction_a!=0 and direction_b!=0,'degenerate final contour winding')
    for rid in sorted({int(side['a']['component'][i]) for i,j in admitted}):
        group=sorted([(i,j) for i,j in admitted if side['a']['component'][i]==rid],key=lambda q:side['a']['along'][q[0]])
        first_i,first_j=group[0]; previous=(-1,-1)
        for i,j in group:
            progress=((direction_a*(i-first_i))%len(points['a']),(direction_b*(j-first_j))%len(points['b']))
            if progress[0]>previous[0] and progress[1]>previous[1]:ordered.append((i,j));previous=progress
    admitted=ordered
    # Explicitly exclude any crossing final correspondence segments as well.
    crossed=set()
    def cross(u,v):return u[0]*v[1]-u[1]*v[0]
    for k,(i,j) in enumerate(admitted):
        p,q=points['a'][i]+t,points['b'][j]
        for l in range(k):
            ii,jj=admitted[l]; r,s=points['a'][ii]+t,points['b'][jj]
            if cross(q-p,r-p)*cross(q-p,s-p)<-1e-8 and cross(s-r,p-r)*cross(s-r,q-r)<-1e-8:
                crossed.update((k,l))
    admitted=[pair for k,pair in enumerate(admitted) if k not in crossed]
    # Smoothness is evaluated in fixed-source order, on BOTH sides, without
    # crossing ignored destructive pieces or large unobserved gaps.
    by_a={i:(j,df[i,j]) for i,j in admitted}; by_b={j:(i,df[i,j]) for i,j in admitted}
    selected=[]
    for i,j in admitted:
        if df[i,j]>policy.final_main_px:
            good=True
            for s,u,lookup in [('a',i,by_a),('b',j,by_b)]:
                ids=np.flatnonzero(valid[s]); rank=int(np.searchsorted(ids,u))
                neighbors=[int(ids[(rank-1)%len(ids)]),int(ids[(rank+1)%len(ids)])]
                if any(v not in lookup for v in neighbors):good=False;break
                if any(abs(lookup[v][1]-df[i,j])>policy.smooth_gap_delta_px for v in neighbors):good=False;break
                if side[s]['overlap_guard'][u]:good=False;break
            if not good:continue
        selected.append((i,j))
    full={s:np.full(len(old[s]),-2,np.int64) for s in 'ab'}
    tight={s:old[s].copy() for s in 'ab'}
    for s in 'ab':
        reliable_outer=valid[s]&side[s]['trusted']&~side[s]['shared']&~side[s]['destructive']&(side[s]['near_distance']>policy.certain_nonmatch_px)
        full[s][reliable_outer]=-1
        tight[s][valid[s]&(old[s]==-1)&(~reliable_outer)]=-2
        tight[s][~valid[s]]=-2
    for i in np.flatnonzero(old['a']>=0):
        j=int(old['a'][i])
        need(old['b'][j]==i,'old targets not reciprocal')
        if df[i,j]>policy.final_max_px or side['a']['destructive'][i] or side['b']['destructive'][j] or not(side['a']['trusted'][i] and side['b']['trusted'][j]):
            tight['a'][i]=tight['b'][j]=-2
    pair_records=[]
    for i,j in selected:
        full['a'][i]=j;full['b'][j]=i
        pair_records.append(dict(a=i,b=j,distance_px=float(df[i,j]),band='le5' if df[i,j]<=5 else '5to8',
            origin='preserved_existing_label' if old['a'][i]==j else 'new_source_anchored_label',
            pairing=modes[(i,j)],damage='intact' if side['a']['intact'][i] and side['b']['intact'][j] else 'smooth_recession',
            source_distance_px=float(np.linalg.norm(side['a']['source'][i]+t-side['b']['source'][j])),
            source_a=side['a']['source'][i].tolist(),source_b=side['b']['source'][j].tolist(),
            source_component=int(side['a']['component'][i]),source_along_a=float(side['a']['along'][i]),
            source_along_b_on_a=float(side['b']['along_on_a'][j])))
    arrays={}
    for s in 'ab':
        arrays.update({k+'_'+s:v for k,v in dict(full=full[s],tight=tight[s],old=old[s],
            source_points=side[s]['source'],source_trusted=side[s]['trusted'],shared=side[s]['shared'],
            destructive=side[s]['destructive'],endpoint=side[s]['endpoint'],ray_depth=side[s]['ray_depth'],
            near_distance=side[s]['near_distance'],source_arc=side[s]['source_arc']).items()})
    info=dict(status='relabeled_review_only',policy=asdict(policy),pairs=pair_records,
              counts={s:{k:state_counts(v,valid[s]) for k,v in [('old',old[s]),('full',full[s]),('tight',tight[s])]} for s in 'ab'},
              strict_parent_frame=strict,training_admitted=False,
              contour_directions=dict(a=direction_a,b=direction_b),
              policy_notes=['Endpoint guard is four actual final tokens per original seam component, not fixed15px.',
                            '5to8px requires the immediately adjacent fixed-source candidates on both sides; missing neighbors are ignored.',
                            'Source contour gaps are never filled by connecting disconnected runs.',
                            'Overlap damage is recorded strict overlap or added raster intersection relative to pre-damage masks, not subpixel baseline contact.'])
    validate(z,arrays,info)
    return arrays,info

def validate(z,a,info):
    for prefix in ('full','tight'):
        ta,tb=a[prefix+'_a'],a[prefix+'_b']; i=np.flatnonzero(ta>=0);j=ta[i]
        need(np.array_equal(tb[j],i),'new labels not reciprocal')
        need(len(np.unique(j))==len(j),'duplicate new partner')
        need(not np.any(~z['contour_valid_a'][i]) and not np.any(~z['contour_valid_b'][j]),'padding matched')
        d=np.linalg.norm(z['points_rc_a'][i].astype(float)+z['translation_a_to_b_rc']-z['points_rc_b'][j],axis=1)
        need(np.all(d<=POLICY.final_max_px+1e-6),'matched gap exceeds8px')
        need(not a['destructive_a'][i].any() and not a['destructive_b'][j].any(),'destructive match survived')
        if prefix=='tight':need(np.array_equal(z['target_a'][i],j),'tight mode added matches')
    for pair in info['pairs']:
        if pair['distance_px']>5:need(pair['damage']=='smooth_recession','wide intact match')
    for rid in {p['source_component'] for p in info['pairs']}:
        pp=sorted([p for p in info['pairs'] if p['source_component']==rid],key=lambda p:p['source_along_a'])
        need(np.all(np.diff([p['source_along_a'] for p in pp])>0),'nonmonotone source A')
        need(np.all(np.diff([p['source_along_b_on_a'] for p in pp])>0),'nonmonotone source B')
        for s in 'ab':
            origin=pp[0][s]; direction=info['contour_directions'][s]
            progress=[(direction*(p[s]-origin))%len(z['target_'+s]) for p in pp]
            need(np.all(np.diff(progress)>0),'nonmonotone final contour '+s)

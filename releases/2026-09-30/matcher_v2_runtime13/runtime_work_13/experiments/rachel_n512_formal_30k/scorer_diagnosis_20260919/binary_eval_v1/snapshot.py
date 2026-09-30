"""Exact binary input/pooling/MLP tensors. Pooling weights are NOT Attention."""
import numpy as np
import torch
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.diagnostics import _Arrays,_proposal,_json_value,write_snapshot
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.binary_scorer_v1.head import SCALAR_NAMES
from .contracts import read,sha

def snapshot_prediction(pair_id,pair,prediction,*,threshold,provenance,trace):
    if len(trace.clusters)!=len(prediction.clusters) or trace.current:
        raise ValueError('one actual head trace per cluster required')
    a=_Arrays();pr={}
    for name in ('q','unmatched_a','unmatched_b','original_a','original_b','mask_a','mask_b'):
        v=getattr(pair,name);pr[name]=None if v is None else a.add('pair/'+name,v)
    variant=provenance['variant'].removeprefix('binary_')
    if variant not in ('patch','stats'):raise ValueError('binary evidence variant required')
    if variant=='patch':
        for name in ('local_a','local_b','context_a','context_b'):pr[name]=a.add('pair/'+name,getattr(pair,name))
    for side in 'ab':
        g=getattr(pair,'g'+side);n=pair.q.shape[0 if side=='a' else 1]
        pr['geometry_'+side]={name:a.add(f'pair/{side}/{name}',getattr(g,name)[0,:n])
                               for name in ('points','next_step_px','arc_px','outward_normal_rc','normal_reliability')}
        pr['geometry_'+side]['perimeter_px']=float(g.perimeter_px[0])
    saved=[]
    for k,(cluster,layers) in enumerate(zip(prediction.clusters,trace.clusters)):
        prefix=f'cluster_{k:03d}';r=cluster.readout;x=r.inputs
        inputs={name:(None if getattr(x,name) is None else a.add(prefix+'/input/'+name,getattr(x,name)))
                for name in ('edge_ids','q','arc_px','mass_weights','normalized_weights','edge_geometry','statistics','patch_context','pose')}
        values=[]
        for name,layer in layers.items():
            values.append(dict(name=name,kind=layer['kind'],**{field:a.add(prefix+'/mlp/'+name+'/'+field,value)
                for field,value in layer.items() if field!='kind'}))
        saved.append(dict(cluster_id=k,selected=k==prediction.selected_cluster_id,
            proposal=_proposal(a,prefix+'/proposal',cluster.proposal,pair),inputs=inputs,
            pooled_features=None if r.pooled_features is None else a.add(prefix+'/pooled_features',r.pooled_features),
            logit=float(r.logit),score=float(r.score),layers=values))
    meta=dict(schema='binary-cluster-evidence/1',pair_id=pair_id,provenance=provenance,variant=variant,
        scalar_names=list(SCALAR_NAMES),threshold=threshold,has_candidate=prediction.has_candidate,
        numeric_valid=prediction.numeric_valid,selected_cluster_id=prediction.selected_cluster_id,
        score=float(prediction.score),accepted=prediction.accepted,
        translation=_json_value(prediction.translation_a_to_b_rc),pair=pr,clusters=saved,arrays=a.manifest,
        semantics=dict(attention_present=False,local_conflict_classifier=False,learned_refinement=False,
            correspondence='all exact deduplicated member (i,j) pairs; original online Q unchanged',
            mass_weights='Q times mean observed A/B arc length; not a confidence learned by Attention',
            normalized_weights='FP64 conditional normalization for mean pooling; absolute Q/count/mass retained in statistics',
            scalar_names='ordered inputs of the actual MLP, not post-hoc importance scores',
            layers='same-forward detached MLP inputs/outputs/parameters; no second model pass',
            zero_q_edges='retained in the original union; zero mass does not imply a confirmed correspondence',
            selection='all clusters separately scored by one shared head, then argmax logit; different poses never unioned'))
    return meta,a.values

def audit_snapshot(path):
    """Independent tensor arithmetic/replay; never accesses a trained model."""
    from pathlib import Path
    import hashlib
    import torch.nn.functional as F
    path=Path(path);m=read(path);errors=[]
    if m.get('schema')!='binary-cluster-evidence/1':raise ValueError('wrong binary snapshot schema')
    side=path.parent/m['sidecar']['path']
    if sha(side)!=m['sidecar']['sha256']:raise ValueError('binary sidecar hash differs')
    with np.load(side,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    if set(data)!=set(m['arrays']):raise ValueError('binary array membership differs')
    for k,v in data.items():
        declared=m['arrays'][k]
        if (list(v.shape)!=declared['shape'] or v.dtype.str!=declared['dtype']
                or hashlib.sha256(v.tobytes()).hexdigest()!=declared['sha256']):raise ValueError('binary array identity differs:'+k)
    def check(condition,message):
        if not condition:errors.append(message)
    def close(x,y,message,atol=2e-5):
        check(np.shape(x)==np.shape(y) and np.allclose(x,y,rtol=2e-5,atol=atol,equal_nan=False),message)
    def get(v):return data[v]
    p=m['pair'];q=get(p['q']);ga=p['geometry_a'];gb=p['geometry_b']
    points_a=get(ga['points']);points_b=get(gb['points'])
    def arcs(g):
        step=get(g['next_step_px']);return np.minimum(.5*step,3.5)+np.minimum(.5*np.roll(step,1),3.5)
    aa,ab=arcs(ga),arcs(gb)
    check(m['semantics']['attention_present'] is False,'binary Attention falsely declared')
    for c in m['clusters']:
        x={k:None if v is None else get(v) for k,v in c['inputs'].items()};ids=x['edge_ids'];i,j=ids.T
        check(np.array_equal(ids,np.unique(ids,axis=0)),'union has duplicated/unsorted pairs')
        proposal=c['proposal'];expected=np.unique(get(proposal['edge_ids']),axis=0)
        check(np.array_equal(ids,expected),'Scorer did not receive full proposal union')
        check(np.array_equal(x['q'],q[i,j]),'online Q differs')
        close(x['arc_px'],(aa[i]+ab[j])/2,'observed arc differs')
        close(x['mass_weights'],x['q']*x['arc_px'],'Q arc mass differs')
        w=x['mass_weights'];cw=(w.astype(np.float64)/w.astype(np.float64).sum()).astype(w.dtype)
        close(x['normalized_weights'],cw,'conditional pooling differs')
        check(np.array_equal(x['pose'],get(proposal['translation'])),'pose differs from common builder fit')
        delta=points_b[j]-points_a[i]-x['pose'];distance=np.linalg.norm(delta,axis=-1)
        close(x['edge_geometry'][:,2],distance/20,'residual input differs')
        stats=x['statistics'];close(stats[:5],np.array([np.log1p(len(ids)),np.log1p(x['q'].sum()),np.log1p(w.sum()),x['q'].mean(),x['q'].max()]),'absolute statistics differ')
        poses=get(proposal['member_translations_rc']);diameter=np.linalg.norm(poses[:,None]-poses[None,:],axis=-1).max()
        check(diameter<=16.0001,'original candidate diameter exceeds fixed16')
        close(stats[14],diameter/16,'recorded diameter differs')
        row=np.zeros(q.shape[0],q.dtype);col=np.zeros(q.shape[1],q.dtype)
        np.add.at(row,i,x['q']);np.add.at(col,j,x['q'])
        ua,ub=get(p['unmatched_a'])[i],get(p['unmatched_b'])[j]
        outside=(np.maximum(q.sum(1)-row,0)[i]+np.maximum(q.sum(0)-col,0)[j])/2
        unmatched=(ua+ub)/2
        edge=np.stack([np.log1p(x['q']*100),x['q'],distance/20,(distance/20)**2,unmatched,
            np.abs(ua-ub),outside,np.log1p(x['arc_px'])/4],axis=-1)
        close(x['edge_geometry'],edge,'edge geometry vector differs')
        coverage=np.array([aa[np.unique(i)].sum()/max(1,ga['perimeter_px']),
                           ab[np.unique(j)].sum()/max(1,gb['perimeter_px'])])
        overlap=0.
        if p['mask_a'] is not None and p['mask_b'] is not None:
            ma,mb=get(p['mask_a']),get(p['mask_b']);dr,dc=np.rint(x['pose']).astype(int)
            r0,c0=max(0,-dr),max(0,-dc);r1,c1=min(ma.shape[0],mb.shape[0]-dr),min(ma.shape[1],mb.shape[1]-dc)
            if r1>r0 and c1>c0:overlap=float((ma[r0:r1,c0:c1]*mb[r0+dr:r1+dr,c0+dc:c1+dc]).sum())/max(1,min(ma.sum(),mb.sum()))
        tail=np.array([1/(np.square(cw).sum()*len(cw)),coverage.mean(),coverage.min(),
            (cw*distance).sum()/20,np.sqrt((cw*distance**2).sum())/20,distance.max()/20,
            (cw*unmatched).sum(),(cw*outside).sum(),overlap,diameter/16,
            np.log1p(len(proposal['merged_hypothesis_ids']))])
        close(stats[5:],tail,'remaining absolute/geometry statistics differ')
        layers=c['layers'];lookup={r['name']:r for r in layers}
        for layer in layers:
            inp=torch.from_numpy(get(layer['input']));actual=get(layer['output'])
            if layer['kind']=='linear':
                result=F.linear(inp,torch.from_numpy(get(layer['weight'])),torch.from_numpy(get(layer['bias'])))
            elif layer['kind']=='gelu':result=F.gelu(inp)
            else:raise ValueError('unknown binary layer')
            close(result.numpy(),actual,'layer replay differs:'+layer['name'])
        for prefix in ('edge_mlp','cluster_mlp'):
            chain=[r for r in layers if r['name'].startswith(prefix+'.')]
            for before,after in zip(chain,chain[1:]):close(get(before['output']),get(after['input']),'MLP chain differs')
        if m['variant']=='patch':
            feats=np.concatenate([(get(p['local_a'])[i]+get(p['local_b'])[j])/2,
                np.abs(get(p['local_a'])[i]-get(p['local_b'])[j]),
                (get(p['context_a'])[i]+get(p['context_b'])[j])/2,
                np.abs(get(p['context_a'])[i]-get(p['context_b'])[j])],axis=-1)
            close(x['patch_context'],feats,'patch/context inputs differ')
            close(get(lookup['edge_mlp.0']['input']),np.concatenate([feats,x['edge_geometry']],axis=-1),'edge input differs')
            h=get(lookup['edge_mlp.3']['output']);pooled=np.concatenate([(h*cw[:,None]).sum(0),h.max(0)])
            close(get(c['pooled_features']),pooled,'mean/max pooling differs')
            inp=np.concatenate([pooled,stats])
        else:
            check(x['patch_context'] is None and c['pooled_features'] is None,'stats arm unexpectedly used features')
            check(not any(r['name'].startswith('edge_mlp') for r in layers),'stats arm has edge MLP')
            inp=stats
        close(get(lookup['cluster_mlp.0']['input']),inp,'cluster MLP input differs')
        logit=float(get(lookup['cluster_mlp.4']['output']).reshape(-1)[0])
        close(logit,c['logit'],'final logit differs')
        close(float(torch.tensor(logit).sigmoid()),c['score'],'final sigmoid differs')
    if m['clusters']:
        winner=int(np.argmax([c['logit'] for c in m['clusters']]))
        check(winner==m['selected_cluster_id'],'winner is not shared Scorer argmax')
        close(m['score'],m['clusters'][winner]['score'],'reported winning score differs')
        close(m['translation'],get(m['clusters'][winner]['inputs']['pose']),'reported winning pose differs')
    else:check(m['selected_cluster_id']==-1 and m['translation'] is None,'empty candidate invented pose')
    check(m['has_candidate']==bool(m['clusters']),'candidate presence differs')
    check(m['accepted']==bool(m['has_candidate'] and m['numeric_valid'] and m['score']>=m['threshold']),'acceptance differs')
    return dict(status='passed' if not errors else 'failed',errors=errors,clusters=len(m['clusters']),
                variant=m['variant'],attention_present=False,raw_union_q_pooling_and_mlp_replayed=True,
                no_external_model_or_gt_opened=True)

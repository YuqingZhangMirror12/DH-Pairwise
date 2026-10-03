"""Read model-only exports; do not load optimizers or touch training sources."""
import sys
import numpy as np
from pathlib import Path
from .io import sha,checked
from .inference import module,RotationEnsemble


def load(protocol,matcher_key,head_keys):
    import torch
    root=protocol['runtime'];sys.path.insert(0,root)
    api=module('matcher_v2_v1.model_runtime',root)
    hashing=module('curriculum_training_v1.checkpoint_io',root)
    head_api=module('binary_scorer_v1.head',root)
    compat=module('s7_consensus_v1.compatibility',root)
    def saved(key):
        record=protocol['models'][key]
        if sha(record['path'])!=record['sha256']:raise ValueError('export bytes changed: '+key)
        x=torch.load(record['path'],map_location='cpu',weights_only=False)
        api.check_export(x)
        if x['updates']!=record['update'] or hashing.tree_sha(x['model'])!=record['model_state_sha256']:
            raise ValueError('export state identity differs: '+key)
        if x['binding']['matcher_v2_experiment']['arm']!='B3':raise ValueError('B3 only')
        return x
    selected=saved(matcher_key);spec=selected['binding']['model_spec']
    matcher=api.load_export_matcher(selected,root)
    # The immutable runtime13 adapter has a known top-level mode omission.
    # Admit only its exact frozen/eval children; do not relax inference guards.
    adapter=module('matcher_v2_v1.adapter',root)
    if not isinstance(matcher,adapter.MatcherV2Adapter):raise ValueError('unexpected adapter')
    if any(p.requires_grad for p in matcher.parameters()) or any(m.training for n,m in matcher.named_modules() if n):
        raise ValueError('mode correction forbidden for unfrozen/active children')
    before=hashing.tree_sha(matcher.state_dict());matcher.eval()
    if hashing.tree_sha(matcher.state_dict())!=before:raise ValueError('mode correction changed tensors')
    sim=saved('matcher_sim') if matcher_key!='matcher_sim' else selected
    sim_state={k[8:]:v for k,v in sim['model'].items() if k.startswith('matcher.')}
    sim_sha=hashing.tree_sha(sim_state);heads={}
    for key in head_keys:
        x=saved(key);s=x['binding']['model_spec']
        if s['geometry']!=spec['geometry']:raise ValueError('head geometry differs')
        if hashing.tree_sha({k[8:]:v for k,v in x['model'].items() if k.startswith('matcher.')})!=sim_sha:
            raise ValueError('head must retain its own B3 SIM-selected Matcher provenance')
        variant=key.split('_')[0]
        if s['scorer_variant']!=variant:raise ValueError('head variant differs')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(x['binding']['common_plan']['head_seed']);head=head_api.BinaryClusterHead(variant)
        hs={k[5:]:v for k,v in x['model'].items() if k.startswith('head.')}
        head.load_state_dict(hs,strict=True)
        if hashing.tree_sha(head.state_dict())!=hashing.tree_sha(hs):raise ValueError('head tensor import differs')
        heads[key]=head.requires_grad_(False).eval().to('cuda:0')
    geometry=compat.CompatibilityConfig(**spec['geometry'])
    return RotationEnsemble(matcher.to('cuda:0'),geometry,heads,root)


class Population:
    def __init__(self,protocol,name,phase):
        self.name=name;self.phase=phase;self.spec=protocol['populations'][name]
        self.meta=checked(self.spec['manifest']);self.runtime=protocol['runtime']
        if name=='sim_cal':
            if phase!='development':raise ValueError('CAL is never a retained TEST population')
            api=module('s7_consensus_v1.data',self.runtime)
            self.dataset=api.Dataset(self.spec['manifest']['path'],self.spec['manifest']['sha256']);self.collate=api.collate
            self.pairs=self.dataset.entries
        else:
            if sha(self.spec['inputs']['path'])!=self.spec['inputs']['sha256']:raise ValueError('real input bytes changed')
            wanted=set(self.spec['roles']['real_test']['pair_ids']) if phase=='test' else set(self.spec['roles']['real_cal']['pair_ids'])|set(self.spec['roles']['real_select']['pair_ids'])
            self.pairs=[p for p in self.meta['pairs'] if p['pair_id'] in wanted]
            if len(self.pairs)!=len(wanted):raise ValueError('population identity mismatch')
            if any((p['fold']==0)!=(phase=='test') for p in self.pairs):raise ValueError('TEST leaked into development')
            self.arrays=np.load(self.spec['inputs']['path'],allow_pickle=False)
            self.index={k:i for i,k in enumerate(self.meta['fragment_ids'])}
        if len({p['pair_id'] for p in self.pairs})!=len(self.pairs):raise ValueError('duplicate pair')

    def batch(self,start,size):
        from .core import INPUTS
        chunk=self.pairs[start:start+size]
        if self.name=='sim_cal':
            b=self.collate([self.dataset[i] for i in range(start,start+len(chunk))])
            return [p['pair_id'] for p in chunk],{k:b[k].numpy() for k in INPUTS}
        b={k:[] for k in INPUTS}
        for p in chunk:
            for side in 'ab':
                i=self.index[p['fragment_'+side+'_id']]
                b['mask_'+side].append(np.unpackbits(self.arrays['packed_masks'][i],axis=-1)[:,:800][None].astype(np.float32))
                b['points_rc_'+side].append(self.arrays['points'][i].astype(np.float32))
                b['contour_valid_'+side].append(self.arrays['valid'][i].astype(bool))
        return [p['pair_id'] for p in chunk],{k:np.stack(v) for k,v in b.items()}

    def targets(self,protocol,only_ids=None):
        # Called only AFTER predictions have been durably written. Model and
        # candidate selectors never receive this return value.
        result={};wanted=None if only_ids is None else set(only_ids)
        if self.name=='sim_cal':
            for i,e in enumerate(self.pairs):
                if wanted is not None and e['pair_id'] not in wanted:continue
                sample,_,_=self.dataset[i]
                target=sample.translation_a_to_b_rc.tolist() if sample.label and sample.translation_valid else None
                if sample.label and target is None:raise ValueError('missing SIM global translation')
                result[e['pair_id']]=dict(label=bool(sample.label),target=target,fold=None,seam_type='unknown')
        else:
            gt={g['pair_id']:g for g in checked(protocol['gt'])['positive_pairs']} if self.name=='dunhuang_cv' else {}
            for p in self.pairs:
                if wanted is not None and p['pair_id'] not in wanted:continue
                g=gt.get(p['pair_id']) if p['label'] else None
                if self.name=='dunhuang_cv' and p['label'] and g is None:raise ValueError('missing valid positive GT')
                if g and (g['fragment_a_token']!=p['fragment_a_id'] or g['fragment_b_token']!=p['fragment_b_id']):raise ValueError('GT orientation differs')
                result[p['pair_id']]=dict(label=bool(p['label']),target=None if g is None else g['translation_gt_a_to_b_rc'],fold=p['fold'],
                    seam_type=protocol['seam_types'].get(p['pair_id'],'unknown') if self.name=='dunhuang_cv' else 'unknown')
        return result

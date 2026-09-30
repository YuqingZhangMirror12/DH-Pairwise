"""Retain valid completed groups across the explicit identity-fallback repair.

Copies group metadata and hardlinks immutable archives into a new output root.
The failed generation and its exact provenance remain unchanged. This is a
recovery dataset, not a claim of bit-identical fresh generation with the new code.
"""
import copy
import os
from pathlib import Path
from ..s7_compound_v1.materialize import read, save_json, digest


def compatible_profile(old, new):
    if old == new:
        return True
    ignored={'name','scale_rule','scale_identity_fallback'}
    return (not old.get('scale_identity_fallback',False)
        and new.get('scale_identity_fallback') is True
        and {k:v for k,v in old.items() if k not in ignored} ==
            {k:v for k,v in new.items() if k not in ignored})


def reuse(source, dest, profile, *, groups, seed, source_plan):
    source, dest = Path(source).resolve(), Path(dest).resolve()
    protocol=read(source/'protocol.json');state=read(source/'status.json')
    if state['status']!='failed' or (Path('/proc')/str(state['pid'])).exists():
        raise ValueError('reuse only a stopped failed generation')
    if (protocol['options']['groups']!=groups or protocol['options']['seed']!=seed
            or protocol['source_plan_sha256']!=digest(source_plan)
            or not compatible_profile(protocol['distribution_revision'],profile)):
        raise ValueError('recovery changed source plan, seed, schedule or scientific recipe')
    copied=[]
    for path in sorted((source/'groups').glob('*.json')):
        r=read(path);slot=r['slot']
        if not 0<=slot<groups or path.name!=f'{slot:05d}.json' or len(r['entries'])!=2:
            raise ValueError('invalid completed group')
        fresh=copy.deepcopy(r)
        for entry in fresh['entries']:
            for key in ('artifact_path','target_metadata'):
                old=source/entry[key] if key=='artifact_path' else Path(entry[key])
                rel=old.resolve().relative_to(source)
                if not old.is_file() or old.stat().st_size==0:
                    raise ValueError('missing archive in completed group')
                target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
                os.link(old,target)
                if key=='target_metadata':entry[key]=str(target)
        save_json(dest/'groups'/path.name,fresh)
        copied.append(dict(slot=slot,source_group_sha256=digest(path)))
    receipt=dict(source=str(source),source_protocol_sha256=digest(source/'protocol.json'),
        source_generator_sha256=protocol['generator_sha256'],groups=len(copied),samples=2*len(copied),
        source_profile=protocol['distribution_revision']['name'],new_profile=profile['name'],
        retained_groups=copied,immutable_archives_hardlinked=True,
        original_source_modified=False,fresh_seed_replay_equivalence_claim=False,
        note='Completed groups remain the accepted revision6 realizations; only missing slots use revision7 identity fallback. Corrosion/source/mirror/length quotas are unchanged.')
    save_json(dest/'reuse_receipt.json',receipt)
    return receipt

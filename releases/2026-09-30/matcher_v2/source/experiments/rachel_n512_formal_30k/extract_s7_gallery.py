"""Export bounded exact S7 TRAIN examples for scientific viewing, CPU only.

Does not regenerate augmentations, run models, or alter training artifacts.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def read_case(root, entry):
    with np.load(root / entry['artifact_path'], allow_pickle=False) as z:
        arrays = {k: np.array(z[k]) for k in z.files}
    report = json.loads(str(arrays['report_json'].item()))
    return arrays, report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    manifest = Path(args.manifest)
    data = json.loads(manifest.read_text())
    root = Path(data['artifact_root'])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    entries = data['entries']
    chosen = []
    used = set()

    def choose(category, candidates, count=1, predicate=None):
        n = 0
        for e in candidates:
            if e['pair_id'] in used:
                continue
            arrays, report = read_case(root, e)
            if predicate and not predicate(e, arrays, report):
                continue
            chosen.append((category, e, arrays, report))
            used.add(e['pair_id'])
            n += 1
            if n == count:
                break
        if n != count:
            raise RuntimeError('not enough selected examples: ' + category)

    def rows(recipe, label=True, changed=True):
        return [e for e in entries if e['s7_recipe'] == recipe and
                bool(e['label']) == label and (changed is None or e['changed_pair'] == changed)]

    choose('reference_clean', rows('reference_e1', changed=False))
    for tier in ['mild', 'moderate']:
        choose('reference_' + tier, rows('reference_e1'), predicate=lambda e,z,r,t=tier:
               t in [r.get('side_'+s,{}).get('tier') for s in 'ab'])
    for recipe in ['wave', 'local', 'seam_gaps', 'partial_curve']:
        choose(recipe, rows(recipe), count=2)
        choose('negative_' + recipe, rows(recipe, label=False))
    choose('area_imbalance', [e for e in entries if e['label'] and
        e.get('source_stratum') == 'union_positive_tiny' and
        e['s7_recipe'] == 'reference_e1'], count=2)
    choose('fallback', rows('seam_gaps', changed=False))
    for pattern in [(1, 2, 2), (3, 1, 1)]:
        choose('gen5_' + ''.join(map(str, pattern)), rows('gen5_partition', changed=False),
            predicate=lambda e,z,r,p=pattern: tuple(r['gen5_partition']['ordered_pattern']) == p)
    records = []
    for number, (category, e, arrays, report) in enumerate(chosen, 1):
        for s in 'ab':
            shape = tuple(int(x) for x in arrays['mask_'+s+'_shape'])
            new = np.unpackbits(arrays['mask_'+s+'_packed'], axis=-1)[..., :shape[-1]].reshape(shape)[0].astype(bool)
            if e['s7_recipe'] == 'gen5_partition':
                # The archived sample already IS the new grouped geometry;
                # never pretend this is a reconstruction of the five inputs.
                old = new.copy()
            else:
                mask_path = Path(e['source_root']) / e['source_row']['fragment_'+s]['model_mask_path']
                old = np.asarray(Image.open(mask_path).convert('L')) > 0
            if old.shape != new.shape or np.any(new & ~old):
                raise RuntimeError('before/after frame or deletion invariant differs: '+e['pair_id'])
            arrays['before_'+s+'_packed'] = np.packbits(old, axis=-1)
            arrays['before_'+s+'_shape'] = np.asarray(old.shape, dtype=np.int64)
        filename = '%02d_%s.npz' % (number, category)
        np.savez_compressed(out / filename, **arrays)
        record = dict(number=number, category=category, entry=e, file=filename,
            source_artifact=str(root/e['artifact_path']), report=report)
        records.append(record)
        print(json.dumps(dict(number=number, category=category, pair_id=e['pair_id'],
              actual_changed=e['changed_pair'], matches=int((arrays['target_a'] >= 0).sum()))), flush=True)
    result = dict(source_manifest=str(manifest), split='train', stats=data['stats'],
        selection='first two changed positives and first changed negative per recipe in frozen manifest order; separate clean/mild/moderate, tiny-union and fallback examples; selected to illustrate mechanisms, not estimate population quality',
        generated_augmentation=False, model_inference=False, records=records)
    (out/'examples.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()

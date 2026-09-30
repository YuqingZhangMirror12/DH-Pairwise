"""Copy approved immutable implementation and add the separate full-data pipeline."""
import argparse,json,shutil
from pathlib import Path
from ..s7_compound_v1.materialize import digest,save_json,read

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('fresh source snapshot only')
    binding=read(a.base/'source_binding.json')
    for name,sha in binding.items():
        if digest(a.base/name)!=sha:raise ValueError('base source changed:'+name)
    for name in binding:
        dest=a.out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a.base/name,dest)
    rel=Path('experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/aggressive_data_full_v17')
    for f in Path(__file__).parent.iterdir():
        if f.suffix not in ('.py','.md'):continue
        dest=a.out/rel/f.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    hashes={str(f.relative_to(a.out)):digest(f) for f in sorted(a.out.rglob('*.py'))}
    save_json(a.out/'source_binding.json',hashes)
    print(json.dumps(dict(files=len(hashes),source_binding_sha256=digest(a.out/'source_binding.json'),base_binding_sha256=digest(a.base/'source_binding.json'))))
if __name__=='__main__':main()


"""Copy verified prior immutable source, adding only independent v17 package."""
import argparse,hashlib,json,shutil
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('fresh source only')
    binding=json.loads((a.base/'source_binding.json').read_text())
    for name,digest in binding.items():
        if sha(a.base/name)!=digest:raise ValueError('previous immutable source changed: '+name)
    a.out.mkdir(parents=True)
    for name in binding:
        dest=a.out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a.base/name,dest)
    package=Path(__file__).parent;relative=Path('experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/aggressive_data_v17')
    for source in package.iterdir():
        if source.suffix not in ('.py','.md'):continue
        dest=a.out/relative/source.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    hashes={str(f.relative_to(a.out)):sha(f) for f in sorted(a.out.rglob('*.py'))}
    (a.out/'source_binding.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(files=len(hashes),binding_sha256=sha(a.out/'source_binding.json'),base_binding_sha256=sha(a.base/'source_binding.json'))))
if __name__=='__main__':main()

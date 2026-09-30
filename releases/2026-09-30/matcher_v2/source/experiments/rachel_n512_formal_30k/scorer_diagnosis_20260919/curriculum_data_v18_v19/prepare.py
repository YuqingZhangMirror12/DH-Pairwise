"""Fresh immutable review source from the bound v17 implementation."""
import argparse,json,hashlib,shutil
from pathlib import Path

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();base=Path(args.base);out=Path(args.out)
    if out.exists():raise ValueError('fresh source only')
    binding=json.loads((base/'source_binding.json').read_text())
    for name,sha in binding.items():
        if digest(base/name)!=sha:raise ValueError('bound base modified: '+name)
    for name in binding:
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/name,dest)
    relative=Path('experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919/curriculum_data_v18_v19')
    for path in Path(__file__).parent.iterdir():
        if path.name.startswith('._') or path.suffix not in ('.py','.md'):continue
        dest=out/relative/path.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    hashes={str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*.py'))}
    (out/'source_binding.json').write_text(json.dumps(hashes,indent=2)+'\n')
    receipt=dict(source=str(out),source_binding_sha256=digest(out/'source_binding.json'),
        base_binding_sha256=digest(base/'source_binding.json'),python_files=len(hashes),bound_training_source_changed=False)
    (out/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
if __name__=='__main__':main()

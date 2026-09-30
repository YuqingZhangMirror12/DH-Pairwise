"""Snapshot CPU-only source into a fresh version, never mutate bound jobs."""
import argparse,hashlib,json,shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    root=Path.cwd();out=Path(a.out).resolve()
    if out.exists():raise ValueError('fresh source path only')
    files=[]
    for folder in ('staging/pairwise_v0_2','experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919'):
        files.extend((root/folder).rglob('*.py'))
    hashes={}
    for f in files:
        if '__pycache__' in f.parts:continue
        r=f.relative_to(root);target=out/r;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
        hashes[str(r)]=hashlib.sha256(f.read_bytes()).hexdigest()
    (out/'source_binding.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps({'out':str(out),'python_files':len(hashes),'training_started':False}))

if __name__=='__main__':main()

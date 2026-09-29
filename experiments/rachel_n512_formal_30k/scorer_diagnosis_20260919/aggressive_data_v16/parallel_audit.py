"""Same independent per-record checks, bounded four-CPU execution."""
import argparse,json,shutil,os
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
from .review_audit import read,sha,need,audit_record,materialize

BASE=None
def initialize(baseline):
    global BASE
    BASE=Path(baseline)
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    materialize.initialize(read(BASE/'protocol.json')['options'])

def one(record):
    try:return audit_record(record,BASE)
    except Exception as e:return dict(id=record['id'],status='failed',error=repr(e))

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.pilot);out=Path(a.out)
    if out.exists():raise ValueError('fresh audit output required')
    complete=read(root/'generation_complete.json');need(not (root/'failure.json').exists(),'generation failure takes precedence')
    manifest=read(root/'manifest.json');records=manifest['entries'];protocol=read(root/'protocol.json')
    need(len(records)==complete['pairs'],'complete manifest');out.mkdir(parents=True)
    receipts=[];errors=[]
    with ProcessPoolExecutor(4,initializer=initialize,initargs=(protocol['baseline'],)) as pool:
        for result in pool.map(one,records):
            (receipts if result['status']=='passed' else errors).append(result)
            (out/'status.json').write_text(json.dumps(dict(status='auditing',audited=len(receipts)+len(errors),
                total=len(records),errors=len(errors),training_started=False))+'\n')
    counts=Counter(r['detail']['trim']['size_class'] for r in records if r['label'])
    if not (len(records)>0 and counts['smaller']*10==len(records)//2*7):errors.append(dict(error='accepted70/30 quota'))
    result=dict(status='failed' if errors else 'passed',pairs=len(records),receipts=receipts,errors=errors,
        generation_complete=complete,pilot_manifest_sha256=sha(root/'manifest.json'),positive_size_counts=dict(counts),
        actual_gap_definition='original partners frozen; bilateral primary peak5–15; unresolved separate',
        full_training_generation_authorized=False,training_started=False)
    (out/'pixel_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if errors:raise ValueError('pixel audit failed:'+repr(errors[:4]))
    for r in records:
        for source,folder in ((r['sample_path'],'samples'),(r['proof_path'],'proof'),(r['baseline_sample_path'],'baseline')):
            dst=out/folder/Path(source).name;dst.parent.mkdir(exist_ok=True);shutil.copy2(source,dst)
    for name in ('manifest.json','protocol.json','generation_audit.json','generation_complete.json'):shutil.copy2(root/name,out/name)
    (out/'status.json').write_text(json.dumps(dict(status='passed',pairs=len(records),errors=0))+'\n')
    print(json.dumps(dict(status='passed',pairs=len(records),side_counts=counts)))

if __name__=='__main__':main()

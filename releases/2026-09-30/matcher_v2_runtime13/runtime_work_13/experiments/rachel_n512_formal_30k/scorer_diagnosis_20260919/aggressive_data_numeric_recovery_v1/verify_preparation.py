import argparse,os,unittest
from pathlib import Path
from recover import inventory,save

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU fixtures only')
    out=Path(a.out)
    if out.exists():raise ValueError('preserve previous test receipt')
    before=inventory(Path(__file__).parent)
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_recovery'))
    same=before==inventory(Path(__file__).parent)
    result=dict(status='passed' if r.wasSuccessful() and same else 'failed',tests=r.testsRun,errors=len(r.errors),
        failures=len(r.failures),skipped=len(r.skipped),source_sha256=before,source_unchanged=same,
        gpu_used=False,real_data_opened=False,pipeline_started=False)
    save(out,result)
    if result['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()

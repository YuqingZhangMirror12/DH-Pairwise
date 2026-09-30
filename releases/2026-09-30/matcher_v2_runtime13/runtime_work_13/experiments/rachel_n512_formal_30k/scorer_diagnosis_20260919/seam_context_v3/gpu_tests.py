import unittest
import torch
from .prepare import save
import argparse


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);args=p.parse_args()
    torch.set_default_device('cuda')
    prefix=__package__+'.tests.'
    suite=unittest.defaultTestLoader.loadTestsFromNames([prefix+'test_model',prefix+'test_metrics',prefix+'test_augmentation'])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    save(args.report,dict(passed=result.wasSuccessful(),tests=result.testsRun,
        failures=[str(x) for x in result.failures],errors=[str(x) for x in result.errors],
        device=torch.cuda.get_device_name(0),torch_version=torch.__version__))
    raise SystemExit(0 if result.wasSuccessful() else 1)

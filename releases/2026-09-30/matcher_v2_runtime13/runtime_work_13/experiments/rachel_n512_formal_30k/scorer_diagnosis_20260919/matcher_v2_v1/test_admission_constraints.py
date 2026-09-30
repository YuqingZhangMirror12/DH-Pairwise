from dataclasses import replace
import unittest
from .admission_constraints import population_counts
from .test_additive_exposure import extras
from .additive_exposure import build_additive_ledger
from ..curriculum_training_v1 import test_exposure as baseline


def generation(shortages):
    request=dict(train=6000,select=900,test=900);n=sum(shortages.values())
    return dict(requested_rows=7800,authorized_omissions=n,total_rows=7800-n,
        datasets={k:dict(rows=v-shortages[k],omitted=shortages[k],requested_rows=v) for k,v in request.items()})


class AdmissionConstraintsTests(unittest.TestCase):
    def test_zero_or_small_explicit_shortfall(self):
        for shortages in (dict(train=0,select=0,test=0),dict(train=10,select=5,test=5)):
            g=generation(shortages);self.assertEqual(sum(population_counts(g).values()),g['total_rows'])

    def test_total_not_twenty_per_split(self):
        with self.assertRaisesRegex(ValueError,'twenty-row'):
            population_counts(generation(dict(train=10,select=10,test=1)))

    def test_unexplained_shortfall_is_not_allowed(self):
        g=generation(dict(train=0,select=0,test=0));g['datasets']['train']['rows']-=1
        with self.assertRaises(ValueError):population_counts(g)

    def test_no_new_duplicate_original_pair_in_exposure_catalog(self):
        rows=list(extras());rows[1]=replace(rows[1],source_base_key=rows[0].source_base_key)
        with self.assertRaisesRegex(ValueError,'original pairing'):
            build_additive_ledger(baseline.ledger(),rows,((0,16,8),),seed=9)


if __name__=='__main__':unittest.main()

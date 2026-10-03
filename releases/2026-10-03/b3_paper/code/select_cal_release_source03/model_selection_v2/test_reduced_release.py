import copy
import unittest
from . import heldout_reduced_release as m


def row(i, stage, gen, label, role):
    return dict(pair_id=f'{role}-{i}', sample_sha256=f'{role}-sample-{i}', model_tensors_sha256=f'{role}-input-{i}',
        supervised_tensors_sha256=f'{role}-supervised-{i}', stage=stage, generator=gen, label=label,
        parent_ids=[role+'-parent'], donor_parent_ids=[role+'-parent'], base_pair_ids=[role+'-base'],
        donor_base_pair_ids=[], fragment_ids=[role+'-fragment'], donor_fragment_ids=[])


def population(role):
    rows=[]
    for stage in m.rp.STAGES:
        for gen in m.rp.GENS:
            n=59 if role=='cal' and stage=='v17_filtered' and gen in ('Gen3','Gen5') else 60
            for label in (False,True):
                for _ in range(n): rows.append(row(len(rows),stage,gen,label,role))
    for label in (False,True):
        for _ in range(80): rows.append(row(len(rows),'strict_straight','straight_strip',label,role))
    return rows


class Tests(unittest.TestCase):
    def test_exact_user_waivers(self): m.check_waivers(list(m.WAIVED),2,[])
    def test_extra_or_duplicate_shortfall_blocked(self):
        for missing in (list(m.WAIVED)+[('cal','Gen4',1,'v18')], list(m.WAIVED)*2, []):
            with self.assertRaises(ValueError): m.check_waivers(missing,2,[])
    def test_nonzero_not_general_amnesty(self):
        for code,errors in ((1,[]),(0,[]),(2,['failure.json'])):
            with self.assertRaises(ValueError):m.check_waivers(list(m.WAIVED),code,errors)
    def test_real_population(self):
        for role,n in (('cal',1596),('select',1600)):
            rows=population(role); self.assertEqual(len(rows),n); m.validate_population(rows,role)
    def test_wrong_cell_not_hidden_by_same_total(self):
        rows=population('cal');rows[0]['generator']='Gen4'
        with self.assertRaises(ValueError):m.validate_population(rows,'cal')
    def test_same_pixels_rejected(self):
        rows=population('select');rows[0]['model_tensors_sha256']=rows[1]['model_tensors_sha256']
        with self.assertRaises(ValueError):m.validate_population(rows,'select')
    def test_source_and_donor_isolation(self):
        rows={r:population(r) for r in m.rp.ROLES};allowed={r:{r+'-parent'} for r in m.rp.ROLES}
        self.assertFalse(any(m.cross_fold(rows,allowed,{'TRAIN-family'}).values()))
        rows['cal'][0]['donor_parent_ids']=['TRAIN-family']
        with self.assertRaises(ValueError):m.cross_fold(rows,allowed,{'TRAIN-family'})
    def test_cross_role_tensor_overlap(self):
        rows={r:population(r) for r in m.rp.ROLES};allowed={r:{r+'-parent'} for r in m.rp.ROLES}
        rows['cal'][0]['model_tensors_sha256']=rows['select'][0]['model_tensors_sha256']
        with self.assertRaises(ValueError):m.cross_fold(rows,allowed,set())


if __name__=='__main__':unittest.main()

import copy,unittest
from .build_snapshot_crop20 import prepare,REVISION


def fixture():
    rows=[];population=[];groups=[];versions={}
    for version in ['v17.5','v18']:
        floor=dict(before=dict(common_over_smaller_perimeter=.30),after=dict(common_over_smaller_perimeter=.24),
            originally_short=False,changed=True,passed=True,erosion_and_light_subject_to_this_gate=False)
        detail=dict(spec=dict(light=[1.,4.],light_scope='whole_postprimary_contour',pristine_protection_enabled=False,pristine_min_fraction=0),
            crop_only_seam_floor=floor,trim=dict(applied=True,retained_fraction=.7),
            background={'a':dict(whole_postprimary_contour=True,contact_protection_enabled=False,applied_max_depth_px=3)})
        rows.append(dict(id=version+'-1',version=version,label='正例',detail=detail,audit=dict(status='passed',crop_only_seam_floor=floor)))
        population.append(dict(id=version+'-1',version=version,label=True,crop_applied=True,crop_floor=floor,trim_fraction=.3,gap_peak=12,inherited=10))
        groups.append(dict(version=version,type='mild',ids=[version+'-1'],count=1));versions[version]=dict(pairs=1)
    artifact=dict(protocol=dict(revision=REVISION,source_binding_sha256='unit-fixture'),complete=False,
        rows=rows,population=population,groups=groups,source_root='unit-fixture',display_slots=2)
    receipt=dict(status='probe_complete',versions=versions,updated_unix=1790596788)
    data=dict(id='report:1b87f48a-d200-4451-96aa-32b1bdf5fcaa',queries={'archive':dict(rows=[dict(kept=True)])},report={})
    return artifact,receipt,data


class SnapshotTests(unittest.TestCase):
    def test_probe_not_claimed_complete_and_archive_preserved(self):
        a,r,d=fixture();old=copy.deepcopy(d['queries']['archive']);out,v=prepare(a,r,d,'unit-fixture')
        self.assertEqual(out['queries']['archive'],old);self.assertFalse(v['complete']);self.assertEqual(out['buildStatus'],'updating')
    def test_negative_does_not_fill_positive_quota(self):
        a,r,d=fixture();a['rows'][0]['label']='负例'
        with self.assertRaises(ValueError):prepare(a,r,d,'unit-fixture')
    def test_final20_gate_is_not_silently_applied(self):
        a,r,d=fixture();a['rows'][0]['detail']['crop_only_seam_floor']['erosion_and_light_subject_to_this_gate']=True
        with self.assertRaises(ValueError):prepare(a,r,d,'unit-fixture')
    def test_below20_crop_rejected(self):
        a,r,d=fixture();a['rows'][0]['detail']['crop_only_seam_floor']['after']['common_over_smaller_perimeter']=.19
        with self.assertRaises(ValueError):prepare(a,r,d,'unit-fixture')
    def test_incomplete_groups_cannot_claim_complete(self):
        a,r,d=fixture();a['complete']=True;r['status']='complete'
        with self.assertRaises(ValueError):prepare(a,r,d,'unit-fixture')
    def test_old_light_scope_rejected(self):
        a,r,d=fixture();a['rows'][0]['detail']['spec']['light_scope']='untouched_original'
        with self.assertRaises(ValueError):prepare(a,r,d,'unit-fixture')

if __name__=='__main__':unittest.main()

"""Only new endpoint-CAL publication, shard and TEST-seal orchestration guards."""
import copy
import unittest
from . import endpoint_calibration as m


def fixture():
    done=dict(status='released_with_authorized_shortfalls_and_exact_deduplication',rows=3183)
    verification=dict(status='verified_data_release',rows=3183,raw_generated_rows=3196,
        original_builder_actual_return=2,authorized_missing_quotas=[['cal','Gen3',19,'v17_filtered'],['cal','Gen5',77,'v17_filtered']],
        cross_role_overlaps={'pair_id':0,'parent_ids':0},train_test_source_overlap=0)
    manifest=dict(split='cal',entries=[dict(pair_id=str(i),model_tensors_sha256=str(i),label=i<793) for i in range(1587)])
    return done,dict(returncode=0),verification,manifest


class Tests(unittest.TestCase):
    def test_actual_deduplicated_cal_admitted(self): m.check_release(*fixture())

    def test_old_or_partial_cal_rejected(self):
        for n in (1500,1596,1586):
            args=fixture(); args[-1]['entries']=args[-1]['entries'][:n] if n<1587 else args[-1]['entries']+[args[-1]['entries'][0]]*(n-1587)
            with self.assertRaises(ValueError):m.check_release(*args)

    def test_changed_balance_or_repeated_input_rejected(self):
        for key,value in (('label',False),('model_tensors_sha256','1')):
            args=fixture();args[-1]['entries'][0][key]=value
            with self.assertRaises(ValueError):m.check_release(*args)

    def test_failed_publish_or_unapproved_missing_quota_rejected(self):
        args=fixture();args[1]['returncode']=1
        with self.assertRaises(ValueError):m.check_release(*args)
        args=fixture();args[2]['authorized_missing_quotas'].append(['cal','Gen4',2,'v18'])
        with self.assertRaises(ValueError):m.check_release(*args)

    def test_two_shards_exact_odd_population(self):
        a,b=[m.phase_indices(1587,i) for i in (0,1)]
        self.assertEqual((len(a),len(b)),(794,793));self.assertFalse(set(a)&set(b))
        self.assertEqual(sorted(a+b),list(range(1587)))

    def test_test_requires_correct_new_cal_seal(self):
        p=dict(setting_lock={'sha256':m.SETTING_SHA},populations={'sim_cal':{'manifest':{'sha256':'new-cal'}}})
        r={'sha256':'new-protocol'}
        s=dict(protocol=r,setting_lock=p['setting_lock'],cal_manifest=p['populations']['sim_cal']['manifest'],
            cal_pairs=1587,cal_positive=793,test_used=False,thresholds={'four_view':{},'identity_baseline':{}})
        m.assert_test_admission(p,s,r)
        for key,value in (('cal_pairs',1596),('test_used',True),('cal_manifest',{'sha256':'old-cal'}),('cal_positive',798)):
            changed=copy.deepcopy(s);changed[key]=value
            with self.assertRaises(ValueError):m.assert_test_admission(p,changed,r)

    def test_new_cal_fpr_uses_actual_794_negatives(self):
        rows=[dict(label=False,numeric_valid=True,has_candidate=True,score=i/794) for i in range(794)]
        for rate,budget in ((.01,7),(.02,15),(.05,39)):
            t=m.fpr_threshold(rows,rate)
            self.assertEqual(sum(r['score']>=t for r in rows),budget)


if __name__=='__main__':unittest.main()

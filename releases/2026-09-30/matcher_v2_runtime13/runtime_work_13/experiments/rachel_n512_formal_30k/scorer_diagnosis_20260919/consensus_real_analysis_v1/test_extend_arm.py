import copy
import json
from pathlib import Path
import tempfile
import unittest

from .extend_arm import append_policies, check_completion, prior_rows, prior_bindings
from .analyze import sha


def completion():
    return dict(status='complete', arm='m12', all_three_populations_verified=True,
        cases_exported=11, training_modified=False,
        jobs=[dict(split=s,status='complete',returncode=0) for s in
              ('sim_test_v14','dunhuang_cv','turufan')])


def population(layout=True):
    meta=dict(split='real' if layout else 'ood',pairs=[],fragment_source_group={})
    rr=[]
    for i in range(20):
        a,b=f'fragment-{i}-a',f'fragment-{i}-b'
        y=i%2==0
        meta['fragment_source_group'].update({a:f'source-{i}',b:f'source-{i}'})
        meta['pairs'].append(dict(pair_id=str(i),label=y,fold=i//4,
                                  fragment_a_id=a,fragment_b_id=b))
        rr.append(dict(pair_id=str(i),label=y,fold=i//4,score=.42 if y else .22,
                       decision_valid=True,layout_good_20=y if layout else None))
    return meta,rr


class ExtensionTests(unittest.TestCase):
    def test_load_named_prior_threshold_arm_preserves_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'm12_turufan'; folder.mkdir()
            protocol = dict(status='complete', variant='threshold', arm='m12', split='turufan',
                            checkpoint_sha256='checkpoint', selected_epoch=8, threshold=.21)
            (folder/'protocol.json').write_text(json.dumps(protocol))
            raw = dict(pair_id='id', label=True, score=.3, has_candidate=True, numeric_valid=True,
                       layout20=False, gt_known=False, error_px=None, translation=[1, 2], fold=0)
            (folder/'case_diagnostics.jsonl').write_text(json.dumps(raw)+'\n')
            bindings = {str(p.resolve()): sha(p) for p in folder.iterdir()}
            entry = dict(checkpoint_sha256='checkpoint', epoch=8, sim_threshold=.21)
            evidence = {}
            result = prior_rows('threshold_m12', entry, 'turufan',
                                {'threshold_m12': temp}, bindings, evidence)
            self.assertEqual(result[0]['score'], .3)
            self.assertIsNone(result[0]['layout_good_20'])
            self.assertEqual(len(evidence), 2)
            with self.assertRaisesRegex(ValueError, 'selected model'):
                prior_rows('threshold_m12', dict(entry, epoch=22), 'turufan',
                           {'threshold_m12': temp}, bindings, {})
            (folder/'case_diagnostics.jsonl').write_text(json.dumps(dict(raw, score=.4))+'\n')
            with self.assertRaisesRegex(ValueError, 'changed prior'):
                prior_rows('threshold_m12', entry, 'turufan', {'threshold_m12': temp}, bindings, {})

    def test_missing_prior_arm_not_guessed_from_key(self):
        with self.assertRaisesRegex(ValueError, 'directory required'):
            prior_rows('threshold_m12', {}, 'turufan', {}, {}, {})

    def test_prior_source_conflict_rejected(self):
        base = dict(summary=dict(status='complete', all_source_hashes_unchanged=True,
            inputs={'same': 'a'}, extensions=[dict(input_sha256={'same': 'b'})]))
        with self.assertRaisesRegex(ValueError, 'binding conflict'):
            prior_bindings(base)

    def test_accept_three_complete_jobs(self):
        check_completion(completion(),'m12')

    def test_reject_failed_population(self):
        x=completion();x['jobs'][0]['returncode']=1
        with self.assertRaises(ValueError):check_completion(x,'m12')

    def test_reject_duplicate_population(self):
        x=completion();x['jobs'][0]['split']='turufan'
        with self.assertRaises(ValueError):check_completion(x,'m12')

    def test_reject_wrong_arm(self):
        with self.assertRaises(ValueError):check_completion(completion(),'scratch_fixed')

    def test_exclusion_does_not_recalibrate(self):
        meta,rr=population()
        p,d,o=append_policies(meta,rr,.21,set(),True)
        q,e,f=append_policies(meta,rr,.21,{'0'},True)
        self.assertEqual(d,e)
        self.assertEqual(o,f)
        for policy in p:
            self.assertEqual(p[policy]['original'],q[policy]['original'])
            self.assertEqual(q[policy]['corrected']['n'],19)
        self.assertEqual(p['bounded_max_f1']['corrected']['f1'],1.)

    def test_no_gt_layout_remains_null(self):
        meta,rr=population(False)
        p,_,_=append_policies(meta,rr,.21,set(),False)
        self.assertIsNone(p['sim_frozen']['original']['joint_f1'])
        self.assertIsNone(p['bounded_max_f1']['corrected']['layout_accuracy'])

    def test_reject_reordered_population(self):
        meta,rr=population()
        with self.assertRaises(ValueError):append_policies(meta,list(reversed(rr)),.21,set(),True)

    def test_inputs_preserved(self):
        meta,rr=population();before=copy.deepcopy((meta,rr))
        append_policies(meta,rr,.21,{'0'},True)
        self.assertEqual((meta,rr),before)


if __name__=='__main__':unittest.main()

"""Targeted CPU regression tests for the new published-release scan boundary."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import released_protocol as rp
from .checkpoint_scan import UPDATES, resource_admission, save
from .protocol import MANIFEST_SCHEMA, STAGE_WEIGHTS, digest, validate_protocol
from .selection import REPORT_SCHEMA, LOSS_SEMANTICS, NATIVE_MICROBATCH1, select_checkpoint


class ScanTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); outputs={}
        for role in ('select', 'cal'):
            rows=[]; removed={True: 4 if role=='select' else 7, False: 0 if role=='select' else 6}
            for stage, generators in rp.DEFAULT_GENERATORS.items():
                for gen in generators:
                    for label in (True, False):
                        for n in range(80 if stage=='strict_straight' else 60):
                            if stage=='v17_filtered' and gen=='Gen3' and n < removed[label]: continue
                            pid=f'{role}-{stage}-{gen}-{label}-{n}'
                            rows.append(dict(pair_id=pid, sample_sha256=digest(pid), stage=stage,
                                generator=gen, label=label, parent_ids=[role+'/parent'],
                                base_pair_ids=[pid+'/base'], fragment_ids=[pid+'/a',pid+'/b'],
                                donor_parent_ids=[], donor_base_pair_ids=[], donor_fragment_ids=[]))
            outputs[role]=dict(lineage=save(self.root/role/'lineage.json',
                dict(schema=MANIFEST_SCHEMA, role=role, entries=rows)))
        ref=save(self.root/'verification.json',dict(status='verified_data_release',rows=3183,
            train_test_source_overlap=0,cross_role_overlaps={'parent_ids':0},task3_overlay_applied=False,
            outputs=outputs))
        self.mock=patch.object(rp,'VERIFICATION_SHA',ref['sha256']);self.mock.start();self.addCleanup(self.mock.stop)
        self.plan=rp.freeze_release_protocol(ref,UPDATES)

    def report(self,update,loss=1,covered=True):
        b=self.plan['manifest_bindings']['select']
        rows=[dict({k:e[k] for k in ('pair_id','sample_sha256','stage','generator','label')},
            gt_known=e['label'],layout20=e['label'] and covered,candidate_coverage=e['label'] and covered,
            numeric_valid=True,has_candidate=True,matcher_loss=loss,loss_semantics=LOSS_SEMANTICS,
            loss_implementation=NATIVE_MICROBATCH1,physical_microbatch=1) for e in b['manifest']['entries']]
        return dict(schema=REPORT_SCHEMA,protocol_sha256=self.plan['sha256'],update=update,role='select',
            manifest_sha256=b['sha256'],status='completed',returncode=0,checkpoint_sha256='a'*64,
            model_state_sha256='b'*64,rows=rows)

    def test_new_release_validates_with_actual_not_original_target_counts(self):
        validate_protocol(self.plan,verify_files=True)
        self.assertEqual(self.plan['manifest_bindings']['select']['pair_count'],1596)
        self.assertEqual(self.plan['manifest_bindings']['cal']['pair_count'],1587)

    def test_foreign_publication_rejected(self):
        x=copy.deepcopy(self.plan);x['publication']['sha256']='a'*64
        x['sha256']=digest({k:v for k,v in x.items() if k!='sha256'})
        with self.assertRaisesRegex(ValueError,'published_03'):validate_protocol(x)

    def test_real_or_cal_selection_forbidden_even_with_resealed_digest(self):
        for field in ('real_used','test_used_for_selection','cal_used_for_matcher_selection'):
            x=copy.deepcopy(self.plan);x[field]=True;x['sha256']=digest({k:v for k,v in x.items() if k!='sha256'})
            with self.assertRaisesRegex(ValueError,'SELECT-only'):validate_protocol(x)

    def test_lineage_tamper_detected_from_real_file(self):
        p=Path(self.plan['manifest_bindings']['select']['path']);p.write_text('{}')
        with self.assertRaisesRegex(ValueError,'SHA256'):validate_protocol(self.plan,verify_files=True)

    def test_full_candidate_inventory_required(self):
        with self.assertRaisesRegex(ValueError,'incomplete'):select_checkpoint(self.plan,[self.report(UPDATES[0])])

    def test_minimum_loss_in_layout_band_not_coverage_key(self):
        reports=[self.report(u,2 if u!=UPDATES[-1] else 1) for u in UPDATES]
        result=select_checkpoint(self.plan,reports)
        self.assertEqual(result['selected_update'],UPDATES[-1]);self.assertFalse(result['coverage_affects_selection'])

    def test_bad_numeric_rows_cannot_win_zero_loss(self):
        reports=[self.report(u) for u in UPDATES];reports[-1]['rows'][0]['numeric_valid']=False
        with self.assertRaisesRegex(ValueError,'numeric-invalid'):select_checkpoint(self.plan,reports)

    def test_active_process_blocks_even_with_zero_instant_utilization(self):
        x=resource_admission([0],'0, GPU-a, 1759\n1, GPU-b, 0','GPU-a, 71686')
        self.assertFalse(x['admitted'])
        self.assertTrue(resource_admission([1],'0, GPU-a, 1759\n1, GPU-b, 0','GPU-a, 71686')['admitted'])

    def test_memory_without_visible_process_not_assumed_free(self):
        self.assertFalse(resource_admission([0],'0, GPU-a, 1759','')['admitted'])

    def test_save_preserves_old_receipts(self):
        p=self.root/'receipt.json';save(p,{'old':1})
        with self.assertRaisesRegex(ValueError,'preserve'):save(p,{'new':1})
        self.assertEqual(json.loads(p.read_text()),{'old':1})


if __name__=='__main__':unittest.main()

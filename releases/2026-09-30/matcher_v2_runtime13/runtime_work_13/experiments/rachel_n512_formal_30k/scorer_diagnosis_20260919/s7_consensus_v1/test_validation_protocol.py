"""Synthetic records exercise protocol logic, not model performance."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .config import TrainingConfig
from .evaluation import summarize_validation
from .metrics import choose_threshold,summarize
from .prepare_data import load_validation_plan,sha,prepare,SOURCE_TRAIN_SHA
from .test_metrics_cache import row
from .validation_protocol import check_rows,layout,learning_curve_csv


def population(prefix,n):
    return [dict(row(prefix+str(i),i%2==0,.9 if i%2==0 else .1,True,True),
                 recipe='clean' if i%10<2 else 'partial',matcher_batch_loss=.1) for i in range(n)]


def fixture(paired=False,cal=4,select=6):
    views=('clean','hard') if paired else ('mixed',)
    data={split:{view:population(split,count) for view in views}
          for split,count in (('cal',cal),('select',select))}
    contract=dict(schema='s7-consensus-data-contract/1' if paired else 's7-consensus-data-contract/2',
        validation={split+'_'+view:dict(pair_count=count) for split,count in (('cal',cal),('select',select))
                    for view in views})
    if not paired:contract['validation_design']=dict(kind='single_mixed',physical_samples=cal+select)
    return data,contract


class ValidationProtocolTests(unittest.TestCase):
    def test_legacy_pair_count_computed_not_hardcoded(self):
        data,contract=fixture(True)
        report=summarize_validation(data,contract,'scorer',TrainingConfig())
        self.assertEqual((report['cal_independent_pairs'],report['select_independent_pairs']),(4,6))
        self.assertEqual(report['validation_physical_samples'],20)
        self.assertAlmostEqual(report['threshold'],.3)
        self.assertEqual(report['selection_value'],1.)

    def test_new6000_population_not_doubled_into_views(self):
        data,contract=fixture(cal=2000,select=4000)
        report=summarize_validation(data,contract,'scorer',TrainingConfig())
        self.assertEqual(report['validation_physical_samples'],6000)
        self.assertEqual(report['select_distinct_pair_ids'],4000)
        self.assertEqual(set(report['selected']),{'mixed'})
        self.assertNotIn('select_independent_pairs',report)
        self.assertFalse(report['clean_hard_are_paired_views'])
        self.assertFalse(report['recipe_diagnostics_affect_selection'])

    def test_threshold_uses_cal_only_even_when_select_wants_another(self):
        data,contract=fixture()
        for r in data['cal']['mixed']:r['score']=.61 if r['label'] else .59
        for r in data['select']['mixed']:r['score']=.29 if r['label'] else .21
        report=summarize_validation(data,contract,'scorer',TrainingConfig())
        self.assertAlmostEqual(report['threshold'],.6)
        self.assertEqual(report['selected']['mixed']['recall'],0.)

    def test_decimal_grid_boundary_matches_saved_threshold(self):
        data,contract=fixture()
        for r in data['cal']['mixed']:r['score']=.61 if r['label'] else .59
        report=summarize_validation(data,contract,'scorer',TrainingConfig())
        self.assertEqual(report['threshold'],.6)
        self.assertEqual(float(f"{report['threshold']:.2f}"),report['threshold'])

    def test_mixed_selection_is_pooled_not_mean_of_unequal_recipe_groups(self):
        data,contract=fixture(select=20)
        for r in data['select']['mixed']:
            if r['recipe']=='partial' and r['label']:r['score']=.1
        report=summarize_validation(data,contract,'scorer',TrainingConfig())
        self.assertEqual(report['selection_value'],summarize(data['select']['mixed'],report['threshold'])['joint_f1'])
        recipe_mean=sum(v['joint_f1'] for v in report['recipe_diagnostics'].values())/2
        self.assertNotAlmostEqual(report['selection_value'],recipe_mean)

    def test_count_duplicates_and_cross_split_ids_rejected(self):
        for kind in ('missing','duplicate','overlap'):
            with self.subTest(kind=kind):
                data,contract=fixture()
                if kind=='missing':data['cal']['mixed'].pop()
                elif kind=='duplicate':data['cal']['mixed'][1]['pair_id']=data['cal']['mixed'][0]['pair_id']
                else:data['select']['mixed'][0]['pair_id']=data['cal']['mixed'][0]['pair_id']
                with self.assertRaises(ValueError):check_rows(data,contract)

    def test_paired_ids_and_labels_must_match(self):
        for key,value in (('pair_id','different'),('label',False)):
            data,contract=fixture(True);data['cal']['hard'][0][key]=value
            with self.assertRaises(ValueError):check_rows(data,contract)

    def test_new_contract_cannot_fall_back_to_legacy(self):
        _,contract=fixture();contract.pop('validation_design')
        with self.assertRaisesRegex(ValueError,'explicit'):layout(contract)
        with self.assertRaises(ValueError):choose_threshold(dict(mixed=population('x',4)),TrainingConfig())

    def test_unknown_layout_gt_not_silently_dropped(self):
        data,contract=fixture();data['select']['mixed'][0]['gt_known']=False
        with self.assertRaisesRegex(ValueError,'known layout'):check_rows(data,contract)

    def test_mixed_matcher_and_csv(self):
        data,contract=fixture()
        report=summarize_validation(data,contract,'matcher',TrainingConfig())
        self.assertEqual(report['selection_value'],1.)
        self.assertAlmostEqual(report['key'][2],-.1)
        text=learning_curve_csv([dict(report,epoch=2,updates=1500,exposures=48000)])
        self.assertIn('mixed_layout20',text)
        self.assertNotIn('hard_layout20',text)


def write_fixture(root,with_test=False):
    """Test-only receipts with no image files and no formal experiment output."""
    plan=dict(schema='s7-consensus-validation-plan/2',kind='single_mixed',status='passed',physical_samples=6000,splits={})
    if with_test:plan['schema']='s7-consensus-validation-plan/3'
    sizes=(('cal',1500),('select',1500),('test',3000)) if with_test else (('cal',2000),('select',4000))
    for split,n in sizes:
        entries=[dict(pair_id=split+str(i),label=i%2==0,source_pair_id=split+'base'+str(i//3),
                      sources=[split+'manuscript_recto',split+'other_verso']) for i in range(n)]
        manifest=root/(split+'.json');manifest.write_text(json.dumps(dict(entries=entries)))
        audit=root/(split+'_audit.json');audit.write_text(json.dumps(dict(status='passed',pairs=n)))
        pixels=root/(split+'_pixels.json');pixels.write_text(json.dumps(dict(status='passed',pairs=n,
            all_actual_masks_reconstructed=True,topology_checked_all=True,
            rows=[dict(pair_id=e['pair_id']) for e in entries])))
        plan['splits'][split]=dict(manifest=manifest.name,manifest_sha256=sha(manifest),pair_count=n,
            validation_path=audit.name,validation_sha256=sha(audit),pixel_audit_path=pixels.name,pixel_audit_sha256=sha(pixels))
    path=root/'plan.json';path.write_text(json.dumps(plan))
    return path,plan


class ValidationPlanTests(unittest.TestCase):
    def test_new30k_test_is_separate_from_validation_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,_=write_fixture(Path(tmp),with_test=True)
            manifests,lineage,design=load_validation_plan(path,{'trainparent'})
            self.assertEqual(design['physical_samples'],3000)
            self.assertEqual(design['test_physical_samples'],3000)
            test=manifests.pop('test_mixed')
            self.assertEqual(test['pair_count'],3000)
            self.assertEqual(set(manifests),{'cal_mixed','select_mixed'})
            data,contract=fixture(cal=1500,select=1500)
            contract.update(validation_design=design,validation=manifests,test=dict(mixed=test))
            self.assertEqual(check_rows(data,contract)['validation_physical_samples'],3000)

    def test_test_cannot_share_source_with_cal_or_select(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,plan=write_fixture(root,with_test=True)
            spec=plan['splits']['test'];file=root/spec['manifest']
            record=json.loads(file.read_text());record['entries'][0]['sources']=['selectmanuscript_verso']
            file.write_text(json.dumps(record));spec['manifest_sha256']=sha(file)
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'TEST shares source'):
                load_validation_plan(path,set())

    def test_train_augmentation_donor_cannot_leak_into_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,plan=write_fixture(root,with_test=True)
            spec=plan['splits']['cal'];file=root/spec['manifest']
            record=json.loads(file.read_text());record['entries'][0]['augmentation_donor_sources']=['trainparent_recto']
            file.write_text(json.dumps(record));spec['manifest_sha256']=sha(file)
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'overlaps training'):
                load_validation_plan(path,{'trainparent'})

    def test_test_cannot_be_in_training_validation_map(self):
        data,contract=fixture()
        contract['validation']['test_mixed']=dict(pair_count=3000)
        with self.assertRaisesRegex(ValueError,'differ'):
            layout(contract)

    def test_v14_cannot_be_bound_to_old_validation_implicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            records={
                'status.json':dict(status='complete',sample_count=24000),
                'validation.json':dict(status='passed',source_identity_and_anchor_quotas_checked=True),
                'pipeline_status.json':dict(status='complete',stage='data_ready'),
                'train.json':dict(protocol=dict(distribution_revision=dict(partial_min_smaller_perimeter_fraction=.15)))}
            with patch(__package__+'.prepare_data.read',side_effect=lambda p:records[Path(p).name]), \
                    patch(__package__+'.prepare_data.sha',return_value=SOURCE_TRAIN_SHA):
                with self.assertRaisesRegex(ValueError,'new6K'):
                    prepare(root/'train',root/'old82_370',root/'historical',root/'contract.json')

    def test_exact6000_source_base_and_instance_counts_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,_=write_fixture(Path(tmp))
            manifests,lineage,design=load_validation_plan(path,{'trainparent'})
            self.assertEqual(design['physical_samples'],6000)
            self.assertEqual(manifests['cal_mixed']['source_count'],2)
            self.assertEqual(manifests['cal_mixed']['base_pair_count'],667)
            self.assertEqual(lineage['cal_mixed'],{'calmanuscript','calother'})

    def test_recto_verso_train_leak_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,_=write_fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError,'overlaps training'):
                load_validation_plan(path,{'calmanuscript'})

    def test_receipts_cannot_be_for_other_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,plan=write_fixture(root)
            spec=plan['splits']['cal'];file=root/spec['pixel_audit_path']
            record=json.loads(file.read_text());record['rows'][0]['pair_id']='wrong'
            file.write_text(json.dumps(record));spec['pixel_audit_sha256']=sha(file)
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'actual validation population'):
                load_validation_plan(path,set())

    def test_cal_select_family_alias_leak_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,plan=write_fixture(root)
            spec=plan['splits']['select'];file=root/spec['manifest']
            record=json.loads(file.read_text());record['entries'][0]['sources']=['calmanuscript_verso']
            file.write_text(json.dumps(record));spec['manifest_sha256']=sha(file)
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'CAL and SELECT share source'):
                load_validation_plan(path,set())

    def test_train_row_cannot_be_relabelled_as_validation_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,plan=write_fixture(root)
            spec=plan['splits']['cal'];file=root/spec['manifest']
            record=json.loads(file.read_text());record['entries'][0]['source_row']=dict(split='train')
            file.write_text(json.dumps(record));spec['manifest_sha256']=sha(file)
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'TRAIN release row'):
                load_validation_plan(path,set())

    def test_manifest_mutation_and_wrong_total_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,plan=write_fixture(root)
            plan['physical_samples']=904;path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'completed6K'):load_validation_plan(path,set())
            plan['physical_samples']=6000;path.write_text(json.dumps(plan))
            file=root/plan['splits']['cal']['manifest'];file.write_text(file.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'manifest changed'):load_validation_plan(path,set())


if __name__=='__main__':unittest.main()

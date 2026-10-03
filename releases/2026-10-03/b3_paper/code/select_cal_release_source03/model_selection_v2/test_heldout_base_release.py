import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from . import heldout_base_release as h


class HeldoutBaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = dict(schema='task1-evaluation-only-parent-plan/1', folds={}, hard_exclusions={},
                         exclusion_families={k:['forbidden'] for k in h.EXCLUSIONS})
        for role in h.ROLES:
            parent = self.root / (role+'.png')
            rgba = np.full((800,800,4), 255 if role == 'cal' else 200, np.uint8)
            Image.fromarray(rgba).save(parent)
            image = dict(path=str(parent), file_sha256=h.file_sha(parent), mode='RGBA',
                         has_embedded_original_alpha=True)
            self.plan['folds'][role] = [dict(family=role,role=role,parent_image=image,
                allowed_same_family_variants=[image],edge_donors_must_be_from_same_role_families=True)]
        parent = self.plan['folds']['cal'][0]['parent_image']
        raw = self.root/'raw'; raw.mkdir()
        donors=[]
        for i in range(4):
            path=self.root/f'edge{i}.png';path.write_bytes(bytes([i]))
            donors.append(dict(family='cal',parent_image_path=parent['path'],parent_image_sha256=parent['file_sha256'],
                raw_edge_path=str(path),raw_edge_sha256=h.file_sha(path),variant_path=str(path),
                variant_sha256=h.file_sha(path),seed=i))
        fragments=[]
        for i in range(4):
            path=raw/f'{i}.jpg'
            rgb=np.zeros((800,800,3),np.uint8);rgb[100:700,100+150*i:250+150*i]=200
            Image.fromarray(rgb).save(path,quality=100,subsampling=0)
            fragments.append(dict(id=i,mask_id=i,path=str(path),sha256=h.file_sha(path)))
        with (raw/'label.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['id','mask_id','image_name','center_x','center_y','width','height','neighbors']);writer.writeheader()
            for i in range(4):
                writer.writerow(dict(id=i,mask_id=i,image_name='cal.png',center_x=175+150*i,center_y=400,width=150,height=600,
                    neighbors=';'.join(str(j) for j in (i-1,i+1) if 0<=j<4)))
        self.group=dict(role='cal',generator='gen4voronoi',family='cal',group_id='cal__cal__pilot',
            parent_image_path=parent['path'],parent_image_sha256=parent['file_sha256'],edge_donors=donors,
            csv_path=str(raw/'label.csv'),csv_sha256=h.file_sha(raw/'label.csv'),fragments=fragments)
        self.plan_path=self.root/'parents.json';self.plan_path.write_text(json.dumps(self.plan))
        self.manifest=dict(status='pilot_complete',parent_plan_sha256=h.file_sha(self.plan_path),groups=[self.group])
        self.manifest_path=self.root/'groups.json';self.manifest_path.write_text(json.dumps(self.manifest))
        self.code_root=Path(__file__).resolve().parents[4]
        self.sources={n:h.file_sha(self.code_root/Path(*(h.MODULE_PREFIX+n).split('.')).with_suffix('.py')) for n in h.MODULES}

    def test_parent_fold_excluded_and_crossrole_rejected(self):
        for bad in ('excluded','crossrole','image'):
            plan=copy.deepcopy(self.plan)
            if bad=='excluded':plan['exclusion_families'][h.EXCLUSIONS[0]].append('cal')
            elif bad=='crossrole':plan['folds']['select'][0]['family']='cal'
            else:plan['folds']['select'][0]['allowed_same_family_variants'][0]['file_sha256']=plan['folds']['cal'][0]['parent_image']['file_sha256']
            with self.subTest(bad=bad), self.assertRaises(ValueError):h.approved_parents(plan)

    def test_actual_group_is_bound(self):
        self.assertGreater(len(h.validate_group(self.group,h.approved_parents(self.plan))),4)
        Path(self.group['fragments'][0]['path']).write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'source changed'):h.validate_group(self.group,h.approved_parents(self.plan))

    def test_wrong_donor_role_and_duplicate_edges_rejected(self):
        for bad in ('role','duplicate','namespace'):
            group=copy.deepcopy(self.group)
            if bad=='role':group['edge_donors'][0]['family']='select'
            elif bad=='duplicate':group['edge_donors'][1]=group['edge_donors'][0]
            else:group['group_id']='123'
            with self.subTest(bad=bad),self.assertRaises(ValueError):h.validate_group(group,h.approved_parents(self.plan))

    def test_native_group_and_target_gates_preserved(self):
        out=self.root/'new'
        kwargs=dict(parent_plan_path=self.plan_path,expected_parent_sha=h.file_sha(self.plan_path),
            groups_manifest_path=self.manifest_path,expected_groups_sha=h.file_sha(self.manifest_path),
            output_new=out,frozen_runtime_root=self.code_root,expected_sources=self.sources)
        receipt=h.materialize_new_groups(**kwargs)
        self.assertEqual('preprocessed_not_mixed_selection_complete',receipt['status'])
        marker=receipt['groups'][0]['marker']
        self.assertEqual('processed_group',marker['status'])
        self.assertEqual(4,len(marker['fragments']))
        self.assertGreater(len(receipt['groups'][0]['admitted_positive_pairs']),0)
        row=receipt['groups'][0]['admitted_positive_pairs'][0]
        self.assertEqual('cal',row['split'])
        self.assertNotIn('target_audit',row['fragment_a'])
        self.assertFalse((out/'pairs/train.jsonl').exists())
        with self.assertRaisesRegex(ValueError,'entirely new'):h.materialize_new_groups(**kwargs)

    def test_wrong_source_hash_fails_before_output(self):
        bad=dict(self.sources);bad['rachel_preprocess']='0'*64
        with self.assertRaisesRegex(ValueError,'source changed'):
            h.native_modules(self.code_root,bad)

    def test_canonical_catalog_preserves_safe_relative_paths_and_audit_separation(self):
        old=self.root/'empty_old_index.json'
        old.write_text(json.dumps(dict(release_root=str(self.root),fragments={},
                                       eligible_positive_rows={'cal':[],'select':[]})))
        self.plan['existing_base_index_contract']={'sha256':h.file_sha(old)}
        self.plan_path.write_text(json.dumps(self.plan))
        self.manifest['parent_plan_sha256']=h.file_sha(self.plan_path)
        self.manifest_path.write_text(json.dumps(self.manifest))
        new=self.root/'new'
        h.materialize_new_groups(parent_plan_path=self.plan_path,expected_parent_sha=h.file_sha(self.plan_path),
            groups_manifest_path=self.manifest_path,expected_groups_sha=h.file_sha(self.manifest_path),
            output_new=new,frozen_runtime_root=self.code_root,expected_sources=self.sources)
        catalog=h.merge_catalog(parent_plan_path=self.plan_path,expected_parent_sha=h.file_sha(self.plan_path),
            existing_index_path=old,expected_existing_sha=h.file_sha(old),
            new_release_path=new/'base_release.json',expected_new_release_sha=h.file_sha(new/'base_release.json'),
            output_new=self.root/'catalog',frozen_runtime_root=self.code_root,expected_sources=self.sources)
        self.assertTrue(catalog['positive_rows']['cal'])
        self.assertEqual(4,len(catalog['fragment_target_audit']))
        for f in catalog['fragments']:
            self.assertNotIn('target_audit',f['row'])
            self.assertTrue(f['row']['model_mask_path'].startswith('model/masks_800/'))
            self.assertFalse(Path(f['row']['model_mask_path']).is_absolute())
        for rel,sha in catalog['copied_files_sha256'].items():
            self.assertEqual(sha,h.file_sha(self.root/'catalog'/rel))
        for row in catalog['positive_rows']['cal']:
            self.assertEqual('val',row['split'])
            self.assertTrue(h.load_native_pair(catalog['release_root'],'cal',dict(row,split='cal'),
                h.native_modules(self.code_root,self.sources)['rachel_training_dataset']).translation_valid)

    def test_new_catalog_extension_reuses_old_receipt_and_checks_only_new_pairs(self):
        previous_plan=self.root/'previous_parent_plan.json'
        previous_plan.write_bytes(self.plan_path.read_bytes())
        self.plan['evidence']={'previous_parent_plan':{'sha256':h.file_sha(previous_plan)}}
        self.plan_path.write_text(json.dumps(self.plan))
        self.manifest['parent_plan_sha256']=h.file_sha(self.plan_path)
        self.manifest_path.write_text(json.dumps(self.manifest))
        new=self.root/'new_extension'
        result=h.materialize_new_groups(parent_plan_path=self.plan_path,expected_parent_sha=h.file_sha(self.plan_path),
            groups_manifest_path=self.manifest_path,expected_groups_sha=h.file_sha(self.manifest_path),
            output_new=new,frozen_runtime_root=self.code_root,expected_sources=self.sources)
        old_root=self.root/'previous_catalog';old_root.mkdir()
        old_base=self.root/'previous_base.json'
        old_base.write_text(json.dumps(dict(parent_plan_sha256=h.file_sha(previous_plan),groups=[])))
        old=old_root/'catalog.json'
        old.write_text(json.dumps(dict(schema='mixed-heldout-native-catalog/1',
            parent_plan_sha256=h.file_sha(previous_plan),release_root=str(old_root),
            native_source_sha256=self.sources,fragments=[],fragment_target_audit=[],
            positive_rows={'cal':[],'select':[]},copied_files_sha256={},existing_index_sha256='b'*64,
            new_release_sha256=h.file_sha(old_base))))
        verification=self.root/'previous_verification.json'
        verification.write_text(json.dumps(dict(catalog={'path':str(old),'sha256':h.file_sha(old)},
            base_release={'path':str(old_base),'sha256':h.file_sha(old_base)},
            all_positive_rows={'cal':0,'select':0},native_source_sha256=self.sources,
            native_rejections=[],catalog_rejections=[],source_files_unchanged=True)))
        with patch.object(h,'load_native_pair',wraps=h.load_native_pair) as native:
            extended=h.extend_verified_catalog(parent_plan_path=self.plan_path,expected_parent_sha=h.file_sha(self.plan_path),
                previous_parent_plan_path=previous_plan,expected_previous_parent_sha=h.file_sha(previous_plan),
                verified_catalog_path=old,expected_catalog_sha=h.file_sha(old),
                verification_path=verification,expected_verification_sha=h.file_sha(verification),
                new_release_path=new/'base_release.json',expected_new_release_sha=h.file_sha(new/'base_release.json'),
                output_new=self.root/'extended',frozen_runtime_root=self.code_root,expected_sources=self.sources)
        admitted=sum(len(g['admitted_positive_pairs']) for g in result['groups'])
        self.assertEqual(admitted,native.call_count)
        self.assertEqual(admitted,extended['new_positive_pairs_native_checked']['cal'])
        self.assertEqual({'cal':0,'select':0},extended['previous_pairs_reused_without_inference_or_native_retest'])
        self.assertEqual('cal.png',extended['fragments'][0]['row']['split_unit_id'])
        self.assertFalse(extended['complete_mixed_select_cal'])
        for relative,digest in extended['copied_files_sha256'].items():
            self.assertEqual(digest,h.file_sha(self.root/'extended'/relative))

    def test_unapproved_boundary_derivation_rejected_before_artifact_reads(self):
        for field,value in (('policy','changed'),('all_other_original_gates_preserved',False)):
            group=copy.deepcopy(self.group)
            group['source_derivation']={'policy':'parent-inherited-outer-boundary-exact/1',
                'exact_one_block_patch':True,'all_other_original_gates_preserved':True}
            group['source_derivation'][field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'unapproved base boundary'):
                h.validate_group(group,h.approved_parents(self.plan),verify_files=False)


if __name__=='__main__':unittest.main()

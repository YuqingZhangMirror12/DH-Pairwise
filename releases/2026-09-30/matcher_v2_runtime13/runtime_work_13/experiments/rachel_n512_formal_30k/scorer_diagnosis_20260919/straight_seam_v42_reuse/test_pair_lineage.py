import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

from .pair_identity import base_pair_sha256
from .pair_lineage import source_token_key,compare_records,resolve_masks,original_rows,digest


def row():
    return dict(source_root='/old',source_pair_id='old-pair',label=True,
        source_row=dict(fragment_a=dict(fragment_token='a'),fragment_b=dict(fragment_token='b')))


class PairLineageTest(unittest.TestCase):
    def test_original_tokens_not_damage_recipe_or_pair_id(self):
        a=row();b=copy.deepcopy(a);b.update(pair_id='changed',recipe='different',source_pair_id='alias')
        b['source_row']['fragment_a'],b['source_row']['fragment_b']=b['source_row']['fragment_b'],b['source_row']['fragment_a']
        self.assertEqual(source_token_key(a),source_token_key(b))

    def test_different_original_pair_not_same_manuscript(self):
        a=row();b=copy.deepcopy(a);b['source_row']['fragment_b']['fragment_token']='c'
        self.assertNotEqual(source_token_key(a),source_token_key(b))

    def test_mirror_variant_with_new_name_is_rejected(self):
        a=np.array([[1,1],[1,0]],bool);b=np.ones((4,3),bool)
        old=[dict(pre_damage_pair_sha256=base_pair_sha256(a,b),source_token_key='old')]
        new=[dict(pair_id='new-damage',pre_damage_pair_sha256=base_pair_sha256(b[:,::-1],a[:,::-1]))]
        self.assertEqual(compare_records(old,new)[0]['reason'],'duplicates_old_pre_damage_pair')

    def test_one_new_cut_cannot_have_multiple_corrosions(self):
        new=[dict(pair_id='clean',pre_damage_pair_sha256='base'),dict(pair_id='gap',pre_damage_pair_sha256='base')]
        self.assertEqual(compare_records([],new)[0]['reason'],'repeated_new_pre_damage_pair')

    def test_source_identity_rejects_before_pixel_check(self):
        old=[dict(pre_damage_pair_sha256='old-mask',source_token_key='same-original')]
        new=[dict(pair_id='other-augmentation',pre_damage_pair_sha256='changed-mask',source_token_key='same-original')]
        self.assertEqual(compare_records(old,new)[0]['reason'],'duplicates_old_source_pair')

    def test_new_pair_passes(self):
        self.assertEqual(compare_records([dict(pre_damage_pair_sha256='old',source_token_key='old-key')],
            [dict(pair_id='new',pre_damage_pair_sha256='new')]),[])

    def test_unknown_derived_pair_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'Unresolved'):
            resolve_masks(row(),{},Path('/pool'),None,None,None)

    def test_clean_gen5_reference_required(self):
        old=row();clean=dict(old,pair_id='old-pair',artifact_path='one.npz')
        sample=SimpleNamespace(pair_id='old-pair',label=True,fragment_a_token='a',fragment_b_token='b',
            mask_a=np.ones((1,4,4)),mask_b=np.ones((1,3,3)))
        loader=lambda _: (sample,dict(physical_damage_applied=False))
        values,method=resolve_masks(old,{'old-pair':clean},Path('/pool'),None,loader,lambda _:None)
        self.assertEqual(method,'clean_gen5_materialization');self.assertEqual(values[1].shape,(3,3))
        with self.assertRaisesRegex(ValueError,'pre-damage'):
            resolve_masks(old,{'old-pair':clean},Path('/pool'),None,
                lambda _: (sample,dict(physical_damage_applied=True)),lambda _:None)

    def test_original_path_traversal_rejected(self):
        old=row()
        for side in 'ab':old['source_row']['fragment_'+side]['model_mask_path']='../../unexpected.png'
        with self.assertRaisesRegex(ValueError,'Unsafe'):
            resolve_masks(old,{},Path('/pool'),None,None,lambda _:None)

    def test_original_provenance_join_bound(self):
        r=dict(row(),pair_id='p');path=Path('/archive/v18.json')
        entry=dict(pair_id='p',source_base_key='/old::old-pair',label=True,source_entry_sha256=digest(r))
        cat=[dict(stage='v18',pair_id='p',source_base_key='/old::old-pair')]
        base=dict(catalog=cat,catalog_sha256=digest(cat),bound_inputs={str(path):'sha'},
                  training_manifests={'v18':dict(path='/admitted',sha256='other')})
        files={str(path):dict(entries=[r]),'/admitted':dict(entries=[entry])}
        read=lambda p,sha:files[str(p)]
        self.assertEqual(len(original_rows(base,{'v18':path},read)),1)
        r['source_pair_id']='different'
        with self.assertRaisesRegex(ValueError,'binding'):
            original_rows(base,{'v18':path},read)


if __name__=='__main__':unittest.main()

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
from . import population
from .contracts import save,sha,SIM_SPLIT


class PopulationTests(unittest.TestCase):
    def fixture(self,root):
        manifest=root/'new_test.json'
        save(manifest,dict(split='test',augmentation_revision=population.REVISION,entries=[]))
        return dict(augmentation_revision=population.REVISION,test={'mixed':dict(path=str(manifest),sha256=sha(manifest),
            pair_count=3000,source_count=28,base_pair_count=3000)})

    def test_new_test_only_uses_bound_new_archives(self):
        with tempfile.TemporaryDirectory() as t:
            contract=self.fixture(Path(t));dataset=Mock();dataset.__len__=Mock(return_value=3000)
            dataset.entries=[dict(pair_id=str(i)) for i in range(3000)]
            with patch.object(population.common,'Dataset',return_value=dataset) as opened,patch.object(population.common,'load_population') as old:
                meta,_,source,actual=population.load_population(SIM_SPLIT,contract,8)
                old.assert_not_called();self.assertIs(actual,dataset);self.assertEqual(len(meta['pairs']),3000)
                opened.assert_called_once_with(contract['test']['mixed']['path'],contract['test']['mixed']['sha256'])
                self.assertEqual(source['augmentation_revision'],population.REVISION)

    def test_wrong_v14_revision_incomplete_or_modified_new_test_rejected(self):
        for case in ('revision','count','changed','old_split'):
            with tempfile.TemporaryDirectory() as t:
                c=self.fixture(Path(t));split=SIM_SPLIT
                if case=='revision':c['augmentation_revision']='v14'
                elif case=='count':c['test']['mixed']['pair_count']=20
                elif case=='changed':Path(c['test']['mixed']['path']).write_text('{}')
                else:split='sim_test_v14'
                with self.assertRaises(ValueError):population.load_population(split,c,8)

    def test_admitted_v17_contract_without_optional_counts_is_valid(self):
        with tempfile.TemporaryDirectory() as t:
            contract=self.fixture(Path(t));spec=contract['test']['mixed']
            del spec['source_count'];del spec['base_pair_count']
            dataset=Mock();dataset.__len__=Mock(return_value=3000)
            dataset.entries=[dict(pair_id=str(i)) for i in range(3000)]
            with patch.object(population.common,'Dataset',return_value=dataset):
                meta,_,source,actual=population.load_population(SIM_SPLIT,contract,8)
            self.assertEqual(len(meta['pairs']),3000);self.assertIs(actual,dataset)
            self.assertIsNone(source['source_count']);self.assertIsNone(source['base_pair_count'])
            self.assertFalse(source['optional_count_metadata_available'])

    def test_gt_helper_translation_opens_no_other_manifest(self):
        with patch.object(population.common,'attach_targets',return_value='synthetic') as attach:
            self.assertEqual(population.attach_targets([],{},SIM_SPLIT,'new_dataset'), 'synthetic')
            attach.assert_called_once_with([],{},'sim_test_v14','new_dataset',None)


if __name__=='__main__':unittest.main()

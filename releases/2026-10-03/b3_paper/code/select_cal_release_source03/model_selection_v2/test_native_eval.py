import unittest
from unittest.mock import patch
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import tempfile,json
from .native_eval import (normalize_native_rows, LOSS_SEMANTICS, evaluate_matcher_select,
                         evaluate_protocol_select, bind_materialized_select, file_sha,
                         check_loaded_runtime_modules)
from .test_protocol import Fixture
from .test_selection import report
from .selection import summarize_report


class NativeRowsTests(unittest.TestCase):
    def setUp(self):
        self.entries=[{'pair_id':'p','label':True},{'pair_id':'n','label':False}]
        self.rows=[dict(pair_id='p',label=True,matcher_batch_loss=.75,numeric_valid=True,
                        has_candidate=True,gt_known=True,layout20=True,candidate_coverage=True),
                   dict(pair_id='n',label=False,matcher_batch_loss=.2,numeric_valid=True,
                        has_candidate=False,gt_known=False,layout20=False,candidate_coverage=False)]

    def test_full_pair_loss_preserved_and_batch_field_removed(self):
        got=normalize_native_rows(self.rows,self.entries)
        self.assertEqual([r['matcher_loss'] for r in got],[.2,.75])
        self.assertTrue(all(r['loss_semantics']==LOSS_SEMANTICS and 'matcher_batch_loss' not in r for r in got))

    def test_failed_positive_kept_in_denominator(self):
        self.rows[0].update(numeric_valid=False,has_candidate=False,layout20=False,candidate_coverage=False)
        got=normalize_native_rows(self.rows,self.entries)
        self.assertEqual(len(got),2);self.assertFalse(got[1]['layout20'])

    def test_missing_duplicate_foreign_rows_rejected(self):
        for rows in (self.rows[:1],[self.rows[0],self.rows[0]], [dict(self.rows[0],pair_id='x'),self.rows[1]]):
            with self.assertRaises(ValueError):normalize_native_rows(rows,self.entries)

    def test_invalid_loss_or_label_rejected(self):
        for changes in ({'matcher_batch_loss':float('nan')},{'matcher_batch_loss':float('inf')},
                        {'matcher_batch_loss':-.1},{'label':False},{'gt_known':False}):
            with self.assertRaises(ValueError):normalize_native_rows([dict(self.rows[0],**changes),self.rows[1]],self.entries)

    def test_cannot_count_invalid_positive_as_layout_success(self):
        with self.assertRaises(ValueError):
            normalize_native_rows([dict(self.rows[0],numeric_valid=False),self.rows[1]],self.entries)

    def test_adapter_forces_batch_one_without_mutating_training_config(self):
        @dataclass(frozen=True)
        class Config:
            microbatch:int=8
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=root/'frozen.py';source.write_text('# fixture\n')
            manifest=root/'select.json'
            manifest.write_text(json.dumps(dict(split='select',real_used=False,test_used=False,entries=self.entries)))
            matcher=SimpleNamespace(parameters=lambda:[],modules=lambda:[],state_dict=lambda:{'weight':'same'})
            model=SimpleNamespace(matcher=matcher);calls=[]
            def evaluate(*args,**kwargs):
                calls.append((args,kwargs));return self.rows
            evaluation=SimpleNamespace(__file__=str(source),evaluate_view=evaluate)
            hashing=SimpleNamespace(__file__=str(source),tree_sha=lambda v:json.dumps(v,sort_keys=True))
            with patch('importlib.import_module',side_effect=[evaluation,hashing]):
                cfg=Config()
                got=evaluate_matcher_select(model,manifest,file_sha(manifest),'cpu',cfg,root,{'frozen.py':file_sha(source)})
            self.assertEqual(cfg.microbatch,8);self.assertEqual(calls[0][0][5].microbatch,1)
            self.assertEqual(calls[0][0][3],'matcher');self.assertEqual(calls[0][1],{'cache':None})
            self.assertEqual(len(got),2)

    def test_adapter_rejects_cal_real_test_before_inference(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'input.json'
            for changes in ({'split':'cal'},{'split':'test'},{'real_used':True},{'test_used':True}):
                d=dict(split='select',real_used=False,test_used=False,entries=self.entries);d.update(changes)
                p.write_text(json.dumps(d))
                with self.assertRaises(ValueError):evaluate_matcher_select(None,p,file_sha(p),'cpu',None,td,{})

    def test_preloaded_foreign_native_dependency_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            name = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.data'
            with patch.dict('sys.modules', {name: SimpleNamespace(__file__='/wrong/source/data.py')}):
                with self.assertRaisesRegex(ValueError, 'outside frozen source'):
                    check_loaded_runtime_modules(td)

    def test_frozen_native_dependency_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            name = 'staging.pairwise_v0_2.fake_runtime_module'
            with patch.dict('sys.modules', {name: SimpleNamespace(__file__=str(Path(td) / 'runtime.py'))}):
                check_loaded_runtime_modules(td)


class MaterializedBindingTests(Fixture):
    def setUp(self):
        super().setUp()
        self.native_entries = []
        for i, entry in enumerate(self.manifests['select']['entries']):
            sample = self.root / ('pair_%02d.npz' % i)
            sample.write_bytes(entry['pair_id'].encode())
            entry['sample_sha256'] = file_sha(sample)
            self.native_entries.append(dict(pair_id=entry['pair_id'], label=entry['label'],
                                            sample_path=str(sample), recipe='clean'))
        self.frozen = self.plan([100])
        self.materialized = self.root / 'native_select.json'
        self.write_materialized()

    def write_materialized(self, **changes):
        data = dict(split='select', real_used=False, test_used=False, entries=self.native_entries)
        data.update(changes)
        self.materialized.write_text(json.dumps(data))
        return file_sha(self.materialized)

    def bind(self):
        return bind_materialized_select(self.frozen, self.materialized, file_sha(self.materialized))

    def native_rows(self):
        rows = [dict(pair_id=e['pair_id'], label=e['label'], numeric_valid=True,
                     has_candidate=True, gt_known=e['label'], layout20=e['label'],
                     candidate_coverage=e['label'], matcher_batch_loss=.5,
                     unused_native_debug='not a selector field') for e in self.native_entries]
        return normalize_native_rows(rows, self.native_entries)

    def test_bridge_projects_exact_selector_rows(self):
        with patch(__package__ + '.native_eval.evaluate_matcher_select', return_value=self.native_rows()):
            rows = evaluate_protocol_select(self.frozen, None, self.materialized,
                file_sha(self.materialized), 'cpu', None, self.root, {})
        result = report(self.frozen, 100); result['rows'] = rows
        self.assertEqual(summarize_report(self.frozen, result)['macro_loss'], .5)
        self.assertTrue(all('unused_native_debug' not in r for r in rows))

    def test_actual_sample_bytes_not_only_declared_hash(self):
        Path(self.native_entries[0]['sample_path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'actual SELECT sample bytes'):
            self.bind()

    def test_native_label_and_recipe_must_be_explicit(self):
        original = dict(self.native_entries[0])
        for change in ({'label': 0}, {'label': not original['label']}, {'recipe': ''}):
            self.native_entries[0] = dict(original, **change); self.write_materialized()
            with self.assertRaises(ValueError):self.bind()
        self.native_entries[0] = original

    def test_relative_sample_path_is_not_reinterpreted(self):
        self.native_entries[0]['sample_path'] = 'pair_00.npz'; self.write_materialized()
        with self.assertRaisesRegex(ValueError, 'absolute sample path'):
            self.bind()

    def test_pair_population_must_equal_normalized_protocol(self):
        self.native_entries.pop(); self.write_materialized()
        with self.assertRaisesRegex(ValueError, 'population differs'):
            self.bind()

    def test_mismatched_optional_stratum_rejected(self):
        self.native_entries[0]['generator'] = 'Gen5'; self.write_materialized()
        with self.assertRaisesRegex(ValueError, 'stratum/hash differs'):
            self.bind()

    def test_sample_mutation_during_inference_blocks_projection(self):
        def mutation(*args, **kwargs):
            Path(self.native_entries[0]['sample_path']).write_bytes(b'changed')
            return self.native_rows()
        with patch(__package__ + '.native_eval.evaluate_matcher_select', side_effect=mutation):
            with self.assertRaisesRegex(ValueError, 'actual SELECT sample bytes'):
                evaluate_protocol_select(self.frozen, None, self.materialized,
                    file_sha(self.materialized), 'cpu', None, self.root, {})


if __name__=='__main__':unittest.main()

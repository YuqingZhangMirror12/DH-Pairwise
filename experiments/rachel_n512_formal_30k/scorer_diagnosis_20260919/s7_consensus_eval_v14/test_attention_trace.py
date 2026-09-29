"""Synthetic CPU hook tests. No experimental data/checkpoint or GPU is used."""
import unittest
import hashlib

import torch
from torch import nn

from .attention_trace import AttentionTrace
from .snapshot import snapshot_prediction
from .audit import validate_arrays
from .frozen import TrainingConfig, registered_protocol
from .view_data import build_case, shared_scales
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.consensus_head import MeasureAttention, ConsensusEvidenceHead
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.compatibility import CompatibilityConfig
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.model import S7Consensus
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.test_evidence import fixture


class AttentionTraceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(428)
        self.head = nn.ModuleDict({'attention':MeasureAttention()}).eval()
        self.att = self.head['attention']
        self.q, self.k = torch.randn(3,96), torch.randn(2,96)
        self.qv = torch.tensor([True,True,False])
        self.kv = torch.ones(2,dtype=torch.bool)
        self.measure = torch.tensor([4.,1.])

    def run_call(self, *, geometry=None, kwargs=False):
        with torch.no_grad(), AttentionTrace(self.head) as trace:
            if kwargs:
                output = self.att(query=self.q,key=self.k,query_valid=self.qv,
                    key_valid=self.kv,key_measure=self.measure,geometry_bias=geometry)
            else:
                output = self.att(self.q,self.k,self.qv,self.kv,self.measure,geometry)
        return output, trace.records[0]

    def test_measures_are_not_matcher_q_and_invalid_queries_are_zero(self):
        with torch.no_grad():
            self.att.q.weight.zero_();self.att.q.bias.zero_()
        _, record = self.run_call()
        torch.testing.assert_close(record['softmax_head_mean'][0,:,0],torch.full((3,),.8))
        torch.testing.assert_close(record['effective_head_mean'][0,2],torch.zeros(2))
        self.assertEqual(record['reconstruction_max_abs'],0.)
        self.assertEqual(record['heads'],4)

    def test_original_return_parameters_rng_are_unchanged(self):
        before = {k:v.clone() for k,v in self.head.state_dict().items()}
        with torch.no_grad():
            expected = self.att(self.q,self.k,self.qv,self.kv,self.measure)
        rng = torch.get_rng_state().clone()
        result,_ = self.run_call(kwargs=True)
        self.assertTrue(torch.equal(result,expected))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for name,value in self.head.state_dict().items():
            self.assertTrue(torch.equal(before[name],value))
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.head.modules()))

    def test_exact_union_bias_and_directional_bias_are_distinct(self):
        with torch.no_grad():
            self.att.q.weight.zero_();self.att.q.bias.zero_()
        mask = torch.tensor([[1.,0.],[1.,1.],[0.,1.]])
        _, union = self.run_call(geometry=mask.clamp_min(1e-30).log())
        _, direction = self.run_call(geometry=torch.tensor([[.2,1.],[1.,1.],[1.,1.]]).log())
        self.assertLess(float(union['effective_head_mean'][0,0,1]),1e-25)
        self.assertGreater(float(direction['effective_head_mean'][0,0,1]),.5)
        torch.testing.assert_close(union['geometry_bias'][0],mask.clamp_min(1e-30).log())

    def test_all_invalid_keys_are_not_invented_evidence(self):
        self.kv.zero_()
        output,record = self.run_call()
        self.assertTrue(torch.equal(output,torch.zeros_like(output)))
        self.assertTrue(torch.equal(record['effective_head_mean'],torch.zeros(1,3,2)))
        torch.testing.assert_close(record['softmax_head_mean'],torch.full((1,3,2),.5))

    def test_empty_keys_and_empty_queries(self):
        self.k = self.k[:0];self.kv = self.kv[:0];self.measure=self.measure[:0]
        _,record = self.run_call()
        self.assertEqual(record['effective_head_mean'].shape,(1,3,0))
        self.q=self.q[:0];self.qv=self.qv[:0]
        _,record = self.run_call()
        self.assertEqual(record['effective_head_mean'].shape,(1,0,0))

    def test_batched_masks_and_measures_are_kept(self):
        self.q=self.q[None].repeat(2,1,1);self.k=self.k[None].repeat(2,1,1)
        self.qv=self.qv[None].repeat(2,1);self.kv=self.kv[None].repeat(2,1)
        self.measure=self.measure[None].repeat(2,1)
        self.kv[1,1]=False
        _,record = self.run_call()
        self.assertEqual(record['effective_head_mean'].shape,(2,3,2))
        self.assertEqual(float(record['effective_head_mean'][1,0,0]),1.)
        self.assertEqual(float(record['effective_head_mean'][1,0,1]),0.)
        self.assertFalse(record['input_was_single'])

    def test_training_or_grad_enabled_capture_rejected(self):
        with self.assertRaisesRegex(ValueError,'eval/no_grad'):
            with AttentionTrace(self.head):pass
        self.head.train()
        with torch.no_grad(),self.assertRaisesRegex(ValueError,'eval/no_grad'):
            with AttentionTrace(self.head):pass

    def test_cleanup_when_forward_raises_and_no_reuse(self):
        trace=AttentionTrace(self.head)
        with torch.no_grad(),self.assertRaisesRegex(RuntimeError,'synthetic'):
            with trace:
                raise RuntimeError('synthetic exception')
        self.assertFalse(trace.handles)
        with torch.no_grad(),self.assertRaisesRegex(ValueError,'single-use'):
            with trace:pass

    def test_preexisting_hooks_not_removed_or_silently_combined(self):
        handle=self.att.register_forward_hook(lambda *x:None)
        try:
            with torch.no_grad(),self.assertRaisesRegex(ValueError,'pre-existing'):
                with AttentionTrace(self.head):pass
            self.assertEqual(len(self.att._forward_hooks),1)
        finally:handle.remove()

    def test_two_pass_real_head_interface_no_second_network_call(self):
        model=S7Consensus(None,CompatibilityConfig(.5,.5,.5,.5,1.,30.),
                          head=ConsensusEvidenceHead(feature_dim=4)).eval()
        pair=fixture()
        with torch.no_grad():
            expected=model.score_pair(pair,capture_diagnostics=True)
            with AttentionTrace(model.head) as trace:
                actual=model.score_pair(pair,capture_diagnostics=True)
        self.assertTrue(torch.equal(expected.score,actual.score))
        self.assertTrue(torch.equal(expected.translation_a_to_b_rc,actual.translation_a_to_b_rc))
        self.assertEqual(len(trace.records),16)
        for name in ('layers.0.self_attention','layers.0.cross_attention',
                     'layers.1.self_attention','layers.1.cross_attention'):
            self.assertEqual([r['call_ordinal'] for r in trace.records if r['module']==name],[0,1,2,3])
        for r in trace.records:
            self.assertEqual(r['effective_head_mean'].shape[0],len(actual.clusters))
            self.assertEqual(r['reconstruction_max_abs'],0.)
            target=(r['query_valid']&r['key_valid'].any(-1)[:,None]).float()
            torch.testing.assert_close(r['effective_head_mean'].sum(-1),target)
            self.assertFalse(r['effective_head_mean'].requires_grad)

    def captured_snapshot(self):
        model=S7Consensus(None,CompatibilityConfig(.5,.5,.5,.5,1.,30.),
                          head=ConsensusEvidenceHead(feature_dim=4)).eval()
        pair=fixture()
        with torch.no_grad(),AttentionTrace(model.head) as trace:
            pred=model.score_pair(pair,capture_diagnostics=True)
        protocol=registered_protocol(TrainingConfig().record())
        meta,arrays=snapshot_prediction('synthetic',pair,pred,threshold=.5,
            provenance=dict(variant=protocol['variant'],evidence_mode=protocol['evidence_mode']),
            attention_trace=trace)
        return meta,arrays,trace,pred,pair

    def test_snapshot_binds_both_stages_and_all_operators(self):
        meta,arrays,_,_,_=self.captured_snapshot()
        self.assertTrue(meta['semantics']['attention_weights_exported'])
        self.assertEqual(validate_arrays(meta,arrays)['status'],'passed')
        view=build_case(meta,arrays,bins=2)
        self.assertTrue(view['semantics']['attention_weights_available'])
        self.assertEqual(shared_scales([view])['scorer_attention']['maximum'],1.)
        for cluster in meta['clusters']:
            for name,stage in cluster['stages'].items():
                self.assertEqual(len(stage['attention']),8)
                a=stage['attention']['1/cross_attention/a']
                b=stage['attention']['1/cross_attention/b']
                self.assertEqual(arrays[a['head_mean']].shape,arrays[b['head_mean']].T.shape)

    def test_readback_rejects_wrong_phase_even_when_matrix_shape_matches(self):
        meta,arrays,_,_,_=self.captured_snapshot()
        meta['clusters'][0]['stages']['final']['attention']['0/self_attention/a']['call_ordinal']=0
        with self.assertRaisesRegex(ValueError,'phase/side'):
            validate_arrays(meta,arrays)

    def test_readback_rejects_renormalized_attention_even_with_hash(self):
        meta,arrays,_,_,_=self.captured_snapshot()
        saved=meta['clusters'][0]['stages']['initial']['attention']['0/cross_attention/a']
        key=saved['head_mean']
        arrays[key]=arrays[key]*.5
        meta['arrays'][key]['sha256']=hashlib.sha256(arrays[key].tobytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,'attention row normalization'):
            validate_arrays(meta,arrays)

    def test_missing_call_is_rejected(self):
        meta,arrays,trace,pred,pair=self.captured_snapshot()
        trace.records.pop()
        with self.assertRaisesRegex(ValueError,'exactly two'):
            snapshot_prediction('synthetic',pair,pred,threshold=.5,provenance=meta['provenance'],attention_trace=trace)

    def test_wrong_contour_identity_rejected_even_with_hash(self):
        meta,arrays,_,_,_=self.captured_snapshot()
        item=meta['clusters'][0]['stages']['initial']['attention']['0/cross_attention/a']
        key=item['query_compact_ids']
        arrays[key]=arrays[key]+1
        meta['arrays'][key]['sha256']=hashlib.sha256(arrays[key].tobytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,'attention contour identity'):
            validate_arrays(meta,arrays)

    def test_modified_packing_is_rejected(self):
        meta,_,trace,pred,pair=self.captured_snapshot()
        trace.records[0]['query_valid'][0,0]=~trace.records[0]['query_valid'][0,0]
        with self.assertRaisesRegex(ValueError,'packing'):
            snapshot_prediction('synthetic',pair,pred,threshold=.5,provenance=meta['provenance'],attention_trace=trace)

    def test_nonfinite_capture_fails_and_cleans_hooks(self):
        with self.assertRaisesRegex(ValueError,'reconstruction differs'):
            self.run_call(geometry=torch.full((3,2),float('nan')))
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.head.modules()))

    def test_no_candidate_exports_explicit_zero_calls(self):
        model=S7Consensus(None,CompatibilityConfig(.5,.5,.5,.5,1.,30.),
                          head=ConsensusEvidenceHead(feature_dim=4)).eval()
        pair=fixture(torch.zeros(5,5))
        with torch.no_grad(),AttentionTrace(model.head) as trace:
            pred=model.score_pair(pair,capture_diagnostics=True)
        meta,arrays=snapshot_prediction('synthetic_empty',pair,pred,threshold=.5,
            provenance=dict(purpose='synthetic CPU test only'),attention_trace=trace)
        self.assertEqual(trace.records,[])
        self.assertEqual(meta['attention_capture']['attention_calls'],0)
        self.assertFalse(meta['semantics']['attention_weights_exported'])
        self.assertEqual(validate_arrays(meta,arrays)['status'],'passed')


if __name__=='__main__':
    unittest.main()

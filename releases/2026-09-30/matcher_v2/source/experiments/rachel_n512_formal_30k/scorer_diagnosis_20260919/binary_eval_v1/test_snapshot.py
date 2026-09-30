from copy import deepcopy
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from .trace import MLPTrace
from .snapshot import snapshot_prediction,write_snapshot,audit_snapshot
from consensus_binary_eval_common.evaluate import state_digest
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.binary_scorer_v1.test_binary import fixture

VARIANT=os.environ.get('BINARY_VERIFY_VARIANT','patch')
class SnapshotTests(unittest.TestCase):
    def capture(self,empty=False,duplicates=False,stats_no_features=False):
        model,pair,proposals,_=fixture(VARIANT);model.eval().requires_grad_(False)
        if empty:proposals=replace(proposals,clusters=())
        if duplicates:proposals=replace(proposals,clusters=tuple(replace(p,edge_ids=p.edge_ids.repeat(3,1)) for p in proposals.clusters))
        if stats_no_features:pair=replace(pair,local_a=None,local_b=None,context_a=None,context_b=None)
        state=state_digest(model);rng=torch.get_rng_state().clone()
        with torch.no_grad():
            baseline=model.score_pair(pair,proposals=proposals,threshold=.3)
            with MLPTrace(model.head) as trace:pred=model.score_pair(pair,proposals=proposals,threshold=.3)
            meta,arrays=snapshot_prediction('synthetic',pair,pred,threshold=.3,
                provenance=dict(variant='binary_'+VARIANT,checkpoint='synthetic fixture, not trained'),trace=trace)
        self.assertEqual(state_digest(model),state);self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        self.assertTrue(torch.equal(baseline.score,pred.score))
        for a,b in zip(baseline.clusters,pred.clusters):self.assertTrue(torch.equal(a.readout.logit,b.readout.logit))
        return meta,arrays
    def test_actual_mlp_same_pass_capture_and_numeric_replay(self):
        meta,arrays=self.capture()
        with tempfile.TemporaryDirectory() as t:
            write_snapshot(Path(t)/'case',meta,arrays);r=audit_snapshot(Path(t)/'case/evidence.json')
            self.assertEqual(r['errors'],[]);self.assertEqual(r['status'],'passed');self.assertEqual(r['clusters'],2)
            self.assertFalse(r['attention_present'])
    def test_exact_duplicate_union_and_empty_candidate(self):
        for options in ({'duplicates':True},{'empty':True}):
            meta,arrays=self.capture(**options)
            with tempfile.TemporaryDirectory() as t:
                write_snapshot(Path(t)/'case',meta,arrays);self.assertEqual(audit_snapshot(Path(t)/'case/evidence.json')['errors'],[])
    def test_stats_trace_never_reads_patch_or_context(self):
        if VARIANT!='stats':return
        meta,arrays=self.capture(stats_no_features=True)
        self.assertFalse(any(k.startswith('pair/local_') or k.startswith('pair/context_') for k in arrays))
        with tempfile.TemporaryDirectory() as t:
            write_snapshot(Path(t)/'case',meta,arrays);self.assertEqual(audit_snapshot(Path(t)/'case/evidence.json')['errors'],[])
    def test_resealed_q_numeric_tampering_detected(self):
        meta,arrays=self.capture();key=meta['clusters'][0]['inputs']['q'];arrays[key][0]*=.5
        meta['arrays'][key]['sha256']=hashlib.sha256(arrays[key].tobytes()).hexdigest()
        with tempfile.TemporaryDirectory() as t:
            write_snapshot(Path(t)/'case',meta,arrays);r=audit_snapshot(Path(t)/'case/evidence.json')
            self.assertEqual(r['status'],'failed');self.assertIn('online Q differs',r['errors'])
    def test_resealed_layer_output_tampering_detected(self):
        meta,arrays=self.capture();key=meta['clusters'][0]['layers'][-1]['output'];arrays[key][0]+=1
        meta['arrays'][key]['sha256']=hashlib.sha256(arrays[key].tobytes()).hexdigest()
        with tempfile.TemporaryDirectory() as t:
            write_snapshot(Path(t)/'case',meta,arrays)
            self.assertEqual(audit_snapshot(Path(t)/'case/evidence.json')['status'],'failed')
    def test_missing_same_forward_trace_rejected(self):
        model,pair,proposals,_=fixture(VARIANT)
        pred=model.score_pair(pair,proposals=proposals)
        with self.assertRaises(ValueError):snapshot_prediction('synthetic',pair,pred,threshold=.5,
            provenance={'variant':'binary_'+VARIANT},trace=MLPTrace(model.head))


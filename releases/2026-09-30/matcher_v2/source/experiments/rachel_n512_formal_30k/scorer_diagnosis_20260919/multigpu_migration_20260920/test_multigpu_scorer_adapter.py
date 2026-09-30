"""CPU numerical contract tests; do not claim CUDA topology equivalence."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch

from . import multigpu_scorer_adapter as runtime


def fixture():
    generator = torch.Generator().manual_seed(29092026)
    batch, count, dim = 16, 12, 96
    a = torch.randn(batch, count, dim, generator=generator)
    b = torch.randn(batch, count, dim, generator=generator)
    valid = torch.ones(batch, count, dtype=torch.bool)
    ids = torch.arange(count)
    indices = torch.stack((ids, ids), -1)[None].expand(batch, -1, -1).clone()
    candidate_valid = torch.ones(batch, count, dtype=torch.bool)
    inliers = torch.zeros_like(candidate_valid)
    inliers[4:, :6] = True
    layout_valid = torch.arange(batch) >= 4
    shift = torch.zeros(batch, 2)
    selection = runtime.CandidateSelection(inliers.clone(), inliers.clone(), layout_valid,
        shift, indices, candidate_valid, inliers, tuple("test" for _ in range(batch)))
    membership = torch.zeros(batch, 5, count, dtype=torch.bool)
    for row in range(4, batch):
        for slot in range(1 + row % 5):
            membership[row, slot, slot:slot+3+(row % 4)] = True
    group_counts = membership.sum(-1)
    present = group_counts > 0
    groups = runtime.stage_cache.StageGroups("edge_multi", membership,
        torch.zeros(batch, 5, 2), present, present & (group_counts >= 3),
        torch.arange(1, 6)[None].expand(batch, -1).clone(),
        torch.zeros(batch, 5, dtype=torch.long), group_counts,
        torch.ones(batch, 5, dtype=torch.int8))
    points = torch.stack((ids.float(), torch.zeros(count)), -1)[None].expand(batch, -1, -1).clone()
    labels = (torch.arange(batch) % 2).float()
    training_valid = torch.ones(batch, dtype=torch.bool)
    training_valid[[1, 7, 14]] = False
    return SimpleNamespace(model_args=(a, b, valid, valid.clone(), selection),
        model_kwargs=dict(groups=groups, candidate_weights=torch.rand(batch, count, generator=generator),
            points_a_rc=points, points_b_rc=points.clone()),
        labels=labels, training_valid=training_valid)


def chunks(adapter):
    def forward(*values):
        results = [adapter(*(v[start:start+4] for v in values)) for start in range(0, 16, 4)]
        return tuple(torch.cat([result[i] for result in results], 0) for i in range(2))
    return forward


class TensorRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_flatten_roundtrip_and_chunk_leading_dimensions(self):
        batch = fixture()
        tensors = runtime.flatten_batch(batch)
        self.assertEqual(len(tensors), 22)
        args, kwargs = runtime.unflatten_tensors(tensors)
        for left, right in zip(args[:4], batch.model_args[:4]):
            self.assertIs(left, right)
        for field in runtime.SELECTION_FIELDS:
            self.assertIs(getattr(args[4], field), getattr(batch.model_args[4], field))
        for field in runtime.GROUP_FIELDS:
            self.assertIs(getattr(kwargs["groups"], field), getattr(batch.model_kwargs["groups"], field))
        self.assertFalse(any(v is batch.labels or v is batch.training_valid for v in tensors))
        local_args, local_kwargs = runtime.unflatten_tensors(tuple(t[4:8] for t in tensors))
        self.assertEqual(len(local_args[4].reasons), 4)
        self.assertEqual(local_args[4].candidate_indices.shape, (4, 12, 2))
        self.assertEqual(local_kwargs["groups"].candidate_inliers.shape, (4, 5, 12))
        with self.assertRaises(ValueError):
            runtime.unflatten_tensors(tensors[:-1])
        with self.assertRaises(ValueError):
            runtime.unflatten_tensors((tensors[0][:4], *tensors[1:]))

    def test_original_and_tensor_forward_match(self):
        batch = fixture()
        base = runtime.original.CandidateStageScorer("edge_multi")
        wrapped = runtime.TensorOnlyEdgeMultiAdapter(base)
        original = base(*batch.model_args, **batch.model_kwargs)
        logits, fallback = wrapped(*runtime.flatten_batch(batch))
        torch.testing.assert_close(logits, original.logit, rtol=0, atol=0)
        torch.testing.assert_close(fallback, original.used_fallback, rtol=0, atol=0)
        self.assertEqual(int(fallback.sum()), 4)
        self.assertIs(wrapped.base, base)
        self.assertTrue(all(not k.startswith("base.") for k in base.state_dict()))

    def test_four_cpu_chunks_preserve_loss_gradients_adam_and_budget(self):
        batch = fixture()
        first = runtime.original.CandidateStageScorer("edge_multi")
        second = deepcopy(first)
        left_opt = runtime.original.create_optimizer(first)
        right_opt = runtime.original.create_optimizer(second)
        tensors = runtime.flatten_batch(batch)
        left = runtime.optimization_step(first, runtime.TensorOnlyEdgeMultiAdapter(first),
            tensors, batch.labels, batch.training_valid, left_opt, capture_gradients=True)
        right = runtime.optimization_step(second, chunks(runtime.TensorOnlyEdgeMultiAdapter(second)),
            tensors, batch.labels, batch.training_valid, right_opt, capture_gradients=True)
        torch.testing.assert_close(left["logits"], right["logits"], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(left["loss"], right["loss"], rtol=1e-6, atol=1e-7)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            left["logits"], batch.labels, reduction="none")
        torch.testing.assert_close(left["loss"], (expected * batch.training_valid).sum() / 16)
        self.assertEqual(len(left["logits"]), 16)
        for name in ("preclip_gradients",):
            difference = runtime.tree_difference(left[name], right[name])
            self.assertEqual(difference["structure_mismatches"], [])
            self.assertEqual(difference["nonfinite_paths"], [])
            self.assertLess(difference["maximum_absolute_error"], 2e-6)
        for a, b in ((first.state_dict(), second.state_dict()), (left_opt.state_dict(), right_opt.state_dict())):
            difference = runtime.tree_difference(a, b)
            self.assertEqual(difference["structure_mismatches"], [])
            self.assertEqual(difference["nonfinite_paths"], [])
            self.assertLess(difference["maximum_absolute_error"], 2e-6)
        self.assertTrue(all(int(state["step"]) == 1 for state in left_opt.state.values()))

    def test_fallback_only_chunk_has_only_scalar_gradient(self):
        batch = fixture()
        base = runtime.original.CandidateStageScorer("edge_multi")
        values = tuple(v[:4] for v in runtime.flatten_batch(batch))
        logits, fallback = runtime.TensorOnlyEdgeMultiAdapter(base)(*values)
        self.assertTrue(fallback.all())
        logits.sum().backward()
        named = dict(base.named_parameters())
        scalar_name = "edge_head.head.no_evidence_logit"
        self.assertEqual(float(named[scalar_name].grad), 4.)
        self.assertTrue(all(p.grad is None for name, p in named.items() if name != scalar_name))

    def test_scope_and_optimizer_guards(self):
        with self.assertRaises(ValueError):
            runtime.TensorOnlyEdgeMultiAdapter(runtime.original.CandidateStageScorer("edge_seed"))
        for ids in ((0,), (0, 1, 2, 2), (0, 1, 2, -1)):
            with self.assertRaises(ValueError):
                runtime.validate_devices(ids)
        first = runtime.original.CandidateStageScorer("edge_multi")
        second = deepcopy(first)
        with self.assertRaises(ValueError):
            runtime.require_optimizer(first, runtime.original.create_optimizer(second))
        batch = fixture()
        with self.assertRaises(ValueError):
            runtime.optimization_step(first, runtime.TensorOnlyEdgeMultiAdapter(first),
                tuple(v[:4] for v in runtime.flatten_batch(batch)), batch.labels[:4],
                batch.training_valid[:4], runtime.original.create_optimizer(first))
        bad = deepcopy(batch)
        bad.model_kwargs["groups"] = replace(bad.model_kwargs["groups"], stage="edge_seed")
        with self.assertRaises(ValueError):
            runtime.flatten_batch(bad)

    def test_difference_does_not_hide_none_or_nonfinite(self):
        difference = runtime.tree_difference({"a": None}, {"a": torch.zeros(1)})
        self.assertEqual(difference["structure_mismatches"], ["root/a"])
        difference = runtime.tree_difference(torch.tensor([float("nan")]), torch.zeros(1))
        self.assertEqual(difference["nonfinite_paths"], ["root"])


if __name__ == "__main__":
    unittest.main()

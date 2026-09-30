"""Disentangle head token visibility from pooling on frozen full-CA features.

These are conditional, off-manifold output contrasts, NOT additive attribution.
All token subsets come from predicted layout endpoints or matched random sets.
"""
import argparse
import types

import numpy as np
import torch

from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919 import token_dilution_probe as original

SCHEMA = "frozen-scorer-pooling-factor/1"


@torch.inference_mode()
def factor_probe(head, a, b, member_a, member_b, *, repeats=16, seed=260919):
    if head.training or a.device.type != "cpu" or b.device.type != "cpu":
        raise ValueError("frozen CPU head required")
    ma, mb = [torch.as_tensor(m, dtype=torch.bool) for m in (member_a, member_b)]
    if ma.shape != a.shape[:1] or mb.shape != b.shape[:1]:
        raise ValueError("mask/token shape mismatch")
    ones = [torch.ones(1, len(x), dtype=torch.bool) for x in (a, b)]
    z = head(a[None], b[None], *ones)[0]
    full_a, full_b = head._decode_pair(a, b)

    def pooled(value, member, local_attentive, local_max):
        av = value[member] if local_attentive else value
        mv = value[member] if local_max else value
        weights = torch.softmax(head.pool_gate(av).squeeze(-1), dim=0)
        return torch.cat(((av * weights[:, None]).sum(0), mv.amax(0)))

    def score_pa_pb(aa, bb):
        symmetric = torch.cat((.5 * (aa + bb), torch.abs(aa - bb)))
        return head.classifier(symmetric).reshape(())

    def row(logit, kind, repeat=0):
        if not torch.isfinite(logit):
            raise ValueError("nonfinite logit")
        return dict(kind=kind, repeat=repeat, logit=float(logit),
                    probability=float(logit.sigmoid()), delta_logit=float(logit - z))

    replay = score_pa_pb(head._pool(full_a), head._pool(full_b))
    error = float((replay - z).abs())
    if error > 2e-5 + 2e-5 * abs(float(z)):
        raise ValueError("factored full forward does not replay original head")
    base = dict(logit=float(z), probability=float(z.sigmoid()),
                count_a=len(a), count_b=len(b), no_evidence=False)
    result = dict(baseline=base, factor_replay_logit_error=error,
                  inlier_count_a=int(ma.sum()), inlier_count_b=int(mb.sum()),
                  interventions=[], attribution_is_additive=False,
                  recomputes_matcher=False, fixed_full_CA_except_local_CA_control=True)
    if not ma.any() or not mb.any():
        result["no_inlier_evidence"] = True
        return result
    masks = [("inliers", 0, ma, mb)]
    rng = np.random.default_rng(seed)
    for repeat in range(repeats):
        subsets = []
        for mask in (ma, mb):
            chosen = torch.zeros_like(mask)
            chosen[rng.choice(len(mask), int(mask.sum()), replace=False)] = True
            subsets.append(chosen)
        masks.append(("random_same_count", repeat, *subsets))
    for kind, repeat, qa, qb in masks:
        # Original full-CA outputs stay fixed: only the learned attentive/max
        # aggregation populations change. Neither head CA nor matcher reruns.
        for local_attentive, local_max, suffix in (
                (True, False, "local_attentive_global_max"),
                (False, True, "global_attentive_local_max"),
                (True, True, "local_both_pool")):
            aa = pooled(full_a, qa, local_attentive, local_max)
            bb = pooled(full_b, qb, local_attentive, local_max)
            measured = row(score_pa_pb(aa, bb), kind + "__full_CA__" + suffix, repeat)
            measured.update(count_a=int(qa.sum()), count_b=int(qb.sum()),
                            retained_inliers_a=int((qa & ma).sum()), retained_inliers_b=int((qb & mb).sum()))
            result["interventions"].append(measured)
        # Matched counterfactual: subset is selected BEFORE head cross-attention.
        # This control also changes contextualization, unlike the three above.
        local_z = head(a[None], b[None], qa[None], qb[None])[0]
        result["interventions"].append(row(local_z, kind + "__local_CA__local_both_pool", repeat))
    return result


def run(args):
    # Reuse exact fixed-source replay/array SHA/checkpoint loader and output
    # contract without mutating its module globals or any model implementation.
    def write_json(path, value):
        if path.name == "protocol.json":
            value = dict(value, fractions=None,
                reused_loader_source_sha256=original.sha(original.__file__),
                controls="full-CA fixed; local attentive/global max, global attentive/local max, local both; local-CA+local-pool; 16 matched random subsets",
                scope="head off-manifold intervention; final full-CA features fixed for pooling variants; matcher never recomputed",
                caveat="conditional non-additive score contrasts, not physical shape damage or evidence that subset-only inference generalizes")
        original.write_json(path, value)
    namespace = dict(original.run.__globals__)
    namespace.update(run_head_probe=factor_probe, SCHEMA=SCHEMA, __file__=__file__, write_json=write_json)
    runner = types.FunctionType(original.run.__code__, namespace)
    return runner(args)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe-root", required=True)
    p.add_argument("--selection-json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--models", nargs="+", default=["s4", "s6", "s6_depth4", "s7"])
    p.add_argument("--repeats", type=int, default=16)
    run(p.parse_args())

"""Per-Pair normalized losses; no Pair-positive pressure on wrong candidates."""
import numpy as np
import torch
from torch.nn import functional as F
from types import SimpleNamespace
from .targets import compact_targets, graph_targets
from .seam_proposals import Candidate


def average(values, zero):
    return torch.stack(values).mean() if values else zero


def segment_mean(value, labels, mask, zero):
    selected = labels[mask]
    if not len(selected):
        return zero
    return torch.stack([value[mask][selected == k].mean() for k in selected.unique()]).mean()


def ot_loss(ot, targets, ga, gb, zero):
    per_pair = []
    for b in range(len(targets['a'])):
        parts = []
        for side, g in (('a', ga), ('b', gb)):
            target = targets[side][b]
            good = (target >= 0) & g.valid[b]
            q = ot.real_transport[b] if side == 'a' else ot.real_transport[b].T
            values = -q.gather(1, target.clamp_min(0)[:, None]).squeeze(1).clamp_min(1e-12).log()
            if good.any():
                parts.append(segment_mean(values, targets['component_'+side][b], good, zero))
            unmatched = (target == -1) & g.valid[b]
            bins = ot.dustbin_col[b] if side == 'a' else ot.dustbin_row[b]
            if unmatched.any():
                parts.append(-bins[unmatched].clamp_min(1e-12).log().mean())
        per_pair.append(average(parts, zero))
    return torch.stack(per_pair)


def candidate_losses(verified, label, gt_known, gt, pose_enabled, zero):
    if not verified.has_candidate or (bool(label) and not bool(gt_known)):
        return zero, zero, zero, 0, 0
    error = (verified.translations.detach()-gt).norm(dim=-1)
    quality = ((error <= 20) & bool(label)).float()
    classification = F.binary_cross_entropy_with_logits(verified.logits, quality)
    good, bad = quality.bool(), ~quality.bool()
    rank = zero
    if good.any():
        wrong = torch.cat((verified.logits[bad], verified.logits.new_zeros(1)))
        rank = F.softplus(.2+wrong[None]-verified.logits[good, None]).mean()
    pose = zero
    eligible = (error <= 40) & bool(label) & bool(pose_enabled)
    if eligible.any():
        pose = F.smooth_l1_loss(verified.translations[eligible]/32, gt.expand(int(eligible.sum()), -1)/32)
    return classification, rank, pose, int(good.sum()), len(quality)


def compute_loss(model, output, batch, stage, teacher_weight=0.):
    t = compact_targets(batch, output)
    zero = output.s1.sum()*0
    ot = .25*ot_loss(output.ot0, t, output.ga, output.gb, zero)+.75*ot_loss(output.ot1, t, output.ga, output.gb, zero)
    support, link, candidate, rank, pose = [], [], [], [], []
    teacher_values, native_values = [], []
    counts = dict(supervised_edges=0, correct_edges=0, supervised_links=0, gap_links=0,
                  candidate_positive=0, candidate_total=0, teacher_pairs=0, native_coverage_miss=0)
    for b, record in enumerate(output.records):
        correct, known, relation, stop = graph_targets(record, t, b)
        terms = []
        if correct.any():
            values = F.binary_cross_entropy_with_logits(record.support, correct.float(), reduction='none')
            terms.append(segment_mean(values, t['component_a'][b, record.edges[:, 0]], correct, zero))
        wrong = known & ~correct
        if wrong.any():
            terms.append(F.softplus(record.support[wrong]).mean())
        support.append(average(terms, zero))
        terms = []
        # Equal weight per present relation class within a Pair.
        for cls in (0, 1, 2):
            mask = relation == cls
            if mask.any():
                terms.append(F.cross_entropy(record.links[mask], relation[mask]))
        if (stop >= 0).any():
            terms.append(F.binary_cross_entropy_with_logits(record.stop[stop >= 0], stop[stop >= 0].float()))
        link.append(average(terms, zero))
        counts['supervised_edges'] += int(known.sum())
        counts['correct_edges'] += int(correct.sum())
        counts['supervised_links'] += int((relation >= 0).sum())
        counts['gap_links'] += int((relation == 1).sum())
        if stage == 'B':
            values = candidate_losses(output.verified[b], batch['labels'][b], batch['translation_valid'][b],
                batch['translation_a_to_b_rc'][b], batch['pose_enabled'][b], zero)
            c, r, p, positives, total = values
            native_values.append(c.detach())
            teacher_value=zero.detach()
            counts['candidate_positive'] += positives; counts['candidate_total'] += total
            counts['native_coverage_miss'] += int(bool(batch['labels'][b]) and positives == 0)
            if teacher_weight > 0 and batch['labels'][b] and batch['translation_valid'][b]:
                # Separate auxiliary call. No teacher enters output.candidates,
                # production forward, validation, ranking metrics or decoder.
                ia = torch.nonzero(t['a'][b] >= 0).flatten()
                if len(ia):
                    ib = t['a'][b, ia]
                    teacher_record = SimpleNamespace(edges=torch.stack((ia, ib), -1))
                    perturbation = torch.randn_like(batch['translation_a_to_b_rc'][b])*4
                    tc = Candidate(np.arange(len(ia)), (batch['translation_a_to_b_rc'][b]+perturbation).detach().cpu().numpy(), 0., True)
                    result = model.verify_candidates(output, b, [tc], teacher_record)
                    cv, rv, pv, _, _ = candidate_losses(result, batch['labels'][b], batch['translation_valid'][b],
                        batch['translation_a_to_b_rc'][b], batch['pose_enabled'][b], zero)
                    c, r, p = c+teacher_weight*cv, r+teacher_weight*rv, p+teacher_weight*pv
                    teacher_value=cv.detach()
                    counts['teacher_pairs'] += 1
            candidate.append(c); rank.append(r); pose.append(p)
            teacher_values.append(teacher_value)
    balance = torch.stack([torch.relu(o.diagnostics.row_residual_max-.001)+
                            torch.relu(o.diagnostics.col_residual_max-.001) for o in (output.ot0, output.ot1)]).mean()
    components = dict(ot=ot.mean(), support=average(support, zero), link=average(link, zero),
        candidate=average(candidate, zero), rank=average(rank, zero), pose=average(pose, zero), balance=balance)
    weights = dict(ot=.5, support=.5, link=.25, candidate=1., rank=.5, pose=.25, balance=.05)
    total = sum(components[k]*w for k, w in weights.items())
    components['native_candidate_diagnostic']=average(native_values,zero.detach())
    components['teacher_candidate_diagnostic']=average(teacher_values,zero.detach())
    return total, components, counts

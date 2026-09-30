"""Split original sources before score-blind within-fold negative sampling."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

ARMS = ("reference512_h4", "cap128_h4", "cap256_h4", "cap512_h8",
        "gcn_pairing_h4", "gcn_shredding_h4", "joint_D_h4", "stable_h4")
SEED = 260921
K = 5
PREPARED = {
    "real": "/root/autodl-tmp/rachel_layout_v2_20260906_001/real_preparation/prepared",
    "ood": "/root/autodl-tmp/turufan_ood_pairwise_20260912_001/prepared",
}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024*1024), b""):
            h.update(data)
    return h.hexdigest()


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


class Union:
    def __init__(self):
        self.parents = {}

    def find(self, key):
        self.parents.setdefault(key, key)
        if self.parents[key] != key:
            self.parents[key] = self.find(self.parents[key])
        return self.parents[key]

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parents[max(a, b)] = min(a, b)


def manuscript(prefix):
    return re.sub(r"(?:[_\-\s]?(?:recto|verso)(?:total)?|[_\-\s]?seite\d+)$",
                  "", prefix.lower()).strip("_- ")


def source_groups(split, meta):
    by_fragment, hash_owner, aliases = {}, {}, []
    union = Union()
    def record(fragment, group, digest):
        by_fragment[fragment] = group
        union.find(group)
        if digest:
            if digest in hash_owner:
                union.join(group, hash_owner[digest])
            hash_owner[digest] = group
    if split == "real":
        original = read(meta["source"]["real_manifest"])
        relevant = set(meta["fragment_ids"])
        for case in original["cases"]:
            for frag in case["fragments"]:
                fid = case["case_uid"] + "/fragment/" + str(frag["fragment_id"])
                if fid in relevant:
                    record(fid, case["case_uid"], frag.get("alpha_mask_sha256"))
    else:
        for frag in meta["fragments"]:
            group = manuscript(frag["prefix"])
            record(frag["fragment_id"], group, frag.get("source_alpha_ge128_sha256"))
            aliases.append({"prefix": frag["prefix"], "source_group": group})
    if set(by_fragment) != set(meta["fragment_ids"]):
        raise ValueError("every prepared fragment must resolve to an original source")
    result = {f: union.find(g) for f, g in by_fragment.items()}
    return result, sorted({(x["prefix"], union.find(x["source_group"])) for x in aliases})


def assign_folds(groups, retained, seed):
    """Balance retained labels and fragment totals, independent of model scores."""
    sizes, pos, neg = Counter(groups.values()), Counter(), Counter()
    for row in retained:
        g = groups[row["fragment_a_id"]]
        if g != groups[row["fragment_b_id"]]:
            raise ValueError("retained genuine pairs must stay within source group")
        (pos if row["label"] else neg)[g] += 1
    rng = random.Random(seed)
    order = list(sizes)
    rng.shuffle(order)
    order.sort(key=lambda g: (-pos[g], -neg[g], -sizes[g]))
    totals = [[0, 0, 0] for _ in range(K)]
    assignment = {}
    for g in order:
        choices = list(range(K)); rng.shuffle(choices)
        # Positive-bearing groups balance positives first. Groups containing no
        # positive cannot improve that dimension, so balance negatives/size.
        key = ((lambda k: tuple(totals[k])) if pos[g] else
               (lambda k: (totals[k][1], totals[k][2])) if neg[g] else
               (lambda k: (totals[k][2], totals[k][0]+totals[k][1])))
        fold = min(choices, key=key)
        assignment[g] = fold
        for i, amount in enumerate((pos[g], neg[g], sizes[g])):
            totals[fold][i] += amount
    if any(t[0] == 0 for t in totals):
        raise ValueError("each test fold must include positives")
    return assignment


def negative_pairs(split, groups, folds, retained, desired, seed):
    by_fold = defaultdict(lambda: defaultdict(list))
    for f, g in sorted(groups.items()):
        by_fold[folds[g]][g].append(f)
    positives = Counter(r["fold"] for r in retained if r["label"])
    strict = Counter(r["fold"] for r in retained if not r["label"])
    total_negative = desired + sum(strict.values())
    quotas_float = [total_negative * positives[k] / sum(positives.values()) for k in range(K)]
    quotas = [int(x) for x in quotas_float]
    for k in sorted(range(K), key=lambda k: (quotas_float[k]-quotas[k], -k), reverse=True)[:total_negative-sum(quotas)]:
        quotas[k] += 1
    rng = random.Random(seed)
    output, seen = [], set()
    for fold in range(K):
        source_names = sorted(by_fold[fold])
        count = quotas[fold] - strict[fold]
        if count < 0 or len(source_names) < 2:
            raise ValueError("negative budget or fold source diversity insufficient")
        attempts = 0
        while sum(r["fold"] == fold for r in output) < count:
            attempts += 1
            if attempts > count*1000:
                raise RuntimeError("negative sampler exhausted")
            ga, gb = rng.sample(source_names, 2)
            a, b = sorted((rng.choice(by_fold[fold][ga]), rng.choice(by_fold[fold][gb])))
            if (a,b) in seen:
                continue
            seen.add((a,b))
            digest = hashlib.sha256((split+"\n"+a+"\n"+b).encode()).hexdigest()
            output.append(dict(pair_id="real-threshold-cv-v1/"+split+"/negative/"+digest,
                fragment_a_id=a, fragment_b_id=b, label=False, fold=fold,
                label_source="user_authorized_cross_source_negative", reuse_original_score=False))
    if len(output) != desired:
        raise AssertionError("negative count changed")
    return output


def prepare(base, output):
    base, output = Path(base), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frozen = {a: read(base/"training"/a/"freeze.json") for a in ARMS}
    if any(f["status"] != "complete" or f["real_ood_used_for_fit"] for f in frozen.values()):
        raise ValueError("models must have completed prior synthetic-only training")
    save(output/"model_freezes.json", frozen)
    for split in ("real", "ood"):
        meta = read(Path(PREPARED[split])/"manifest.json")
        reference = rows(base/"evaluation/reference512_h4"/split/"pair_results.jsonl")
        retained = []
        for r in reference:
            keep = split == "ood" or (r["label"] and r["review_status"] == "keep") or (not r["label"] and r["strict_member"])
            if keep:
                retained.append(dict(pair_id=r["pair_id"], fragment_a_id=r["fragment_a"],
                    fragment_b_id=r["fragment_b"], label=r["label"],
                    label_source="existing_positive" if r["label"] else "existing_within_case_negative",
                    reuse_original_score=True))
        expected = (295, 39) if split == "real" else (301, 0)
        assert (sum(r["label"] for r in retained), sum(not r["label"] for r in retained)) == expected
        group, aliases = source_groups(split, meta)
        assignment = assign_folds(group, retained, SEED + (split == "ood"))
        for r in retained:
            r["fold"] = assignment[group[r["fragment_a_id"]]]
        negatives = negative_pairs(split, group, assignment, retained, 469 if split == "real" else 301, SEED+11+(split=="ood"))
        population = sorted(retained+negatives, key=lambda r: r["pair_id"])
        checks = []
        for k in range(K):
            cal = [r for r in population if r["fold"] != k]
            test = [r for r in population if r["fold"] == k]
            def sources(rr):return {group[r[s]] for r in rr for s in ("fragment_a_id","fragment_b_id")}
            def fragments(rr):return {r[s] for r in rr for s in ("fragment_a_id","fragment_b_id")}
            assert not sources(cal) & sources(test)
            assert not fragments(cal) & fragments(test)
            checks.append(dict(fold=k, calibration_count=len(cal), test_count=len(test),
                calibration_positive=sum(r["label"] for r in cal), test_positive=sum(r["label"] for r in test),
                shared_sources=0, shared_fragments=0))
        save(output/split/"manifest.json", dict(schema="real-threshold-cv-v1/1",split=split,
            prepared=PREPARED[split], seed=SEED, folds=K, fragment_ids=meta["fragment_ids"], pairs=population,
            fragment_source_group=group, source_folds=assignment, manuscript_aliases=aliases,
            original_manifest_sha256=hash_file(Path(PREPARED[split])/"manifest.json"),
            negative_label_assumption="Different recorded sources are negative per user instruction; not manual geometry verification.",
            fold_checks=checks, score_blind_negative_sampling=True, normalization="unchanged original-case prepared inputs"))
        print(json.dumps(dict(split=split,pairs=len(population),positive=expected[0],negative=len(population)-expected[0],
            source_groups=len(set(group.values())),fold_checks=checks)),flush=True)
    save(output/"protocol.json", dict(status="prepared_before_new_scores", base=str(base),
        arms=ARMS, seed=SEED, k=K, policies=["max_f1","recall_95"], threshold_training_only=True,
        weights_frozen=True, model_selection_performed=False, historical_real_design_exposure=True,
        negative_sampling_independent_of_predictions=True, no_html=True))


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--base",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();prepare(a.base,a.output)

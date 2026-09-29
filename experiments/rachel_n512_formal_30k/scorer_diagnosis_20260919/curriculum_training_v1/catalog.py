"""Separate actual Matcher input identity from GT/supervision identity.

Legacy audit 'model_input_sha256' includes several GT fields. Preserve that
receipt, but compute the six actual Matcher input fields independently before
using exact-input deduplication in a curriculum training catalog.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .exposure import SampleRef, STAGES, canonical_catalog, is_sha

MATCHER_INPUTS = ('mask_a', 'mask_b', 'points_rc_a', 'points_rc_b',
                  'contour_valid_a', 'contour_valid_b')
TARGETS = ('target_a', 'target_b', 'translation_a_to_b_rc', 'translation_valid', 'label')
LEGACY_FIELDS = ('mask_a', 'mask_b', 'points_rc_a', 'points_rc_b',
                 'target_a', 'target_b', 'translation_a_to_b_rc')


def tensor_digest(sample, names):
    h = hashlib.sha256()
    for name in names:
        value = np.asarray(getattr(sample, name))
        if not np.isfinite(value).all():
            raise ValueError('nonfinite materialized tensor: ' + name)
        header = json.dumps([name, value.dtype.str, list(value.shape)], separators=(',', ':')).encode()
        h.update(len(header).to_bytes(8, 'little')); h.update(header)
        h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def legacy_numerical_digest(sample):
    h = hashlib.sha256()
    for name in LEGACY_FIELDS:
        h.update(name.encode()); h.update(getattr(sample, name).tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class InspectedSample:
    ref: SampleRef
    target_sha256: str
    legacy_audit_sha256: str


def inspect_sample(stage, row, audit, loader=None):
    if stage not in STAGES:
        raise ValueError('unsupported curriculum stage')
    if row.get('source_row', {}).get('split') != 'train':
        raise ValueError('only existing TRAIN samples may enter the curriculum')
    name = row['pair_id']
    if (audit.get('status') != 'passed' or audit.get('id') != name
            or row.get('id', name) != name or type(row['label']) is not bool):
        raise ValueError('sample/audit identity mismatch')
    if audit.get('sample_sha256') != row['sample_sha256'] or not is_sha(row['sample_sha256']):
        raise ValueError('sample/audit file binding differs')
    path = Path(row.get('sample_path') or row['artifact_path'])
    if not path.is_absolute() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sample_sha256']:
        raise ValueError('bound absolute sample file differs')
    key = str(row['source_root']) + '::' + str(row['source_pair_id'])
    if row.get('source_base_key', key) != key:
        raise ValueError('original pair identity is inconsistent')
    if loader is None:
        from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
        loader = load_sample
    sample, _ = loader(path)
    if sample.pair_id != name or float(sample.label) not in (0., 1.) or bool(sample.label) != row['label']:
        raise ValueError('actual loader identity or label differs')
    legacy = legacy_numerical_digest(sample)
    if legacy != audit.get('model_input_sha256'):
        raise ValueError('legacy audited numerical content changed')
    inputs = tensor_digest(sample, MATCHER_INPUTS)
    targets = tensor_digest(sample, TARGETS)
    ref = SampleRef(stage, name, key, inputs, str(path.resolve()), row['sample_sha256'], row['label'])
    return InspectedSample(ref, targets, legacy)


def deduplicate_basic(basic, hard):
    """Prefer fixed hard rows; log exact basic duplicates, never change labels.

    Cross-stage original base reuse is a stale exclusion error, not an excuse to
    silently trim the input plan. The basic reference must exclude it beforehand.
    """
    basic = tuple(basic); hard = tuple(hard)
    if (any(x.ref.stage != 'v17_filtered' for x in basic)
            or any(x.ref.stage not in ('v17.5', 'v18') for x in hard)):
        raise ValueError('wrong catalog partition')
    hard_bases = {x.ref.source_base_key for x in hard}
    if any(x.ref.source_base_key in hard_bases for x in basic):
        raise ValueError('basic reference did not exclude later-stage original pairs')
    seen = {}; kept = []; excluded = []
    for item in sorted(hard, key=lambda x: (x.ref.stage, x.ref.pair_id)):
        key = item.ref.model_input_sha256
        if key in seen:
            raise ValueError('hard release contains duplicated actual Matcher inputs')
        seen[key] = item
    for item in sorted(basic, key=lambda x: x.ref.pair_id):
        previous = seen.get(item.ref.model_input_sha256)
        if previous is not None:
            if item.target_sha256 != previous.target_sha256 or item.ref.label != previous.ref.label:
                raise ValueError('identical Matcher inputs have conflicting supervision; do not select a label')
            excluded.append(dict(pair_id=item.ref.pair_id, keep_pair_id=previous.ref.pair_id,
                reason='identical six Matcher inputs and supervision', sample_sha256=item.ref.sample_sha256,
                input_sha256=item.ref.model_input_sha256))
        else:
            kept.append(item); seen[item.ref.model_input_sha256] = item
    result = canonical_catalog([x.ref for x in kept + list(hard)])
    return result, dict(basic_before=len(basic), basic_after=len(kept),
        basic_positive_after=sum(x.ref.label for x in kept),
        basic_negative_after=sum(not x.ref.label for x in kept),
        discarded_exact_duplicates=excluded, hard_rows_unchanged=len(hard),
        generated_new_data=False, pixel_admission_replaced=False,
        sampling_balance='both classes retained; balance exposures in each global batch')

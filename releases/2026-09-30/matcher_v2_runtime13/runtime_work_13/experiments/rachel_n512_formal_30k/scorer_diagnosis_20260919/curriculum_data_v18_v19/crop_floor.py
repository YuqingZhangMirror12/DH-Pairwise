"""20% is a CROP-only gate; neither erosion nor light is measured here.

The numerator is surviving original bilateral GT-supported outer arc. New cut
edges and links across removed intervals never count. The denominator is the
full post-crop perimeter of the smaller-by-pixel-area fragment.
"""
import numpy as np
from ..s7_balanced_v2.partial_v14 import retained_support

MINIMUM = .20


def measure(base, cropped, bands=None):
    result, proof = retained_support(base, cropped, bands)
    result = dict(result, stage='crop_only_before_primary_erosion_and_final_light')
    return result, proof


def check(before, after, changed):
    originally_short = before['common_over_smaller_perimeter'] < MINIMUM
    if originally_short and changed:
        raise ValueError('original common seam below20%; structural crop forbidden')
    if not originally_short and after['common_over_smaller_perimeter'] < MINIMUM:
        raise ValueError('crop-only common seam below20% smaller full perimeter')
    return dict(minimum=MINIMUM, stage='after_all_structural_cuts_before_any_erosion',
        before=before, after=after, originally_short=originally_short,
        changed=bool(changed), passed=True,
        erosion_and_light_subject_to_this_gate=False,
        exception='originally_below20_keep_crop_pixels_unchanged' if originally_short else None)


def evaluate(base, cropped, bands=None):
    before, _ = measure(base, base, bands)
    after, _ = measure(base, cropped, bands)
    changed = any(not np.array_equal(getattr(base, 'mask_'+s), getattr(cropped, 'mask_'+s)) for s in 'ab')
    return check(before, after, changed)

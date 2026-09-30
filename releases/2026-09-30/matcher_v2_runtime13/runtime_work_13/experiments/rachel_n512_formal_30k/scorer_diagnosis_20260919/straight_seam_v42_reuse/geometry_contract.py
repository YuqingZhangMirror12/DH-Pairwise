"""Fixed per-pair MD bounds; median targets are not per-sample rejection rules."""
import math

CONTRACT = dict(schema='v42-strict-geometry/1', positive_bend_max_px=15.,
                j_rectangularity_upper_exclusive=.9, r_rectangularity_min=.9,
                unchanged_negative_samples=True, model_selection=False,
                test_policy='Same predeclared generator-quality predicate; no TEST aggregate calibration or inference.')


def violations(kind, measured):
    if kind not in 'MJR' or len(kind) != 1:
        raise ValueError('Unknown seam type')
    if measured is None or measured.get('seam') is None:
        return ['unmeasurable_seam']
    bend = measured['seam']['bend_range']
    rect = measured['smaller_rectangularity']
    if not all(math.isfinite(value) for value in (bend, rect)):
        raise ValueError('Nonfinite geometry; do not silently resample a broken metric')
    result = []
    if bend > CONTRACT['positive_bend_max_px']:
        result.append('bend_gt15')
    if kind == 'J' and rect >= CONTRACT['j_rectangularity_upper_exclusive']:
        result.append('J_rect_ge0.9')
    if kind == 'R' and rect < CONTRACT['r_rectangularity_min']:
        result.append('R_rect_lt0.9')
    return result

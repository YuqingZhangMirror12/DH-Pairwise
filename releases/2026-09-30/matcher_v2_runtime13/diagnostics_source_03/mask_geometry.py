"""Self-contained copies of the existing pure D10 mask definitions.

The training snapshot deliberately does not include data-generation code. This
evaluation companion therefore carries the two needed functions, rather than
changing that live snapshot or importing a reference module's top-level runs.
"""
import cv2
import numpy as np
from scipy import ndimage


def seam_profile(ma, mb, t, *, delta=6):
    edge_a = ma & ~ndimage.binary_erosion(ma)
    edge_b = mb & ~ndimage.binary_erosion(mb)
    dist_b = ndimage.distance_transform_edt(~edge_b)
    r, c = np.nonzero(edge_a)
    mapped = np.rint(np.column_stack([r, c]) + t).astype(int)
    valid = ((mapped >= 0) & (mapped < 800)).all(1)
    distances = np.full(len(r), np.inf)
    distances[valid] = dist_b[tuple(mapped[valid].T)]
    keep = distances <= delta
    points = np.column_stack([r[keep], c[keep]]).astype(float)
    if len(points) < 20:
        return None
    centred = points - points.mean(0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    along, residual = (centred @ vt.T).T
    order = np.argsort(along)
    grid = np.arange(along.min(), along.max() + 1., 1.)
    profile = np.interp(grid, along[order], residual[order])
    smooth = ndimage.gaussian_filter1d(profile, 15, mode='nearest')
    return dict(extent_px=float(np.ptp(along)), seam_px=int(keep.sum()),
                bend_range=float(np.ptp(smooth)))


def rectangularity(mask):
    contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError('cannot classify an empty fragment')
    contour = max(contours, key=cv2.contourArea)
    _, (width, height), _ = cv2.minAreaRect(contour)
    return float(mask.sum() / max(width * height, 1.))

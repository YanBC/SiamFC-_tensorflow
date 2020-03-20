# -*- coding: utf-8 -*-
from typing import Dict

import numpy as np

max_area_rate = 0.6
min_area_rate = 0.001
max_ratio = 10


def is_reasonable(im: np.array, bbox) -> bool:
    r""" 
    Filter too small,too large objects and objects with extreme ratio
    No input check. Assume that all imput (im, bbox) are valid object

    Arguments
    ---------
    im: np.array
        image, formate=(H, W, C)
    bbox: np.array or indexable object
        bounding box annotation
    """
    eps = 1e-6
    im_area = im.shape[0] * im.shape[1]
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    bbox_area_rate = bbox_area / im_area
    bbox_ratio = (bbox[3] - bbox[1] + 1) / max(bbox[2] - bbox[0] + 1, eps)
    # valid trainng box condition
    conds = [(min_area_rate < bbox_area_rate,
              bbox_area_rate < max_area_rate),
             max(bbox_ratio, 1.0 / max(bbox_ratio, eps)) < max_ratio]
    # if not all conditions are satisfied, filter the sample
    reasonable_flag = all(conds)

    return reasonable_flag

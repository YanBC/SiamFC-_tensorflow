import cv2
import numpy as np

import sys
sys.path.append('./')
from utils.bbox import cxywh2xyxy



def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans=(0, 0, 0)):
    r"""
    Get subwindow via cv2.warpAffine

    Arguments
    ---------
    im: numpy.array
        original image, (H, W, C)
    pos: numpy.array
        subwindow position
    model_sz: int
        output size
    original_sz: int
        subwindow range on the original image
    avg_chans: tuple
        average values per channel

    Returns
    -------
    numpy.array
        image patch within _original_sz_ in _im_ and  resized to _model_sz_, padded by _avg_chans_
        (model_sz, model_sz, 3)
    """
    crop_cxywh = np.concatenate(
        [np.array(pos), np.array((original_sz, original_sz))], axis=-1)
    crop_xyxy = cxywh2xyxy(crop_cxywh)
    # warpAffine transform matrix
    M_13 = crop_xyxy[0]
    M_23 = crop_xyxy[1]
    M_11 = (crop_xyxy[2] - M_13) / (model_sz - 1)
    M_22 = (crop_xyxy[3] - M_23) / (model_sz - 1)
    mat2x3 = np.array([
        M_11,
        0,
        M_13,
        0,
        M_22,
        M_23,
    ]).reshape(2, 3)
    im_patch = cv2.warpAffine(im,
                              mat2x3, (model_sz, model_sz),
                              flags=(cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=tuple(map(int, avg_chans)))
    return im_patch


def get_crop(im, target_pos, target_sz, z_size, x_size=None, avg_chans=(0, 0, 0), context_amount=0.5, func_get_subwindow=get_subwindow_tracking):
    r"""
    Get cropped patch for tracking

    Arguments
    ---------
    im: numpy.array
        input image
    target_pos: list-like or numpy.array
        position, (x, y)
    target_sz: list-like or numpy.array
        size, (w, h)
    z_size: int
        template patch size
    x_size: int
        search patch size, None in case of template (z_size == x_size)
    avg_chans: tuple
        channel average values, (B, G, R)
    context_amount: float
        context to be includede in template, set to 0.5 by convention
    func_get_subwindow: function object
        function used to perform cropping & resizing

    Returns
    -------
        cropped & resized image, (x_size, x_size, 3) if x_size provided, (z_size, z_size, 3) otherwise
    """
    wc = target_sz[0] + context_amount * sum(target_sz)
    hc = target_sz[1] + context_amount * sum(target_sz)
    s_crop = np.sqrt(wc * hc)
    scale = z_size / s_crop

    # im_pad = x_pad / scale
    if x_size is None:
        x_size = z_size
    s_crop = x_size / scale

    # extract scaled crops for search region x at previous target position
    im_crop = func_get_subwindow(im, target_pos, x_size, round(s_crop),
                                 avg_chans)

    return im_crop, scale




if __name__ == '__main__':
    import os

    imagePath = './temp/images_and_videos/lab.png'
    x_min, y_min, x_max, y_max = 825, 724, 948, 853

    target_pos = (x_min+x_max)//2, (y_min+y_max)//2
    target_sz = x_max - x_min, y_max - y_min
    z_size = 303
    x_size = 127

    image = cv2.imread(imagePath)
    crop_object, scale_object = get_crop(image, target_pos, target_sz, z_size)
    crop_image, scale_image = get_crop(image, target_pos, target_sz, z_size, x_size)

    crop_object_name = 'croped_object.jpg'
    crop_image_name = 'croped_image.jpg'
    desDir = './temp/images_and_videos'
    cv2.imwrite(os.path.join(desDir, crop_object_name), crop_object)
    cv2.imwrite(os.path.join(desDir, crop_image_name), crop_image)


    target = image[y_min:y_max, x_min:x_max, :]
    cv2.imwrite('show.jpg', target)


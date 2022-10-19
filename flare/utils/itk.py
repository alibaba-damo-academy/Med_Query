# Copyright (c) DAMO Health

import numpy as np
import SimpleITK as sitk
from flare.utils.frame import voxel_to_world
from typing import Optional, Tuple


def normalize_sitk_im(
    sitk_im: sitk.Image,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    clip: bool = True,
    min_out_is_zero: bool = False,
) -> sitk.Image:
    """
    normalize a simpleitk image, if either min_value or max_value is not given, will use the
    intensities at 1 and 99 percentiles to adaptively normalize the image

    :param sitk_im: the input image to be normalized
    :param min_value: min value to use
    :param max_value: max value to use
    :param clip: whether clip values out of the range, defaults to True
    :return: the normalized image
    """
    pixel_type = sitk_im.GetPixelID()
    if pixel_type not in [sitk.sitkFloat32, sitk.sitkFloat64]:
        raise TypeError("the dtype of the image to be normalized should be float32 or float64!")

    im_np = sitk.GetArrayFromImage(sitk_im)
    if min_value is None:
        min_value = np.percentile(im_np, 1)
    if max_value is None:
        max_value = np.percentile(im_np, 99)
    if clip:
        sitk_im = sitk.Clamp(sitk_im, lowerBound=min_value, upperBound=max_value)

    shift = -min_value
    scale = 1 / (max_value - min_value)
    if min_out_is_zero:
        sitk_im = sitk.ShiftScale(sitk_im, shift, scale)
    else:
        shift -= (max_value - min_value) / 2
        scale *= 2
        sitk_im = sitk.ShiftScale(sitk_im, shift, scale)

    return sitk_im


def mask_bbox(mask: sitk.Image, mask_value: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    get the bounding box of foreground label(s) in world coordinate system,
    if foreground label not found in the mask, will return (None, None)

    :param mask: the input sitk image object for mask
    :param mask_value: foreground label value, will use all positive labels if not specified
    :return: the bounding box in world coordinate system (wx, wy, wz)
    """
    mask_np = sitk.GetArrayFromImage(mask)
    if mask_value is not None:
        vz, vy, vx = np.where(mask_np == mask_value)
    else:
        vz, vy, vx = np.where(mask_np > 0)

    min_corner, max_corner = None, None
    if vz.size > 0:
        start = [vx.min(), vy.min(), vz.min()]
        end = [vx.max(), vy.max(), vz.max()]

        coords = list(zip(start, end))
        corners = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    voxel_coord = [coords[0][i], coords[1][j], coords[2][k]]
                    corners.append(voxel_to_world(mask, voxel_coord))
        corners = np.vstack(corners)
        min_corner = corners.min(axis=0)
        max_corner = corners.max(axis=0)
    return min_corner, max_corner

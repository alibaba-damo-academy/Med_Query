# Copyright (c) DAMO Health

import numpy as np
import SimpleITK as sitk
import torch
from typing import Iterable, Optional, Tuple, Union

from med_query.utils.frame import voxel_to_world


def crop(sitk_im: sitk.Image, sp: Iterable[int], ep: Iterable[int]) -> sitk.Image:
    """
    crop a sitk image with voxel coordinates.

    :param sitk_im: a sitk image object
    :param sp: a vector with 3 int values, inclusive
    :param ep: a vector with 3 int values, exclusive
    :return: the croped sitk image
    """
    size = np.array(ep) - np.array(sp)
    index = np.array(sp, np.int32)
    im_crop = sitk.RegionOfInterest(sitk_im, size=size.tolist(), index=index.tolist())
    return im_crop


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


def cal_dsc(
    result_mask: Union[sitk.Image, np.ndarray],
    label_mask: Union[sitk.Image, np.ndarray],
    num_classes: Optional[int] = None,
    use_gpu=False,
):
    """
    calculates the dice coefficients for multi-label masks, note that nan infers that
    the label mask also does not contain that label

    :param result_mask: sitk image object for the result mask from model
    :param label_mask: sitk image object for the labelled mask
    :param num_classes: number of classes including background, it can be set to 2
                        to get binary dice of multilabel mask
    :return: a list of float
    """

    if isinstance(result_mask, sitk.Image):
        assert (
            np.linalg.norm(np.array(result_mask.GetSpacing()) - np.array(label_mask.GetSpacing()))
            < 1e-2
        ), "the spacings of label mask and result mask do not match"
        result_mask = sitk.GetArrayFromImage(result_mask)
        label_mask = sitk.GetArrayFromImage(label_mask)

    assert (
        result_mask.shape == label_mask.shape
    ), "the shapes of label mask and result mask do not match"

    result_tensor = torch.from_numpy(result_mask.astype(np.int32))
    label_tensor = torch.from_numpy(label_mask.astype(np.int32))
    if use_gpu and torch.cuda.is_available():
        result_tensor = result_tensor.cuda()
        label_tensor = label_tensor.cuda()

    v1 = torch.vstack(torch.where(result_tensor > 0))
    if 0 in v1.size():
        min_corner_1 = result_tensor.new_zeros(3)
        max_corner_1 = result_tensor.new_tensor(result_tensor.size())
    else:
        min_corner_1, _ = v1.min(dim=1)
        max_corner_1, _ = v1.max(dim=1)

    v2 = torch.vstack(torch.where(label_tensor > 0))
    if 0 in v2.size():
        min_corner_2 = label_tensor.new_zeros(3)
        max_corner_2 = label_tensor.new_tensor(label_tensor.size())
    else:
        min_corner_2, _ = v2.min(dim=1)
        max_corner_2, _ = v2.max(dim=1)

    spz, spy, spx = torch.minimum(min_corner_1, min_corner_2)
    epz, epy, epx = torch.maximum(max_corner_1, max_corner_2) + 1

    result_index = result_tensor[spz:epz, spy:epy, spx:epx].long()
    label_index = label_tensor[spz:epz, spy:epy, spx:epx].long()

    if num_classes is None:
        num_classes = torch.unique(label_index).__len__()

    if num_classes == 2:
        result_index[result_index > 0] = 1
        label_index[label_index > 0] = 1
    result_tensor_oh = result_tensor.new_zeros(
        [num_classes, *result_index.size()], dtype=torch.uint8
    )
    result_tensor_oh.scatter_(0, result_index.unsqueeze(0), 1)
    label_tensor_oh = label_tensor.new_zeros([num_classes, *label_index.size()], dtype=torch.uint8)
    label_tensor_oh.scatter_(0, label_index.unsqueeze(0), 1)

    eps = 1e-7
    intersection = torch.sum(result_tensor_oh * label_tensor_oh, dim=[1, 2, 3])
    dices = (2.0 * intersection) / (
        result_tensor_oh.sum(dim=[1, 2, 3]) + label_tensor_oh.sum(dim=[1, 2, 3]) + eps
    )

    dices = torch.where(label_tensor_oh.sum(dim=[1, 2, 3]) > 0, dices, dices.new_tensor(np.nan))
    dices = dices.cpu().numpy()[1:]
    return dices

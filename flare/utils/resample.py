# Copyright (c) DAMO Health

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from flare.utils.frame import Frame3d, world_box
from flare.utils.framed_tensor import FramedTensor
from typing import Iterable, List, Tuple, Union


def resample_itkimage_torai(
    sitk_im: sitk.Image,
    spacing: Union[List, Tuple, np.ndarray],
    interpolator: str = "nearest",
    pad_value: Union[int, float] = -1024,
) -> sitk.Image:
    """
    resample an image to RAI coordinate system

    :param sitk_im: input image
    :param spacing: destination spacing
    :param interpolator: interpolation method, can be 'nearest' and linear, defaults to 'nearest'
    :param pad_value: value for voxels extroplated
    :return: the resampled SimpleITK image object
    """

    min_corner, max_corner = world_box(sitk_im)

    origin = min_corner.tolist()
    direction = np.eye(3, dtype=np.double).flatten().tolist()
    size = ((max_corner - min_corner) / (np.array(spacing))).round().astype(np.int32).tolist()

    return resample_base(
        sitk_im,
        origin=origin,
        direction=direction,
        spacing=spacing,
        size=size,
        interpolator=interpolator,
        pad_value=pad_value,
    )


def resample_itkimage_withsize(
    itkimage: sitk.Image,
    new_size: Union[List, Tuple, np.ndarray],
    interpolator: str = "nearest",
    pad_value: Union[int, float] = -1024,
) -> sitk.Image:
    """
    Image resize with size by sitk resampleImageFilter.

    :param itkimage: input itk image or itk volume.
    :param new_size: the target size of the resampled image, such as [120, 80, 80].
    :param interpolator: for mask used nearest, for image linear is an option.
    :param pad_value: the value for the pixel which is out of image.
    :return: resampled itk image.
    """

    # get resize factor
    origin_size = np.array(itkimage.GetSize())
    new_size = np.array(new_size)
    factor = origin_size / new_size

    # get new spacing
    origin_spcaing = itkimage.GetSpacing()
    new_spacing = factor * origin_spcaing

    itkimg_resampled = resample_base(
        itkimage,
        itkimage.GetOrigin(),
        itkimage.GetDirection(),
        new_spacing,
        new_size,
        interpolator,
        pad_value,
    )

    return itkimg_resampled


def resample_itkimage_withspacing(
    itkimage: sitk.Image,
    new_spacing: Union[List, Tuple, np.ndarray],
    interpolator: str = "nearest",
    pad_value: Union[int, float] = -1024,
) -> sitk.Image:
    """
    Image resize with size by sitk resampleImageFilter.

    :param itkimage: input itk image or itk volume.
    :param new_spacing: the target spacing of the resampled image, such as [1.0, 1.0, 1.0].
    :param interpolator: for mask used nearest, for image linear is an option.
    :param pad_value: the value for the pixel which is out of image.
    :return: resampled itk image.
    """

    # get resize factor
    origin_spacing = itkimage.GetSpacing()
    new_spacing = np.array(new_spacing, float)
    factor = new_spacing / origin_spacing

    # get new image size
    origin_size = itkimage.GetSize()
    new_size = origin_size / factor
    new_size = new_size.astype(np.int)

    itkimg_resampled = resample_base(
        itkimage,
        itkimage.GetOrigin(),
        itkimage.GetDirection(),
        new_spacing,
        new_size,
        interpolator,
        pad_value,
    )

    return itkimg_resampled


def resample_base(
    sitk_im: sitk.Image,
    origin: Union[List, Tuple, np.ndarray],
    direction: Union[List, Tuple],
    spacing: Union[List, Tuple, np.ndarray],
    size: Union[List, Tuple, np.ndarray],
    interpolator: str = "nearest",
    pad_value: Union[int, float] = -1024,
) -> sitk.Image:
    """
    the base resample function, can be used to resample a small patch out of the original image
    or to resample to sample patch back to the original image, and of course, to resize a volume

    :param sitk_im: input image
    :param origin: the origin of the resampled volume
    :param direction: the direction of the resampled volume
    :param spacing: the spacing of the resampled volume
    :param size: the output size of the resampled volume
    :param interpolator: interpolation method, can be 'nearest' and linear, defaults to 'nearest'
    :param pad_value: value for voxels extroplated
    :return: the resampled SimpleITK image object
    """
    size = [int(s) for s in size]
    SITK_INTERPOLATOR_DICT = {
        "nearest": sitk.sitkNearestNeighbor,
        "linear": sitk.sitkLinear,
        "gaussian": sitk.sitkGaussian,
        "label_gaussian": sitk.sitkLabelGaussian,
        "bspline": sitk.sitkBSpline,
        "hamming_sinc": sitk.sitkHammingWindowedSinc,
        "cosine_windowed_sinc": sitk.sitkCosineWindowedSinc,
        "welch_windowed_sinc": sitk.sitkWelchWindowedSinc,
        "lanczos_windowed_sinc": sitk.sitkLanczosWindowedSinc,
    }

    assert (
        interpolator in SITK_INTERPOLATOR_DICT.keys()
    ), "`interpolator` should be one of {}".format(SITK_INTERPOLATOR_DICT.keys())

    sitk_interpolator = SITK_INTERPOLATOR_DICT[interpolator]

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize(size)
    resample_filter.SetOutputSpacing(np.array(spacing).tolist())
    resample_filter.SetOutputOrigin(np.array(origin).tolist())
    resample_filter.SetOutputDirection(direction)
    resample_filter.SetOutputPixelType(sitk_im.GetPixelID())
    resample_filter.SetDefaultPixelValue(pad_value)
    resample_filter.SetInterpolator(sitk_interpolator)

    img = resample_filter.Execute(sitk_im)

    return img


def resample_itkimage_withspacing_by_torch(
    itkimage: sitk.Image,
    new_spacing: Union[List, Tuple, np.ndarray],
    interpolator: str = "nearest",
    pad_value: Union[int, float] = -1024,
) -> sitk.Image:

    mode = "nearest"
    if interpolator == "linear":
        mode = "trilinear"

    dtype_conversion = False
    if itkimage.GetPixelID() != 8:
        img = sitk.Cast(itkimage, sitk.sitkFloat32)
        dtype_conversion = True
    else:
        img = itkimage

    im_tensor = torch.from_numpy(sitk.GetArrayFromImage(img)).unsqueeze(0).unsqueeze(0)
    if torch.cuda.is_available():
        im_tensor = im_tensor.cuda()
    ratio = np.array(itkimage.GetSpacing()) / new_spacing
    ratio = ratio[::-1]
    im_tensor = F.interpolate(im_tensor, scale_factor=ratio, mode=mode, align_corners=False)
    im_tensor = im_tensor.squeeze().cpu()
    img = sitk.GetImageFromArray(im_tensor.numpy())
    img.SetSpacing(np.array(new_spacing).tolist())
    img.SetOrigin(itkimage.GetOrigin())

    if dtype_conversion:
        img = sitk.Cast(img, itkimage.GetPixelID())

    return img


def crop_roi_with_center(
    itk_img: sitk.Image,
    center_w: Iterable[float],
    spacing: Iterable[float],
    x_axis: Iterable[float],
    y_axis: Iterable[float],
    z_axis: Iterable[float],
    size: Iterable[int],
    interpolator: str = "nearest",
    pad_value=-1024,
):
    frame = Frame3d()
    frame.origin = list(center_w)
    frame.direction = np.vstack([x_axis, y_axis, z_axis]).transpose().flatten().tolist()
    frame.spacing = spacing
    size = np.array(size).reshape(3)
    true_origin = frame.voxel_to_world(-size / 2)
    roi = resample_base(
        itk_img, true_origin, frame.direction, frame.spacing, size, interpolator, pad_value
    )
    return roi


def resample_framed_tensor(
    framed_tensor: FramedTensor,
    origin: Union[List, Tuple, np.ndarray],
    direction: Union[List, Tuple],
    spacing: Union[List, Tuple, np.ndarray],
    size: Union[List, Tuple, np.ndarray],
    interpolator: str = "nearest",
    pad_value: Union[int, float] = -1024,
    use_gpu: bool = True,
    work_with_float16=False,
):
    working_dtype = torch.float32
    if work_with_float16:
        working_dtype = torch.float16
    work_device = torch.device("cpu")
    if use_gpu and torch.cuda.is_available():
        work_device = framed_tensor.device
    else:
        framed_tensor = framed_tensor.cpu()
        # "arange_cpu" not implemented for 'Half'
        working_dtype = torch.float32
    original_dtype = framed_tensor.tensor.dtype

    # if framed_tensor.tensor.dtype not in [torch.float32, torch.float16]:
    #     print(
    #         "warning: as torch.grid_sample only support float tensors,\n"
    #         "it's better to convert the input tensor to float beforehand.\n"
    #         "this function will automatically convert non-float input to float \n"
    #         "and then convert the dtype of the output back to the same as input"
    #     )

    framed_tensor = framed_tensor.type(working_dtype)

    range_tensors = [
        torch.arange(0, s, dtype=working_dtype, device=work_device) for s in size[::-1]
    ]
    if "indexing" in torch.meshgrid.__code__.co_varnames:
        output_voxel = torch.meshgrid(range_tensors, indexing="ij")
    else:
        output_voxel = torch.meshgrid(range_tensors)
    output_voxel = torch.stack(output_voxel, dim=0).flip(0).view(3, -1)
    output_frame = Frame3d()
    output_frame.spacing = spacing
    output_frame.origin = origin
    output_frame.direction = direction
    output_world = output_frame.voxel_to_world(output_voxel).type(working_dtype)
    voi_x, voi_y, voi_z = framed_tensor.frame.world_to_voxel(output_world, is_round=False).type(
        working_dtype
    )
    voi_x = voi_x.view(*size[::-1])
    voi_y = voi_y.view(*size[::-1])
    voi_z = voi_z.view(*size[::-1])

    VZ, VY, VX = framed_tensor.tensor.size()
    voi_x = voi_x / (VX - 1) * 2 - 1
    voi_y = voi_y / (VY - 1) * 2 - 1
    voi_z = voi_z / (VZ - 1) * 2 - 1
    grid_on_input = torch.stack([voi_x, voi_y, voi_z], dim=3).unsqueeze(0).type(working_dtype)

    mode = "bilinear"
    if interpolator == "nearest":
        mode = "nearest"
    input_tensor = (framed_tensor.tensor - pad_value).type(working_dtype).unsqueeze(0).unsqueeze(0)
    if torch.cuda.is_available() and input_tensor.device == torch.device("cpu"):
        input_tensor = input_tensor.cuda()
        grid_on_input = grid_on_input.cuda()

    output_tensor = (
        F.grid_sample(
            input_tensor, grid_on_input, mode=mode, padding_mode="zeros", align_corners=True
        )
        + pad_value
    )
    if output_tensor.dtype != original_dtype:
        if original_dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            output_tensor = output_tensor.round()
        output_tensor = output_tensor.type(original_dtype)
    output = FramedTensor(output_tensor.squeeze(), output_frame)
    output = output.to(framed_tensor.device)

    return output


def merge_mask_by_prob_patches(
    output_frame: Frame3d,
    output_size: List[int],
    framed_mask_crops: List[FramedTensor],
    framed_prob_crops: List[FramedTensor],
    labels: List[int] = None,
    use_gpu=True,
    work_with_float16=False,
    pad_on_cpu=True,
) -> Tuple[FramedTensor, FramedTensor, Tuple[np.array, np.array]]:
    """
    merge masks patches by prob patches, which generated in the detection-then-segmentation paradigm

    :param output_frame: output frame
    :param output_size: output size
    :param framed_mask_crops: list of FramedTensors for mask crops
    :param framed_prob_crops: list of FramedTensors for prob crops
    :param labels: list of labels, can ben None for binary segmentation
    :param use_gpu: whether to use gpu for resampling
    :param work_with_float16: whether to work with float16 to save gpu memroy
    :param pad_on_cpu: whether to copy roi data to cpu then pad or pad directly on gpu,
                       will save memory but a bit slower
    :return: FramedTensor for merged mask, FramedTensor for merged probability map,
             a tuple of numpy array for roi start voxel and end voxel, in x, y, z order
    """
    if len(framed_mask_crops) == 0 or len(framed_prob_crops) == 0:
        raise ValueError("input framed mask and prob crops lists should not be empty!")
    if labels is None:
        labels = np.ones(len(framed_prob_crops), dtype=np.uint8)
    assert len(framed_prob_crops) == len(labels), "length of patches must be equal to len(labels)"

    working_dtype = torch.float32
    if work_with_float16:
        working_dtype = torch.float16
    framed_prob_crops = [t.type(working_dtype) for t in framed_prob_crops]

    work_device = torch.device("cpu")
    if use_gpu and torch.cuda.is_available():
        work_device = framed_prob_crops[0].device
    else:
        framed_prob_crops = [t.cpu() for t in framed_prob_crops]
        framed_mask_crops = [t.cpu() for t in framed_mask_crops]

    eps, sps = [], []
    for idx, framed_prob_crop in enumerate(framed_prob_crops):
        start = [0, 0, 0]
        end = framed_prob_crop.tensor.size()[::-1]
        coords = list(zip(start, end))
        corners = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    corner_on_crop_v = [coords[0][i], coords[1][j], coords[2][k]]
                    corner_on_crop_w = framed_prob_crop.frame.voxel_to_world(corner_on_crop_v)
                    corner_on_out_v = output_frame.world_to_voxel(corner_on_crop_w)
                    corners.append(corner_on_out_v)
        corners = np.vstack(corners)
        sp = np.maximum([0, 0, 0], corners.min(axis=0))
        ep = np.minimum(output_size, corners.max(axis=0))
        sps.append(sp)
        eps.append(ep)
    roi_sp, roi_ep = np.array(sps).min(axis=0), np.array(eps).max(axis=0)
    roi_size = list(roi_ep - roi_sp)

    roi_mask_tensor = torch.zeros(roi_size[::-1], device=work_device, dtype=torch.uint8)
    roi_prob_tensor = torch.zeros(roi_size[::-1], device=work_device, dtype=working_dtype)

    for idx, framed_prob_crop in enumerate(framed_prob_crops):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sp, ep = sps[idx], eps[idx]
        resample_origin = output_frame.voxel_to_world(sp).tolist()
        resample_size = ep - sp

        resampled_prob = resample_framed_tensor(
            framed_prob_crop,
            resample_origin,
            output_frame.direction,
            output_frame.spacing,
            resample_size,
            "linear",
            0,
            use_gpu,
            work_with_float16,
        ).tensor
        resampled_mask = resample_framed_tensor(
            framed_mask_crops[idx],
            resample_origin,
            output_frame.direction,
            output_frame.spacing,
            resample_size,
            "nearest",
            0,
            use_gpu,
            work_with_float16,
        ).tensor
        sp, ep = sp - roi_sp, ep - roi_sp
        prob_buffer = roi_prob_tensor[sp[2] : ep[2], sp[1] : ep[1], sp[0] : ep[0]]
        mask_buffer = roi_mask_tensor[sp[2] : ep[2], sp[1] : ep[1], sp[0] : ep[0]]
        buffer_index = resampled_prob > prob_buffer
        mask_buffer[buffer_index] = resampled_mask[buffer_index] * labels[idx]
        prob_buffer[buffer_index] = resampled_prob[buffer_index]

    pad_size = []
    # note that the pad_size for torch starts from last dimension, i.e. X axis in patient coord
    for ps in zip(roi_sp, np.array(output_size) - roi_ep):
        pad_size.extend(ps)
    if pad_on_cpu:
        roi_mask_tensor = roi_mask_tensor.cpu()
        roi_prob_tensor = roi_prob_tensor.cpu()
    output_mask_tensor = torch.constant_pad_nd(roi_mask_tensor, pad_size, 0)
    output_prob_tensor = torch.constant_pad_nd(roi_prob_tensor, pad_size, 0)
    output_mask = FramedTensor(output_mask_tensor, output_frame)
    output_prob = FramedTensor(output_prob_tensor, output_frame).type(torch.float32)

    return output_mask, output_prob, (roi_sp, roi_ep)

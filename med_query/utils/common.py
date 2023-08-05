# Copyright (c) DAMO Health
import numpy as np
from monai.networks.layers.factories import Norm
from monai.networks.nets import UNETR, DynUNet, UNet
from torch.optim import lr_scheduler
from typing import List, Optional, Tuple


def convert_df_to_dicts(case_df):
    case_dicts = []
    for idx, row in case_df.iterrows():
        bbox = {
            "filename": row["filename"],
            "label": row["label"],
            "center_w": [row["world_x"], row["world_y"], row["world_z"]],
            "x_axis_local": [row["dx1"], row["dx2"], row["dx3"]],
            "y_axis_local": [row["dy1"], row["dy2"], row["dy3"]],
            "z_axis_local": [row["dz1"], row["dz2"], row["dz3"]],
        }
        for key in ["width", "height", "depth"]:
            bbox.update({key: row[key]})
        case_dicts.append(bbox)
    return case_dicts


def axis_angle_to_rotation_matirx(axis, degree):
    """
    Compute rotation matrix from axis-angle.

    :param axis: array-like, shape (3,), Axis of rotation: (x, y, z)
    :param degree: degree to rotate
    :return: array-like, shape (3, 3), Rotation matrix
    """

    axis = axis / np.linalg.norm(axis)
    ux, uy, uz = axis
    theta = np.deg2rad(degree)
    c = np.cos(theta)
    s = np.sin(theta)
    ci = 1.0 - c
    rotation_matrix = np.array(
        [
            [ci * ux * ux + c, ci * ux * uy - uz * s, ci * ux * uz + uy * s],
            [ci * uy * ux + uz * s, ci * uy * uy + c, ci * uy * uz - ux * s],
            [ci * uz * ux - uy * s, ci * uz * uy + ux * s, ci * uz * uz + c],
        ]
    )
    return rotation_matrix


def get_scheduler(optimizer, opt):
    if opt.lr_policy == "lambda":
        lambda_rule = lambda epoch: 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(
            opt.niter_decay + 1
        )
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == "multi_step":
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_milestones, gamma=0.1)
    elif opt.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.max_iterations)
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, modee="min", factor=0.2, threshold=0.01, patience=5
        )
    elif opt.lr_policy == "invariant":
        lambda_rule = lambda epoch: 1.0
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "warm_up_multi_step":
        lambda_rule = (
            lambda epoch: (epoch + 1) / opt.warm_up_epochs
            if epoch < opt.warm_up_epochs
            else 0.1 ** len([m for m in opt.lr_milestones if m <= epoch])
        )
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:
        return NotImplementedError(f"Learning rate policy {opt.lr_policy} is not implemented")

    return scheduler


def mask_np_bbox(
    mask_np: np.ndarray, mask_value: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    get the bounding box of foreground label(s) in voxel coordinate system,
    if foreground label not found in the mask, will return (None, None)

    :param mask: the input ndarray object for mask
    :param mask_value: foreground label value, will use all positive labels if not specified
    :return: the bounding box in voxel coordinate system of numpy, i.e: (vz, vy, vx)!
    """
    if mask_value is not None:
        vz, vy, vx = np.where(mask_np == mask_value)
    else:
        vz, vy, vx = np.where(mask_np > 0)

    min_corner, max_corner = None, None
    if vz.size > 0:
        start = [vz.min(), vy.min(), vx.min()]
        end = [vz.max(), vy.max(), vx.max()]

        coords = list(zip(start, end))
        corners = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    corner_coord = [coords[0][i], coords[1][j], coords[2][k]]
                    corners.append(corner_coord)
        corners = np.vstack(corners)
        min_corner = corners.min(axis=0)
        max_corner = corners.max(axis=0)
    return min_corner, max_corner


def get_network(
    net_name, in_channels, num_classes, dimensions, normalization, base=8, downsample=5
):
    if net_name.startswith("unet"):
        norms = {"instancenorm": Norm.INSTANCE, "batchnorm": Norm.BATCH, "groupnorm": Norm.GROUP}
        max_stride = 2 ** downsample
        channels = [base * 2 ** i for i in range(downsample + 1)]
        strides = [2] * downsample
        if net_name.endswith("16"):
            max_stride = max_stride // 2
            strides.pop()
            channels.pop()
        net = UNet(
            dimensions=dimensions,
            in_channels=in_channels,
            out_channels=num_classes,
            channels=channels,
            strides=strides,
            num_res_units=2,
            norm=norms[normalization],
        )
    elif net_name == "unetr":
        max_stride = 16
        norms = {"instancenorm": "instance", "batchnorm": "batch", "groupnorm": "group"}
        net = UNETR(
            in_channels=in_channels,
            out_channels=num_classes,
            img_size=(32, 128, 256),
            feature_size=8,
            num_heads=6,
            norm_name=norms[normalization],
        )
    elif net_name == "dynunet":
        strides = [[1, 2, 2]] * downsample
        kernels = [3] * downsample
        strides.insert(0, 1)
        kernels.insert(0, 3)
        max_stride = 2 ** downsample
        channels = [base * 2 ** i for i in range(downsample + 1)]
        net = DynUNet(
            spatial_dims=dimensions,
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            filters=channels,
            deep_supervision=True,
            deep_supr_num=2,
        )
    else:
        raise ValueError(f"unsupported network: {net_name}")

    return net, max_stride


def longest_common_prefix_str(names: List[str]) -> str:
    chars_list = []
    for chars in zip(*names):
        if len(set(chars)) == 1:
            chars_list.append(chars[0])
        else:
            break
    return "".join(chars_list)


def longest_common_postfix_str(names: List[str]) -> str:
    reversed_names = map(lambda x: x[::-1], names)
    reversed_lcp = longest_common_prefix_str(reversed_names)
    return reversed_lcp[::-1]

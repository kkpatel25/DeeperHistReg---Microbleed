### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union, Tuple

### External Imports ###
import math
import numpy as np
import scipy.ndimage as nd
import torch as tc
import torch.nn.functional as F
import torchvision.transforms as tr
import utils as u

### Internal Imports ###



########################


def generate_grid(tensor : tc.Tensor=None, tensor_size: tc.Tensor=None, device: str=None) -> tc.Tensor:
    """
    Generates the identity grid for a given tensor size.

    Parameters
    ----------
    tensor : tc.Tensor
        The tensor to be used as template
    tensor_size : tc.Tensor or tc.Size
        The tensor size used to generate the regular grid
    device : str
        The device to generate the grid on
    Returns
    ----------
    grid : tc.Tensor
        The regular grid (relative for warp_tensor with align_corners=False)
    """
    if tensor is not None:
        tensor_size = tensor.size()
    if device is None:
        identity_transform = tc.eye(len(tensor_size)-1)[:-1, :].unsqueeze(0).type_as(tensor)
    else:
        identity_transform = tc.eye(len(tensor_size)-1, device=device)[:-1, :].unsqueeze(0)
    identity_transform = tc.repeat_interleave(identity_transform, tensor_size[0], dim=0)
    grid = F.affine_grid(identity_transform, tensor_size, align_corners=False)
    return grid

def warp_tensor(tensor: tc.Tensor, displacement_field: tc.Tensor, grid: tc.Tensor=None, mode: str='bilinear', padding_mode: str='zeros', device: str=None) -> tc.Tensor:
    """
    Transforms a tensor with a given displacement field.
    Uses F.grid_sample for the structured data interpolation (only linear and nearest supported).
    Be careful - the autogradient calculation is possible only with mode set to "bilinear".

    Parameters
    ----------
    tensor : tc.Tensor
        The tensor to be transformed (BxYxXxZxD)
    displacement_field : tc.Tensor
        The PyTorch displacement field (BxYxXxZxD)
    grid : tc.Tensor (optional)
        The input identity grid (optional - may be provided to speed-up the calculation for iterative algorithms)
    mode : str
        The interpolation mode ("bilinear" or "nearest")
    device : str
        The device to generate the warping grid if not provided
    Returns
    ----------
    transformed_tensor : tc.Tensor
        The transformed tensor (BxYxXxZxD)
    """
    if grid is None:
        grid = generate_grid(tensor=tensor, device=device)
    sampling_grid = grid + displacement_field
    transformed_tensor = F.grid_sample(tensor, sampling_grid, mode=mode, padding_mode=padding_mode, align_corners=False)
    return transformed_tensor

def warp_large_tensor(tensor: tc.Tensor, displacement_field: tc.Tensor, grid: tc.Tensor=None, padding_mode: str='zeros', device: str="cpu"):
    """
    Efficiently warp an int64 label map with nearest-neighbor sampling and preserve label precision.
    """
    if grid is None:
        grid = generate_grid(tensor=tensor, device=device)

    sampling_grid = grid + displacement_field

    # Vectorized mapping of int64 labels to indices
    labels = tc.unique(tensor)
    labels_sorted, _ = tc.sort(labels)
    index_tensor = tc.searchsorted(labels_sorted, tensor)

    # Warp as float32 to use grid_sample
    warped_index_tensor = F.grid_sample(
        index_tensor.float(),
        sampling_grid.float(),
        mode="nearest",
        padding_mode=padding_mode,
        align_corners=False
    ).long()

    # Map back to original int64 labels
    warped_tensor = labels_sorted[warped_index_tensor]

    return warped_tensor


def transform_tensor(tensor: tc.Tensor, sampling_grid: tc.Tensor, grid: tc.Tensor=None, device: str="cpu", mode: str='bilinear') -> tc.Tensor:
    """
    Transforms a tensor with a given sampling grid.
    Uses F.grid_sample for the structured data interpolation (only linear and nearest supported).
    Be careful - the autogradient calculation is possible only with mode set to "bilinear".

    Parameters
    ----------
    tensor : tc.Tensor
        The tensor to be transformed (BxYxXxZxD)
    sampling_grid : tc.Tensor
        The PyTorch sampling grid
    grid : tc.Tensor (optional)
        The input identity grid (optional - may be provided to speed-up the calculation for iterative algorithms)
    device : str
        The device used for warping (e.g. "cpu" or "cuda:0")
    mode : str
        The interpolation mode ("bilinear" or "nearest")

    Returns
    ----------
    transformed_tensor : tc.Tensor
        The transformed tensor (BxYxXxZxD)
    """
    transformed_tensor = F.grid_sample(tensor, sampling_grid, mode=mode, padding_mode='zeros', align_corners=False)
    return transformed_tensor

def compose_displacement_fields(displacement_field_1 : tc.Tensor, displacement_field_2 : tc.Tensor) -> tc.Tensor:
    """
    TODO
    """
    sampling_grid = generate_grid(tensor_size=(displacement_field_1.size(0), 1, displacement_field_1.size(1), displacement_field_1.size(2)), device=displacement_field_1.device)
    composed_displacement_field = F.grid_sample((sampling_grid + displacement_field_1).permute(0, 3, 1, 2), sampling_grid + displacement_field_2, padding_mode='border', align_corners=False).permute(0, 2, 3, 1)
    composed_displacement_field = composed_displacement_field - sampling_grid
    return composed_displacement_field

def tc_transform_to_tc_df(transformation: tc.Tensor, size: tc.Size) -> tc.Tensor:
    """
    Transforms the transformation tensor into the displacement field tensor.

    Parameters
    ----------
    transformation : tc.Tensor
        The transformation tensor (B x transformation size (2x3 or 3x4))
    size : tc.Tensor (or list, or tuple)
        The desired displacement field size
    Returns
    ----------
    resampled_displacement_field: tc.Tensor
        The resampled displacement field (BxYxXxZxD)
    """
    deformation_field = F.affine_grid(transformation, size=size, align_corners=False)
    size = (deformation_field.size(0), 1) + deformation_field.size()[1:-1]
    grid = generate_grid(tensor_size=size, device=transformation.device)
    displacement_field = deformation_field - grid
    return displacement_field

def affine2theta(affine : np.ndarray, shape : tuple) -> tc.Tensor:
    """
    TODO
    """
    h, w = shape[0], shape[1]
    temp = affine
    theta = tc.zeros([2, 3])
    theta[0, 0] = temp[0, 0]
    theta[0, 1] = temp[0, 1]*h/w
    theta[0, 2] = temp[0, 2]*2/w + theta[0, 0] + theta[0, 1] - 1
    theta[1, 0] = temp[1, 0]*w/h
    theta[1, 1] = temp[1, 1]
    theta[1, 2] = temp[1, 2]*2/h + theta[1, 0] + theta[1, 1] - 1
    return theta

def theta2affine(theta : tc.Tensor, shape : tuple) -> np.ndarray:
    """
    TODO
    """
    h, w = shape[0], shape[1]
    temp = theta
    affine = np.zeros((2, 3))
    affine[1, 2] = (temp[1, 2] - temp[1, 0] - temp[1, 1] + 1)*h/2
    affine[1, 1] = temp[1, 1]
    affine[1, 0] = temp[1, 0]*h/w
    affine[0, 2] = (temp[0, 2] - temp[0, 0] - temp[0, 1] + 1)*w/2
    affine[0, 1] = temp[0, 1]*w/h
    affine[0, 0] = temp[0, 0]
    return affine

def compose_transforms(t1 : tc.Tensor, t2 : tc.Tensor) -> tc.Tensor:
    """
    TODO
    """
    tr1 = tc.zeros((3, 3)).type_as(t1)
    tr2 = tc.zeros((3, 3)).type_as(t2)
    tr1[0:2, :] = t1
    tr2[0:2, :] = t2
    tr1[2, 2] = 1
    tr2[2, 2] = 1
    result = tc.mm(tr1, tr2)
    return result[0:2, :]

def generate_rigid_matrix(angle : float, x0 : float, y0 : float, tx : float, ty : float) -> np.ndarray:
    """
    TODO
    """
    angle = angle * np.pi/180
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    cm1 = np.array([
        [1, 0, x0],
        [0, 1, y0],
        [0, 0, 1]
    ])
    cm2 = np.array([
        [1, 0, -x0],
        [0, 1, -y0],
        [0, 0, 1]
    ])
    translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
    transform = cm1 @ rotation_matrix @ cm2 @ translation_matrix
    return transform[0:2, :]

def warp_landmarks(landmarks : np.ndarray, displacement_field : np.ndarray) -> np.ndarray:
    """
    TODO
    """
    landmarks_x = landmarks[:, 0]
    landmarks_y = landmarks[:, 1]
    ux = nd.map_coordinates(displacement_field[0, :, :], [landmarks_y, landmarks_x], mode='nearest')
    uy = nd.map_coordinates(displacement_field[1, :, :], [landmarks_y, landmarks_x], mode='nearest')
    new_landmarks = np.stack((landmarks_x + ux, landmarks_y + uy), axis=1)
    return new_landmarks

### Ecosystem Imports ###
import os
import sys
current_file = sys.modules[__name__]
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from typing import Tuple
import numpy as np
import torch as tc
import cv2

### Internal Imports ###
from deeperhistreg.dhr_paths import model_paths as p
from deeperhistreg.dhr_utils import utils as u
from deeperhistreg.dhr_utils import warping as w
from deeperhistreg.dhr_networks import superpoint as sp
########################


def superpoint_ransac(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO - documentation
    """
    resolution = params['registration_size']
    try:
        return_num_matches = params['return_num_matches']
    except KeyError:
        return_num_matches = False

    ### Initial Resampling ###
    resampled_source, resampled_target = u.initial_resampling(source, target, resolution)   

    ### SuperPoint keypoints and descriptors ###
    src = u.tensor_to_image(resampled_source)[:, :, 0]
    trg  = u.tensor_to_image(resampled_target)[:, :, 0]
    try:
        source_keypoints, _, target_keypoints, _, num_matches = calculate_keypoints(src, trg, params)
    except:
        final_transform = np.eye(3)
        final_transform = w.affine2theta(final_transform, (resampled_source.size(2), resampled_source.size(3))).type_as(source).unsqueeze(0)
        num_matches = 0
        return final_transform


    try:
        # [0, cv2.RANSAC, cv2.RHO, cv2.LMEDS]
        transform, _ = cv2.estimateAffinePartial2D(source_keypoints, target_keypoints, 1)
    except:
        transform = np.eye(3)[0:2, :]

    final_transform = np.eye(3)
    final_transform[0:2, 0:3] = transform
    try:
        final_transform = np.linalg.inv(final_transform)
    except:
        final_transform = np.eye(3)
    final_transform = w.affine2theta(final_transform, (resampled_source.size(2), resampled_source.size(3))).type_as(source).unsqueeze(0)

    if return_num_matches:
        return final_transform, num_matches
    else:
        return final_transform

def calculate_keypoints(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> Tuple[tc.Tensor, tc.Tensor, tc.Tensor, tc.Tensor]:
    """
    TODO
    """
    ### Params Unpack ###
    default_params = {'weights_path': p.superpoint_model_path, 'nms_dist': 4, 'conf_thresh': 0.015, "nn_thresh": 0.7, 'cuda': True, 'show': False}
    params = {**default_params, **params}
    weights_path = params['weights_path']
    nms_dist = params['nms_dist']
    conf_thresh = params['conf_thresh']
    nn_thresh = params['nn_thresh']
    cuda = params['cuda']

    ### Model Creation ###
    model = sp.SuperPointFrontend(weights_path, nms_dist, conf_thresh, nn_thresh, cuda=cuda)

    ### Keypoints / Descriptors Calculation ###
    src_pts, src_desc, src_heatmap = model.run(source)
    trg_pts, trg_desc, trg_heatmap = model.run(target)

    matches = sp.nn_match_two_way(src_desc, trg_desc, nn_thresh)

    src_pts = src_pts[:, matches[0, :].astype(np.int32)].swapaxes(0, 1)[:, 0:2].astype(np.float32)
    trg_pts = trg_pts[:, matches[1, :].astype(np.int32)].swapaxes(0, 1)[:, 0:2].astype(np.float32)
    src_desc = src_desc[:, matches[0, :].astype(np.int32)].swapaxes(0, 1)
    trg_desc = trg_desc[:, matches[1, :].astype(np.int32)].swapaxes(0, 1)
    return src_pts, src_desc, trg_pts, trg_desc, len(matches[0])

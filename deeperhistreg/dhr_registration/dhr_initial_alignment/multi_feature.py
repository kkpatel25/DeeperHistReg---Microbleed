### Ecosystem Imports ###
import os
import sys
current_file = sys.modules[__name__]
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

### External Imports ###
import numpy as np
import torch as tc
import cv2

### Internal Imports ###
from deeperhistreg.dhr_utils import utils as u
from deeperhistreg.dhr_utils import warping as w

import deeperhistreg.dhr_registration.dhr_initial_alignment.superpoint_ransac as spr
import deeperhistreg.dhr_registration.dhr_initial_alignment.sift_ransac as sr
import deeperhistreg.dhr_registration.dhr_initial_alignment.superpoint_superglue as sg
########################




def multi_feature(
    source : tc.Tensor,
    target : tc.Tensor,
    params : dict) -> tc.Tensor:
    """
    TODO
    """
    transforms = []
    echo = params['echo']
    angle_step = params['angle_step']
    device = params['device']
    resolution = params['registration_size']
    run_sift_ransac = params['run_sift_ransac']
    run_superpoint_superglue = params['run_superpoint_superglue']
    run_superpoint_ransac = params['run_superpoint_ransac']
    source, target = u.initial_resampling(source, target, resolution) 
    
    angle_start, angle_stop = -45, 45
    for angle in range(angle_start, angle_stop, angle_step):
        _, _, y_size, x_size = source.shape
        x_origin = x_size // 2 
        y_origin = y_size // 2
        r_transform = w.generate_rigid_matrix(angle, x_origin, y_origin, 0, 0)
        r_transform = w.affine2theta(r_transform, (source.size(2), source.size(3))).to(device).unsqueeze(0)
        current_displacement_field = w.tc_transform_to_tc_df(r_transform, (1, 1, source.size(2), source.size(3)))
        transformed_source = w.warp_tensor(source, current_displacement_field)
        registration_sizes = params['registration_sizes']
        for registration_size in registration_sizes:
            ex_params = {**params, **{'registration_size': registration_size}}
            if run_sift_ransac:
                if echo:
                    print("SIFT RANSAC")
                current_transform, num_matches = sr.sift_ransac(transformed_source, target, {**ex_params, **{'return_num_matches': True}})
                current_transform = w.compose_transforms(r_transform[0], current_transform[0]).unsqueeze(0)
                transforms.append((current_transform, num_matches))
            if run_superpoint_superglue:
                if echo:
                    print("SUPERPOINT SUPERGLUE")
                current_transform, num_matches = sg.superpoint_superglue(transformed_source, target, {**ex_params, **{'return_num_matches': True}})
                current_transform = w.compose_transforms(r_transform[0], current_transform[0]).unsqueeze(0)
                transforms.append((current_transform, num_matches))
            if run_superpoint_ransac:
                if echo:
                    print("SUPERPOINT RANSAC")
                current_transform, num_matches = spr.superpoint_ransac(transformed_source, target, {**ex_params, **{'return_num_matches': True}})
                current_transform = w.compose_transforms(r_transform[0], current_transform[0]).unsqueeze(0)
                transforms.append((current_transform, num_matches))
        
    best_matches = 0
    best_transform = tc.eye(3, device=source.device)[0:2, :].unsqueeze(0)
    for transform, num_matches in transforms:
        if num_matches > best_matches:
            best_transform = transform
            best_matches = num_matches
    if echo:
        print(f"Final matches: {best_matches}")
        print(f"Final transform: {best_transform}")
    return best_transform    
    

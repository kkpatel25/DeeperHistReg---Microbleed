### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union, Tuple

### External Imports ###
import numpy as np
import torch as tc


### Internal Imports ###
from dhr_utils import utils as u
from dhr_utils import warping as w

########################


def apply_deformation(
    source : np.ndarray,
    target_shape : Tuple[int,int,int],
    displacement_field : tc.Tensor,
    ) -> tc.Tensor:

    source = u.image_to_tensor(source).to(tc.float32)

    with tc.set_grad_enabled(False):
        displacement_field = u.resample_displacement_field_to_size(displacement_field, target_shape[:2])
        warped_source = w.warp_tensor(source, displacement_field)

    warped_source = warped_source.cpu().to(tc.uint8)
    return warped_source


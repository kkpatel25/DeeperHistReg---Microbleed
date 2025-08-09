### Ecosystem Imports ###
import os
import sys
current_file = sys.modules[__name__]
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from typing import Tuple

### External Imports ###
import numpy as np
import torch as tc


### Internal Imports ###
from arvind.deeperhistreg.dhr_utils import utils as u
from arvind.deeperhistreg.dhr_utils import warping as w

########################


def apply_deformation(
    source: np.ndarray,
    target_shape: Tuple[int, ...],
    displacement_field: tc.Tensor, device: str, mode: str = 'bilinear', dtype_in=tc.float32, dtype_out=tc.uint8, padding_mode: str = 'zeros'
    ) -> np.ndarray:

    source = u.image_to_tensor(source, device=device).to(dtype_in)

    with tc.set_grad_enabled(False):
        displacement_field = u.resample_displacement_field_to_size(displacement_field, target_shape[:2])
        if dtype_in == tc.float32:
            warped_source = w.warp_tensor(source, displacement_field, mode=mode, device = device, padding_mode=padding_mode)
        else:
            warped_source = w.warp_large_tensor(source, displacement_field, device=device, padding_mode=padding_mode)

    warped_source = warped_source.cpu().to(dtype_out)
    return u.tensor_to_image(warped_source)


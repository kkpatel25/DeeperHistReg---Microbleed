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
from dhr_input_output.dhr_savers import pair_full_saver as pair_saver
from dhr_input_output.dhr_savers import tiff_saver as tiff_saver
from dhr_utils import warping as w

########################


def apply_deformation(
    source : np.ndarray,
    target_shape : Tuple[int,int,int],
    warped_image_path : Union[pathlib.Path, str],
    displacement_field : tc.Tensor,
    saver : tiff_saver.WSISaver = tiff_saver.WSISaver,
    save_params : dict = tiff_saver.default_params,
    ) -> None:

    source = u.image_to_tensor(source).to(tc.float32)

    with tc.set_grad_enabled(False):
        displacement_field = u.resample_displacement_field_to_size(displacement_field, target_shape[:2])
        warped_source = w.warp_tensor(source, displacement_field)

    warped_source = warped_source.cpu().to(tc.uint8)

    to_save = pair_saver.PairFullSaver(saver, save_params)
    to_save.save_images(warped_source, target_shape[:2], warped_image_path)



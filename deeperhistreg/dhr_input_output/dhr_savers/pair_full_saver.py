### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union, Iterable, Tuple
import logging

### External Imports ###
import numpy as np
import torch as tc

import pyvips

### Internal Imports ###

from saver import WSISaver
from tiff_saver import TIFFSaver, default_params
from dhr_utils import utils as u

########################


class PairFullSaver():
    """
    TODO - documentation
    """
    def __init__(
        self,
        saver : WSISaver = TIFFSaver,
        save_params : dict = default_params):
        """
        TODO
        """
        self.saver : WSISaver = saver()
        self.save_params = save_params

    def unpad_images(
        self,
        source : Union[np.ndarray, tc.Tensor, pyvips.Image],
        target : Union[np.ndarray, tc.Tensor, pyvips.Image],
        initial_padding : Iterable[int] = None,
        unpad_with_target : bool = False,
        ) -> Tuple[Union[np.ndarray, tc.Tensor, pyvips.Image], Union[np.ndarray, tc.Tensor, pyvips.Image]]:
        """
        TODO
        """
        unpadded_source, unpadded_target = u.unpad(source, target, initial_padding, unpad_with_target)
        return unpadded_source, unpadded_target
    
    def crop_to_template(
        self,
        image : Union[np.ndarray, tc.Tensor, pyvips.Image],
        target_shape : Tuple[int, int],
        ) -> Union[np.ndarray, tc.Tensor, pyvips.Image]:
        """
        TODO
        """
        cropped_image = u.crop_to_template(image, target_shape)
        return cropped_image
        
    def save_images(
        self,
        source : Union[np.ndarray, tc.Tensor, pyvips.Image],
        target_shape : Tuple[int, int],
        source_path : Union[str, pathlib.Path],
        ) -> None:

        source_to_save = self.crop_to_template(source, target_shape)
        self.saver.save(source_to_save, source_path, self.save_params)

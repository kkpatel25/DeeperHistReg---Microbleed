### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

### External Imports ###
import numpy as np
import torch as tc
from typing import Union

### Internal Imports ###
from dhr_preprocessing import preprocessing as pre
from dhr_registration import initial_alignment_methods as ia
from dhr_registration import nonrigid_registration_methods as nr
from dhr_deformation import apply_deformation as adf
from dhr_utils import utils as u
from dhr_utils import warping as w

from dhr_input_output.dhr_savers import results_saver as rs

########################

class DeeperHistReg_FullResolution():
    def __init__(self, registration_parameters : dict):
        """
        TODO
        """
        self.registration_parameters = registration_parameters
        self.registration_parameters['device'] = self.registration_parameters['device'] if tc.cuda.is_available() else "cpu"
        self.device = self.registration_parameters['device']
        self.echo = self.registration_parameters['echo']
        self.case_name = self.registration_parameters['case_name']

    def process_array(self) -> None:
        """
        TODO
        """
        loading_params = self.registration_parameters['loading_params']
        pad_value = loading_params['pad_value']
        source_resample_ratio = loading_params['source_resample_ratio']
        target_resample_ratio = loading_params['target_resample_ratio']


        self.moving = u.smooth_and_resample_color_image(self.moving, source_resample_ratio)
        self.fixed = u.smooth_and_resample_color_image(self.fixed, target_resample_ratio)

        self.fixed, self.moving, self.padding_params = u.pad_to_same_size_np(self.fixed, self.moving, pad_value)

        self.padding_params['source_resample_ratio'] = source_resample_ratio
        self.padding_params['target_resample_ratio'] = target_resample_ratio

        # this is addition
        self.fixed, self.moving = u.image_to_tensor(self.fixed), u.image_to_tensor(self.moving)
        self.org_source, self.org_target = self.fixed.to(tc.float32).to(self.device), self.moving.to(tc.float32).to(self.device)

    def run_prepreprocessing(self) -> None:
        """
        TODO
        """
        with tc.set_grad_enabled(False):
            self.preprocessing_params = self.registration_parameters['preprocessing_params']
            preprocessing_function = pre.get_function(self.preprocessing_params['preprocessing_function'])
            self.pre_source, self.pre_target, _, _, self.postprocessing_params = preprocessing_function(self.org_source, self.org_target, None, None, self.preprocessing_params)

            self.padding_params['initial_resampling'] = self.postprocessing_params['initial_resampling']
            if self.postprocessing_params['initial_resampling']:
                self.padding_params['initial_resample_ratio'] = self.postprocessing_params['initial_resample_ratio']

            self.current_displacement_field = u.create_identity_displacement_field(self.pre_source)
            tc.cuda.empty_cache()

    def run_initial_registration(self) -> None:
        """
        TODO
        """
        if self.registration_parameters['run_initial_registration']:
            initial_registration_params = self.registration_parameters['initial_registration_params']
            initial_registration_function = ia.get_function(initial_registration_params['initial_registration_function'])
            self.initial_transform = initial_registration_function(self.pre_source, self.pre_target, initial_registration_params)
            self.initial_displacement_field = w.tc_transform_to_tc_df(self.initial_transform, self.pre_source.size())
            self.current_displacement_field = self.initial_displacement_field
            tc.cuda.empty_cache()

    def run_nonrigid_registration(self) -> None:
        """
        TODO
        """
        if self.registration_parameters['run_nonrigid_registration']:
            nonrigid_registration_params = self.registration_parameters['nonrigid_registration_params']
            nonrigid_registration_function = nr.get_function(nonrigid_registration_params['nonrigid_registration_function'])
            self.nonrigid_displacement_field = nonrigid_registration_function(self.pre_source, self.pre_target, self.current_displacement_field, nonrigid_registration_params)
            self.current_displacement_field = self.nonrigid_displacement_field
            tc.cuda.empty_cache()

    def preprocessing(self) -> None:
        self.run_prepreprocessing()

    def initial_registration(self) -> None:
        self.run_initial_registration()

    def nonrigid_registration(self) -> None:
        self.run_nonrigid_registration()

    def run_registration(
        self,
        fixed: Union[tc.Tensor, np.ndarray],
        moving: Union[tc.Tensor, np.ndarray]
    ) -> Union[tc.Tensor, np.ndarray]:

        self.fixed, self.moving = fixed, moving
        self.process_array()
        self.preprocessing()
        self.initial_registration()
        self.nonrigid_registration()
        # this now returns the displacement field as a tensor, I believe
        return self.current_displacement_field

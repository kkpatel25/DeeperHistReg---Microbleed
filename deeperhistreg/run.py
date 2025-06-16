### Ecosystem Imports ###
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from typing import Iterable
import argparse
from datetime import datetime
import shutil

### External Imports ###

### Internal Imports ###
from dhr_pipeline import full_resolution as fr
from dhr_pipeline import registration_params as rp


def run_registration(fixed: np.ndarray, moving: np.ndarray, **config):
    ### Parse Config ###
    try:
        registration_parameters_path = config['registration_parameters_path']
        registration_parameters = rp.load_parameters(registration_parameters_path)
    except KeyError:
        registration_parameters = config['registration_parameters']

    experiment_name = config['case_name']
    save_path = config['temporary_path']

    if save_path is None:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ### Run Registration ###
    try:
        registration_parameters['case_name'] = experiment_name
        pipeline = fr.DeeperHistReg_FullResolution(registration_parameters)
        # this function will return the displacement field

        displacement_field = pipeline.run_registration(fixed, moving)
        shutil.rmtree(save_path)
        return displacement_field

    except Exception as e:
        print(f"Exception: {e}")

def parse_args(args : Iterable) -> dict:
    ### Create Parser ###
    parser = argparse.ArgumentParser(description="DeeperHistReg arguments")
    parser.add_argument('--out', dest='output_path', type=str, help="Path to the output folder")

    ### Optional ###
    nonrigid_parameters_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deeperhistreg_params", "default_nonrigid.json")
    parser.add_argument('--params', dest='registration_parameters_path', type=str, default=nonrigid_parameters_path, help="Path to the JSON with registration parameters")
    parser.add_argument('--exp', dest='case_name', type=str, default="WSI_Registration", help="Case name")
    parser.add_argument('--dtmp', dest='delete_temporary_results', action='store_true', help='Delete temporary results?')
    parser.add_argument('--temp', dest='temporary_path', type=str, default=None,
                        help='Path to save the temporary results. Defaults to random folder inside the file directory.')

    ### Parse Parameters ###
    config = parser.parse_args()
    config = vars(config)
    return config

if __name__ == "__main__":
    config = parse_args(sys.argv[1:])
    run_registration(fixed, moving, **config)

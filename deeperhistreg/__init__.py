"""
Edited version of the DeeperHistReg package for use with the arvind package.

This module provides the dhr class, which handles:
- Elastic alignment of histological tissue to a scan
"""
from deeperhistreg.dhr_pipeline import full_resolution as direct_registration
from deeperhistreg.dhr_pipeline import registration_params as configs

from deeperhistreg.dhr_deformation.apply_deformation import apply_deformation
from deeperhistreg.run import run_registration


from download import download_file_gd

DHR_MODEL1_URL = "https://drive.google.com/uc?id=1rHih1wzwwsgi1864HOrzvJVi4VtqLtYr"
DHR_MODEL1_FILENAME = "superglue_outdoor.pth"

DHR_MODEL2_URL = "https://drive.google.com/uc?id=1gxDN9w2czyy9vDDRmQVOkkbwiKWY9TdL"
DHR_MODEL2_FILENAME = "superpoint_v1.pth"

# DOWNLOAD ALL REQUIRED FILES AT RUNTIME
download_file_gd(DHR_MODEL1_URL,DHR_MODEL1_FILENAME)
download_file_gd(DHR_MODEL2_URL,DHR_MODEL2_FILENAME)


"""
Edited version of the DeeperHistReg package for use with the arvind package.

This module provides the dhr class, which handles:
- Elastic alignment of histological tissue to a scan
"""


from deeperhistreg.dhr_pipeline import full_resolution as direct_registration
from deeperhistreg.dhr_pipeline import registration_params as configs

from deeperhistreg.dhr_deformation.apply_deformation import apply_deformation
from deeperhistreg.run import run_registration

### Ecosystem Imports ###
import os
import sys
current_file = sys.modules[__name__]
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

### External Imports ###
import numpy as np
import SimpleITK as sitk
import PIL
import matplotlib.pyplot as plt


import openslide

### Internal Imports ###
from dhr_utils import utils as u
from dhr_utils import warping as w

########################

def save_image(image, save_path, renormalize=True):
    # TODO - documentation
    if not save_path.parents[0].exists():
        save_path.parents[0].mkdir(parents=True, exist_ok=True)
    extension = os.path.splitext(save_path)[1]
    if image.shape[2] == 3:
        if extension == ".jpg" or extension == ".jpeg" or extension == ".png":
            if renormalize:
                image = (image * 255)
                image = image.astype(np.uint8)
                image = PIL.Image.fromarray(image)
                image.save(str(save_path))
            else:
                image = image.astype(np.uint8)
                image = PIL.Image.fromarray(image)
                image.save(str(save_path))
        else:
            sitk.WriteImage(sitk.GetImageFromArray(image), str(save_path))
    elif image.shape[2] == 1:
        if renormalize:
            image = (image[:, :, 0]*255)
            image = image.astype(np.uint8)
        else:
            image = (image[:, :, 0])
            image = image.astype(np.float32)
        sitk.WriteImage(sitk.GetImageFromArray(image), str(save_path))
    else:
        raise ValueError("Unsupported image format.")

def load_displacement_field(load_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(load_path))).astype(np.float32)

def save_displacement_field(displacement_field, save_path):
    sitk.WriteImage(sitk.GetImageFromArray(displacement_field.astype(np.float32)), str(save_path), useCompression=True)
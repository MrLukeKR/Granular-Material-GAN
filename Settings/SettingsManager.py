import numpy as np

from Settings import FileManager as fm
import os

settings_file = []
configuration = dict()

rescale_factor = 0
image_channels = 1
segmentation_amount = 3
segmentation_mode = "3D"
enable_normalisation = True
display_available = os.environ["DISPLAY"] is not None

# Reduction factor of downscaling an image (imres * resc) (DO NOT EDIT!)
rescale_amount = 2 ** rescale_factor

# Resolution of image (images will be resized to square [imres * imres]
image_resolution = np.uint(1024 / rescale_amount)


USE_BW = True


def load_settings():
    for line in open("Phase1.conf", "r"):
        line = line.strip()
        if not len(line) == 0 and not line.startswith('#'):
            sanitised = line.replace('"', '')
            setting = sanitised.split('=')

            configuration.__setitem__(setting[0], setting[1])

    # Resolve clashing settings
    if configuration.get("ENABLE_SEGMENTATION") == "True":
        configuration.__setitem__("ENABLE_VOXEL_SEPARATION", "True")

    return configuration

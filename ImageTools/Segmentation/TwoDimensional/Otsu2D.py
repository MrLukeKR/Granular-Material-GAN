import ImageTools.ImageManager as im
from skimage.filters import threshold_multiotsu
import numpy as np


def segment_image(image, return_separate=False):
    if len(image.shape) != 2:
        raise Exception("This segmentation method only accepts two-dimensional images. "
                        "The shape given is " + image.shape)

    try:
        thresholds = threshold_multiotsu(image)
        regions = np.digitize(image, bins=thresholds)

        void = regions == 0
        binder = regions == 1
        aggregate = regions == 2
    except (ValueError, IndexError) as e:
        void = np.ones(image.shape)
        aggregate = binder = regions = np.zeros(image.shape)

    if return_separate:
        return void, aggregate, binder, regions
    else:
        return regions

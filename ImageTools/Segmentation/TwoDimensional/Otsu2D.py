import ImageTools.ImageManager as im
import numpy as np


def segment_image(image, return_separate=False):
    if len(image.shape) != 2:
        raise Exception("This segmentation method only accepts two-dimensional images. "
                        "The shape given is " + image.shape)

    try:
        from skimage.filters import threshold_multiotsu
        thresholds = threshold_multiotsu(image)
        regions = np.digitize(image, bins=thresholds)

        void = regions == 0
        binder = regions == 1
        aggregate = regions == 2
    except ValueError:
        void = np.ones(image.shape)
        aggregate = binder = regions = np.zeros(image.shape)

    if return_separate:
        return void, aggregate, binder, regions
    else:
        return regions
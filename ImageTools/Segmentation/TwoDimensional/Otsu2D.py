import ImageTools.ImageManager as im
import numpy as np
from skimage.filters import threshold_multiotsu


def segment_image(image):
    if len(image.shape) != 2:
        raise Exception("This segmentation method only accepts two-dimensional images. "
                        "The shape given is " + image.shape)

    thresholds = threshold_multiotsu(image)

    regions = np.digitize(image, bins=thresholds)

    void = regions.min
    aggregate = regions.max
    binder = (regions.min < regions < regions.max)

    return void, aggregate, binder, regions
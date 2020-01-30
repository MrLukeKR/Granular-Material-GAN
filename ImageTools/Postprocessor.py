from scipy.ndimage import binary_closing, binary_opening

from Settings import SettingsManager as sm
from ExperimentTools.MethodologyLogger import Logger

import ImageTools.ImageManager as im


def close_segment(image):
    return binary_closing(image, iterations=0)


def open_segment(image):
    return binary_opening(image, iterations=0)


def clean_segments(images, pool):
    fixed_images = list()

    total = len(images)
    curr = 0

    Logger.print("\tMorphologically Cleaning Segments... ", end='\r')
    for ind, res in enumerate(pool.map(clean_segment, images)):
        fixed_images.insert(ind, res)
        curr += 1
        Logger.print("\tMorphologically Cleaning Segments... " + str(curr / total * 100) + "%", end='\r', flush=True)

        if sm.configuration.get("ENABLE_IMAGE_SAVING") == "True":
            im.save_image(res, str(ind), "Pre-processing/De-Noised/")

    Logger.print("\tDe-noising Images... done!")
    return fixed_images


def clean_segment(image):
    # return restoration.denoise_nl_means(image)
    # open_image = ndimage.binary_opening(image)
    # close_image = ndimage.binary_closing(open_image)

    clean_image = image > 0

    return clean_image


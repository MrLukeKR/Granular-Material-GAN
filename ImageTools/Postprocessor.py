from scipy.ndimage import binary_closing, binary_opening, grey_closing, grey_opening, binary_fill_holes
import cv2
import numpy as np
from Settings import SettingsManager as sm
from ExperimentTools.MethodologyLogger import Logger

import ImageTools.ImageManager as im


def remove_particles(image):
    max_contour_area = int(sm.configuration.get("MAXIMUM_BLOB_AREA"))

    prev_image = np.zeros(image.shape, dtype=np.uint8)
    fixed_image = np.zeros(image.shape, dtype=np.uint8)

    for i in range(0, 2):
        segment = np.array([x == i for x in image], dtype=np.uint8) * 255
        segment -= prev_image

        contours, hierarchy = cv2.findContours(segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours_area = []
        for con in contours:
            area = cv2.contourArea(con)
            if area <= max_contour_area:
                contours_area.append(con)

        image = cv2.cvtColor(segment, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(image, contours_area, -1, (0, 0, 255), 1)
        cv2.imshow("Blobs", image)
        cv2.waitKey(0)
        fixed_image += ([i for x in segment if x == 255])
        prev_image = segment


def close_segment(image, iterations=1):
    return binary_closing(image, iterations=iterations)


def open_segment(image, iterations=1):
    return binary_opening(image, iterations=iterations)


def fill_holes(image):
    return binary_fill_holes(image)


def open_close_segment(image, iterations=5):
    for i in range(iterations):
        image = open_segment(image)
        image = close_segment(image)

    return image


def close_greyscale_segment(image):
    return grey_closing(image)


def open_greyscale_segment(image):
    return grey_opening(image)


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


from scipy.ndimage import binary_closing, binary_opening, grey_closing, grey_opening, binary_fill_holes
import cv2
import numpy as np
from Settings import SettingsManager as sm
from ExperimentTools.MethodologyLogger import Logger

import ImageTools.ImageManager as im


def get_contours(image):
    max_contour_area = int(sm.configuration.get("MAXIMUM_BLOB_AREA"))

    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get contours
    contours_area = []
    for con in contours:
        area = cv2.contourArea(con)
        if area <= max_contour_area:
            contours_area.append(con)

    return contours_area


def remove_particles(image):
    fixed_image = np.zeros(image.shape, dtype=np.uint8)

    for i in range(3):
        segment = np.array([x == i for x in image], dtype=np.uint8) * 255
        segment -= np.array((fixed_image != 0), dtype=np.uint8) * 255

        contours_area = get_contours(segment)

        #debug_image = cv2.cvtColor(segment, cv2.COLOR_GRAY2BGR)
        #cv2.drawContours(debug_image, contours_area, -1, (0, 0, 255), 1)
        #cv2.imshow("Blobs Before", debug_image)

        # Remove white contours
        cv2.drawContours(segment, contours_area, -1, (0, 0, 0), -1)

        contours_area = get_contours(segment)
        # Remove black contours
        cv2.drawContours(segment, contours_area, -1, (255, 255, 255), -1)
        fixed_image += np.array([x == 255 for x in segment], dtype=np.uint8) * i

        #debug_image = cv2.cvtColor(segment, cv2.COLOR_GRAY2BGR)
        #cv2.drawContours(debug_image, contours_area, -1, (0, 0, 255), 1)
        #cv2.imshow("Blobs After", debug_image)

        #cv2.waitKey(0)

    #cv2.imshow("Fixed image", fixed_image * 127)
    #cv2.waitKey(0)
    return fixed_image


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


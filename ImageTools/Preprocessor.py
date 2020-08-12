import statistics

import cv2
import numpy as np
import matplotlib.pyplot as plt
import ImageTools.ImageManager as im
import Settings.FileManager as fm
import scipy.signal as ss

from scipy.ndimage import gaussian_filter, median_filter
from Settings.MessageTools import print_notice
from skimage.restoration import denoise_tv_chambolle
from tqdm import tqdm
from sklearn.preprocessing import binarize
from Settings import SettingsManager as sm, MessageTools as mt


def remove_empty_scans(images):
    print_notice("\tRemoving Empty Images... ", mt.MessagePrefix.INFORMATION, end='')
    threshold = 0.1 * len(images[0]) * len(images[0][0])
    valid_scans = list()

    for image in images:
        binaryimage = (0 < image) & (image < 100)
        image_sum = np.sum(binaryimage)

        if image_sum >= threshold:
            valid_scans.append(image)

    print("done!")
    return valid_scans


def normalise_images(images, pool):
    fixed_images = list()

    print_notice("\tNormalising Images... ", mt.MessagePrefix.INFORMATION, end='')
    for ind, res in enumerate(pool.map(normalise_image, images)):
        fixed_images.insert(ind, res)

    if sm.get_setting("ENABLE_IMAGE_SAVING") == "True":
        im.save_images(fixed_images, "Normalised", fm.SpecialFolder.SCAN_DATA, pool,
                       "Pre-Processed/Normalised/")

    print("done!")
    return fixed_images


def reshape_images(images, pool):
    reshaped_images = list()
    dimensions = statistics.mode([x.shape for x in images])

    print_notice("\tReshaping Images... ", mt.MessagePrefix.INFORMATION, end='')
    for ind, res in enumerate(pool.starmap(reshape_image, zip(images, [dimensions] * len(images)))):
        reshaped_images.insert(ind, res)

    print("done!")
    return reshaped_images


def reshape_image(image, dimensions):
    if image.shape == dimensions:
        return image

    reshaped = cv2.resize(image, dsize=(dimensions[1], dimensions[0]))

    return reshaped


def enhanced_contrast_images(images, pool):
    enhanced_images = list()

    print_notice("\tEnhancing Contrast... ", mt.MessagePrefix.INFORMATION, end='')
    for ind, res in enumerate(pool.map(enhance_contrast, images)):
        enhanced_images.insert(ind, res)

    print("done!")
    return enhanced_images


def enhance_contrast(image):
    return cv2.equalizeHist(image)


def normalise_image(image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(20, 20))

    return clahe.apply(image)


def denoise_images(images, pool):
    print_notice("\tDe-noising Images... ", mt.MessagePrefix.INFORMATION)
    print_notice("\t\tPerforming 3D Gaussian Blur... ", mt.MessagePrefix.INFORMATION, end='')
    gaussian_images = gaussian_filter(images, 2)

    if sm.get_setting("ENABLE_IMAGE_SAVING") == "True":
        im.save_images(gaussian_images, "Gaussian", fm.SpecialFolder.SCAN_DATA, pool,
                       "Pre-Processed/De-Noised/")
    print("done!")

    print_notice("\t\tPerforming 3D Median Blur... ", mt.MessagePrefix.INFORMATION, end='')
    fixed_images = median_filter(gaussian_images, 2)

    if sm.get_setting("ENABLE_IMAGE_SAVING") == "True":
        im.save_images(fixed_images, "Gaussian_Median", fm.SpecialFolder.SCAN_DATA, pool,
                       "Pre-Processed/De-Noised/")

    print("done!")
    return fixed_images


def denoise_image(image):
    img_denoise = denoise_tv_chambolle(image)
    return img_denoise


def remove_anomalies(images):
    fixed_images = list()
    print_notice("\tResolving X-Ray Intensity Anomalies... ", mt.MessagePrefix.INFORMATION, end='')

    for x in tqdm(range(len(images))):
        fixed_images.append(remove_anomaly(images[x]))

        if sm.get_setting("ENABLE_IMAGE_SAVING") == "True":
            im.save_image(fixed_images[x], str(x), "Pre-processing/AnomalyRemoved/", "AnomalyRemoved")

    print("done!")
    return fixed_images


def remove_backgrounds(images):
    fixed_images = list()
    print_notice("\tRemoving backgrounds (air-voids)... ", mt.MessagePrefix.INFORMATION, end='')

    for x in tqdm(range(len(images))):
        fixed_images.append(remove_background(images[x]))

        if sm.get_setting("ENABLE_IMAGE_SAVING") == "True":
            im.save_image(fixed_images[x], str(x), "Pre-processing/BackgroundRemoved/")

    print("done!")
    return fixed_images


def remove_background(image):
    image_array = np.squeeze(image, 2).astype(dtype=float)
    image_array = image_array.astype(dtype=int)

    if sm.get_setting("ENABLE_IMAGE_DISPLAY") == "True":
        content = image_array[np.nonzero(image_array)]
        min_val = np.min(content)
        max_val = np.max(content)
        hist, _ = np.histogram(image_array, bins=255, range=(min_val, max_val))
        hist = hist[np.nonzero(hist)]

        width_val = 2
        peaks, _ = ss.find_peaks(hist, width=width_val)

        black_value = peaks[1] + width_val / 2

        fig, axarr = plt.subplots(1, 3)

        anomaly_mask = image_array <= black_value

        axarr[0].imshow(image_array)
        axarr[0].set_title("Before")

        axarr[1].imshow(anomaly_mask)
        axarr[1].set_title("Anomaly Mask")

        plt.show()

        return np.expand_dims(anomaly_mask, 2)
    return None


def remove_anomaly(image):
    threshold = float(sm.get_setting("PREPROCESSING_BINARY_THRESHOLD"))
    image_array = np.squeeze(image, 2).astype(dtype=float)
    anomaly_mask = np.array(binarize(image_array, (1 - threshold) * np.amax(image))).astype(dtype=float)

    altered_image = (image_array - np.mean(image_array)/np.std(image_array))

    if sm.get_setting("ENABLE_IMAGE_DISPLAY") == "True":
        fig, axarr = plt.subplots(1, 3)

        axarr[0].imshow(image_array)
        axarr[0].set_title("Before")

        axarr[1].imshow(anomaly_mask)
        axarr[1].set_title("Anomaly Mask")

        axarr[2].imshow(altered_image)
        axarr[2].set_title("After")

        plt.show()
        plt.close(fig)

    return np.expand_dims(altered_image, 2)

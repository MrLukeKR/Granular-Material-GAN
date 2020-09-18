import statistics
import cv2
import numpy as np
import matplotlib.pyplot as plt
import ImageTools.ImageManager as im
import Settings.FileManager as fm

from skimage import exposure
from skimage.filters import difference_of_gaussians
from scipy.ndimage import gaussian_filter
from Settings.MessageTools import print_notice, get_notice
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

    print_notice("\tNormalising... ", mt.MessagePrefix.INFORMATION, end='')
    for ind, res in enumerate(pool.imap(normalise_image, images)):
        fixed_images.insert(ind, res)

    if sm.get_setting("ENABLE_IMAGE_SAVING") == "True":
        im.save_images(fixed_images, "Normalised", fm.SpecialFolder.SCAN_DATA, pool,
                       "Pre-Processed/Normalised/")

    print("done!")
    return fixed_images


def reshape_images(images, pool):
    reshaped_images = list()
    dimensions = statistics.mode([x.shape for x in images])

    for ind, res in tqdm(enumerate(pool.starmap(reshape_image, zip(images, [dimensions] * len(images)))),
                         desc=get_notice("\tReshaping"),
                         total=len(images)):
        reshaped_images.insert(ind, res)

    return reshaped_images


def reshape_image(image, dimensions):
    if image.shape == dimensions:
        return image

    reshaped = cv2.resize(image, dsize=(dimensions[1], dimensions[0]))

    return reshaped


def enhance_contrasts(images, pool):
    enhanced_images = list()

    for ind, res in tqdm(enumerate(pool.imap(enhance_contrast, images)),
                         desc=get_notice("\tEnhancing Contrast"),
                         total=len(images)):
        enhanced_images.insert(ind, res)

    return enhanced_images


def enhance_contrast(image):
    return exposure.equalize_adapthist(image, clip_limit=0.03)


def normalise_image(image):
    return np.array((image / np.max(image)) * 255, dtype=np.uint8)


def bandpass_filter_images(images, pool):
    filtered_images = list(tqdm(pool.imap(bandpass_filter, images),
                           desc=get_notice("\tBandpass Filtering"),
                           total=len(images)))

    return filtered_images


def bandpass_filter(image):
    filtered_image = difference_of_gaussians(image, 1, 12)

    return filtered_image


def denoise_images(images, pool):
    print_notice("\tDe-noising... ", mt.MessagePrefix.INFORMATION)
    print_notice("\t\tPerforming 3D Gaussian Blur... ", mt.MessagePrefix.INFORMATION, end='')
    gaussian_images = gaussian_filter(images, 3)

    if sm.get_setting("ENABLE_IMAGE_SAVING") == "True":
        im.save_images(gaussian_images, "Gaussian", fm.SpecialFolder.SCAN_DATA, pool,
                       "Pre-Processed/De-Noised/")
    print("done!")

    #print_notice("\t\tPerforming 3D Median Blur... ", mt.MessagePrefix.INFORMATION, end='')
    #fixed_images = median_filter(gaussian_images, 5)
    fixed_images = gaussian_images

    #if sm.get_setting("ENABLE_IMAGE_SAVING") == "True":
#        im.save_images(fixed_images, "Gaussian_Median", fm.SpecialFolder.SCAN_DATA, pool,
 #                      "Pre-Processed/De-Noised/")
#    print("done!")

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


def remove_backgrounds(images, pool):
    print_notice("\tRemoving backgrounds... ", mt.MessagePrefix.INFORMATION, end='')

    fixed_images = list(tqdm(pool.imap(remove_background, images)))

    print("done!")
    return fixed_images


def remove_background(image):
    raise NotImplemented


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

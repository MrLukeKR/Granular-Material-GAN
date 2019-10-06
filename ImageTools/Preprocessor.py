import numpy as np
import matplotlib.pyplot as plt
import ImageTools.ImageManager as im
import scipy.signal as ss


from tqdm import tqdm
from sklearn.preprocessing import binarize
from Settings import SettingsManager as sm
from skimage import exposure, restoration


def remove_empty_scans(images):
    print("\tRemoving Empty Images... ", end='')
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

    print("\tNormalising Images... ", end='')
    for ind, res in enumerate(pool.map(normalise_image, images)):
        fixed_images.insert(ind, res)

        if sm.configuration.get("ENABLE_IMAGE_SAVING") == "True":
            im.save_image(fixed_images[ind], str(ind), "Pre-processing/Normalised/")

    print("done!")
    return fixed_images


def reshape_images(images, pool):
    reshaped_images = list()

    print("\tReshaping Images... ", end='')
    for ind, res in enumerate(pool.map(reshape_image, images)):
        reshaped_images.insert(ind, res)

    print("done!")
    return reshaped_images


def reshape_image(image):
    shape = image.shape
    reshaped = np.reshape(image, (shape[0], shape[1]))

    return reshaped


def normalise_image(image):
    n_image = exposure.equalize_hist(image)

    p2, p98 = np.percentile(n_image, (2, 98))

    n_image = exposure.rescale_intensity(n_image, in_range=(p2, p98))
    return n_image


def denoise_images(images, pool):
    fixed_images = list()

    print("\tDe-noising Images... ", end='')
    for ind, res in enumerate(pool.map(denoise_image, images)):
        fixed_images.insert(ind, res)

        if sm.configuration.get("ENABLE_IMAGE_SAVING") == "True":
            im.save_image(res, str(ind), "Pre-processing/De-Noised/")

    print("done!")
    return fixed_images


def denoise_image(image):
    return restoration.denoise_tv_chambolle(image)


def remove_anomalies(images):
    fixed_images = list()
    print("\tResolving X-Ray Intensity Anomalies... ", end='')

    for x in tqdm(range(len(images))):
        fixed_images.append(remove_anomaly(images[x]))

        if sm.configuration.get("ENABLE_IMAGE_SAVING") == "True":
            im.save_image(fixed_images[x], str(x), "Pre-processing/AnomalyRemoved/")

    print("done!")
    return fixed_images


def remove_backgrounds(images):
    fixed_images = list()
    print("\tRemoving backgrounds (air-voids)", end='')

    for x in tqdm(range(len(images))):
        fixed_images.append(remove_background(images[x]))

        if sm.configuration.get("ENABLE_IMAGE_SAVING") == "True":
            im.save_image(fixed_images[x], str(x), "Pre-processing/BackgroundRemoved/")

    print("done!")
    return fixed_images


def remove_background(image):
    image_array = np.squeeze(image, 2).astype(dtype=float)
    #image_array *= 255
    image_array = image_array.astype(dtype=int)

    if sm.configuration.get("ENABLE_IMAGE_DISPLAY") == "True":
        content = image_array[np.nonzero(image_array)]
        min_val = np.min(content)
        max_val = np.max(content)
        hist, _ = np.histogram(image_array, bins=255, range=(min_val, max_val))
        hist = hist[np.nonzero(hist)]

        width_val = 2
        peaks, _ = ss.find_peaks(hist, width=width_val)
        # plt.plot(hist)
        # plt.plot(peaks, hist[peaks], marker='o')

        black_value = peaks[1] + width_val / 2

        fig, axarr = plt.subplots(1, 3)

        anomaly_mask = image_array <= black_value

        axarr[0].imshow(image_array)
        axarr[0].set_title("Before")

        axarr[1].imshow(anomaly_mask)
        axarr[1].set_title("Anomaly Mask")

        #axarr[2].imshow(altered_image)
        #axarr[2].set_title("After")


        plt.show()
        #plt.close(fig)

    return np.expand_dims(anomaly_mask, 2)


def remove_anomaly(image):
    threshold = float(sm.configuration.get("PREPROCESSING_BINARY_THRESHOLD"))
    image_array = np.squeeze(image, 2).astype(dtype=float)
    anomaly_mask = np.array(binarize(image_array, 1- threshold * np.amax(image))).astype(dtype=float)

    altered_image = (image_array - np.mean(image_array)/np.std(image_array))

    if sm.configuration.get("ENABLE_IMAGE_DISPLAY") == "True":
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

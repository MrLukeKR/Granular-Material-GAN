from DCGAN import DCGAN
from tensorflow.python.client import device_lib

# File I/O >>>
from glob import glob
from os import walk
# <<< File I/O

# Utilities >>>
import numpy as np

from tqdm import tqdm
from multiprocessing import Pool, cpu_count
# <<< Utilities

# Image Processing >>>
import ImageTools.Preprocessor as itp
import ImageTools.VoxelProcessor as vp
import ImageTools.ImageManager as im

from Settings import SettingsManager as sm
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
# <<< Image Processing

# Machine Learning >>>
import MachineLearningTools.MachineLearningManager as mlm
# <<< Machine Learning

print("   Optimal Material Generator using Generative Adversarial Networks   ")
print("                    Developed by ***REMOVED*** (BSc)                    ")
print("In fulfilment of Doctor of Engineering at the University of Nottingham")
print("----------------------------------------------------------------------")
print()
print("Running hardware checks...")
print(device_lib.list_local_devices())

pool = Pool()


def prepare_directories():
    data_directories = [f.replace('\\', '/')
                        for f in glob(sm.configuration.get("IO_DATA_ROOT_DIR") + "**/", recursive=True)]

    to_remove = set()

    for data_directory in data_directories:
        for dPaths, dNames, fNames in walk(data_directory):
            if len(fNames) == 0:
                to_remove.add(data_directory)

    for directory in to_remove:
        data_directories.remove(directory)

    return data_directories


def segment_images(cnn, images):
    segmented_ims = list()

    print("Segmenting images...")

    for ind in tqdm(range(len(images))):
        result = cnn.predict(np.expand_dims(images[ind], 0))
        segmented_ims.append(np.squeeze(result, 0))

    return segmented_ims


def segment_voxels(cnn, voxels):
    segmented_vox = list()

    print("Segmenting voxels...")

    for ind in tqdm(range(len(voxels))):
        result = cnn.predict(np.expand_dims(voxels[ind], 0))
        segmented_vox.append(np.squeeze(result, 0))

    return segmented_vox


def preprocess_image_collection(images):
    print("Pre-processing Image Collection...")
    processed_images = images

    processed_images = itp.normalise_images(processed_images, pool=pool)
    # processed_images = itp.denoise_images(processed_images)
    # processed_images = itp.remove_empty_scans(processed_images)
    # processed_images = itp.remove_anomalies(processed_images)
    # processed_images = itp.remove_backgrounds(processed_images)

    return processed_images


def process_voxels(images):
    voxels = list()

    if sm.configuration.get("ENABLE_VOXEL_SEPARATION") == "True":
        voxels = vp.split_to_voxels(images, int(sm.configuration.get("VOXEL_RESOLUTION")))

        if sm.configuration.get("ENABLE_VOXEL_INPUT_SAVING") == "True":
            im.save_voxel_images(voxels, "Unsegmented")
    return voxels


def main():
    sm.load_settings()
    data_directories = prepare_directories()

    for data_directory in data_directories:

# | DATA PREPARATION MODULE
# \-- | DATA LOADING SUB-MODULE
        images = im.load_images_from_directory(data_directory)
        sm.current_directory = data_directory.replace(sm.configuration.get("IO_DATA_ROOT_DIR"), '')

        if not sm.current_directory.endswith('/'):
            sm.current_directory += '/'

        if sm.configuration.get("ENABLE_PREPROCESSING") == "True":
            images = preprocess_image_collection(images)

        sm.images = images

# \-- | DATA REPRESENTATION CONVERSION SUB-MODULE
        voxels = process_voxels(images)

# \-- | DATA SEGMENTATION SUB-MODULE

        for ind, res in enumerate(pool.map(im.segment_vox, voxels)):
            continue

# | GENERATIVE ADVERSARIAL NETWORK MODULE
        DCGAN.initialise_network(images)

# \-- | 2D Noise Generation
        vX, vY, vC = images[0].shape
        noise = im.get_noise_image((vX, vY))
        im.show_image(noise)

# \-- | 3D Noise Generation
        # vX, vY, vZ, vC = voxels[0].shape
        # noise = get_noise_image((vX, vY, vZ))
        # im.display_voxel(noise)


if __name__ == "__main__":
    main()

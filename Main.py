import numpy as np

from DCGAN import DCGAN
from tensorflow.python.client import device_lib

from ImageTools import ImageManager

print("   Optimal Material Generator using Generative Adversarial Networks   ")
print("                    Developed by ***REMOVED*** (BSc)                    ")
print("In fulfilment of Doctor of Engineering at the University of Nottingham")
print("----------------------------------------------------------------------")
print()

print("Running hardware checks...")
print(device_lib.list_local_devices())

numepochs = 100

visdim = 50

# File I/O >>>
from glob import glob
from os import walk
# <<< File I/O

# Utilities >>>
import numpy as np


from tqdm import tqdm
# <<< Utilities

# Image Processing >>>
import ImageTools.Preprocessor as itp
import ImageTools.VoxelProcessor as vp
import ImageTools.ImageManager as im


from Settings import SettingsManager as sm
# <<< Image Processing

# Machine Learning >>>
# import MachineLearningTools.MachineLearningManager as mlm
# <<< Machine Learning


def main():
    sm.load_settings()

    data_directories = [f.replace('\\', '/')
                        for f in glob(sm.configuration.get("IO_DATA_ROOT_DIR") + "**/", recursive=True)]

    to_remove = set()

    for data_directory in data_directories:
        for dPaths, dNames, fNames in walk(data_directory):
            if len(fNames) == 0:
                to_remove.add(data_directory)

    for directory in to_remove:
        data_directories.remove(directory)

    for data_directory in data_directories:
        images = im.load_images_from_directory(data_directory)
        sm.current_directory = data_directory.replace(sm.configuration.get("IO_DATA_ROOT_DIR"), '')

        if not sm.current_directory.endswith('/'):
            sm.current_directory += '/'

        if sm.configuration.get("ENABLE_PREPROCESSING") == "True":
            images = preprocess_image_collection(images)

        voxels = list()

        if sm.segmentation_mode == "3D":
            if sm.configuration.get("ENABLE_VOXEL_SEPARATION") == "True":
                voxels = vp.split_to_voxels(images, int(sm.configuration.get("VOXEL_RESOLUTION")))

                if sm.configuration.get("ENABLE_VOXEL_INPUT_SAVING") == "True":
                    im.save_voxel_images(voxels, "Unsegmented")

#        if sm.configuration.get("ENABLE_SEGMENTATION") == "True":
#
#            if sm.segmentation_mode == "3D":
#                voxel_segmentation_net = mlm.train_network_3d(voxels)
#                segmented_voxels = segment_voxels(voxel_segmentation_net, voxels)
#
#                if sm.configuration.get("ENABLE_VOXEL_OUTPUT_SAVING") == "True":
#                    im.save_voxel_images(segmented_voxels, "Segmented")
#            elif sm.segmentation_mode == "2D":
#                image_segmentation_net = mlm.train_network_2d(images)
#                segmented_images = segment_images(image_segmentation_net, images)

        sm.images = images
        for v in voxels:
            ImageManager.display_voxel(v)
        # generateAnimation()


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

    processed_images = itp.normalise_images(processed_images)
    # processed_images = itp.denoise_images(processed_images)

    # processed_images = itp.remove_empty_scans(processed_images)

    # processed_images = itp.remove_anomalies(processed_images)

    # processed_images = itp.remove_backgrounds(processed_images)

    return processed_images


if __name__ == "__main__":
    main()


# Initialise GAN

# DCGAN.initialise_network()

#vX, vY, vZ = volumes.shape

#noise = np.random.normal(0, 1, size=(vX, vY, vZ)).astype(np.int)

#mlab.contour3d(noise, contours=2)
#mlab.show()

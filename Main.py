from GAN import DCGAN
from tensorflow.python.client import device_lib

# File I/O >>>
from glob import glob
from os import walk
# <<< File I/O

# Utilities >>>

from tqdm import tqdm
from multiprocessing import Pool
# <<< Utilities

# Image Processing >>>
import ImageTools.Preprocessor as itp
import ImageTools.VoxelProcessor as vp
import ImageTools.ImageManager as im

from ImageTools.Segmentation.ThreeDimensional import StackedOtsu2D as segmentor3D
from ImageTools.Segmentation.TwoDimensional import Otsu2D as segmentor2D

from Settings import SettingsManager as sm


# <<< Image Processing

# Machine Learning >>>
# <<< Machine Learning


pool = Pool()


def print_introduction():
    print("   Optimal Material Generator using Generative Adversarial Networks   ")
    print("                    Developed by ***REMOVED*** (BSc)                    ")
    print("In fulfilment of Doctor of Engineering at the University of Nottingham")
    print("----------------------------------------------------------------------")
    print()
    print("Running hardware checks...")
    print(device_lib.list_local_devices())


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


def preprocess_image_collection(images):
    print("Pre-processing Image Collection...")
    processed_images = images

    processed_images = itp.reshape_images(processed_images, pool=pool)
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
    print_introduction()

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

# \-- | 2D DATA SEGMENTATION SUB-MODULE
        print("Segmenting images... ")
        for i in tqdm(range(len(images))):
            segmentor2D.segment_image(images[i])


# \-- | DATA REPRESENTATION CONVERSION SUB-MODULE
        voxels = process_voxels(images)

# \-- | 3D DATA SEGMENTATION SUB-MODULE
#        print("Segmenting voxels... ", end='')
#        for v in tqdm(range(len(voxels))):
#            segmentor3D.segment_image(voxels[v])
#        print("done!")

# | GENERATIVE ADVERSARIAL NETWORK MODULE
        my_net = DCGAN.Network
        my_net.create_network(images)
        # my_net.train_network()

# \-- | 2D Noise Generation
        v_x, v_y = images[0].shape
        noise = im.get_noise_image((v_x, v_y))

# \-- | 3D Noise Generation
        # vX, vY, vZ = voxels[0].shape
        # noise = get_noise_image((vX, vY, vZ))
        # im.display_voxel(noise)


if __name__ == "__main__":
    main()

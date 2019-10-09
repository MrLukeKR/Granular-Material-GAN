from GAN import DCGAN
from tensorflow.python.client import device_lib

# File I/O >>>
from glob import glob
from os import walk
# <<< File I/O

# Utilities >>>

from tqdm import tqdm
from multiprocessing import Pool
import numpy as np

# <<< Utilities

# Image Processing >>>
import ImageTools.Preprocessor as itp
import ImageTools.VoxelProcessor as vp
import ImageTools.ImageManager as im

from ImageTools.Segmentation.ThreeDimensional import StackedOtsu2D as segmentor3D
from ImageTools.Segmentation.TwoDimensional import KMeans2D as segmentor2D

from Settings import SettingsManager as sm


# <<< Image Processing

# Machine Learning >>>
# <<< Machine Learning


pool = None


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
    processed_images = itp.denoise_images(processed_images, pool=pool)
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
    global pool

    pool = Pool()
    print_introduction()

    sm.load_settings()
    data_directories = prepare_directories()

    for data_directory in data_directories:

# | DATA PREPARATION MODULE
# \-- | DATA LOADING SUB-MODULE
        original_images = im.load_images_from_directory(data_directory)
        sm.current_directory = data_directory.replace(sm.configuration.get("IO_DATA_ROOT_DIR"), '')

        if not sm.current_directory.endswith('/'):
            sm.current_directory += '/'

        if sm.configuration.get("ENABLE_PREPROCESSING") == "True":
            images = preprocess_image_collection(original_images)
        else:
            images = original_images

        sm.images = images

#        ind = 0
#        for image in images:
#            im.save_image(image, str(ind), 'data/core/train/image/', False)
#            ind += 1

# \-- | 2D DATA SEGMENTATION SUB-MODULE
        voids = list()
        aggregates = list()
        binders = list()

        print("Segmenting images... ")
        for i in tqdm(range(len(images))):
            void, aggregate, binder, segment = segmentor2D.segment_image(images[i])
            voids.append(void)
            aggregates.append(aggregate)
            binders.append(binders)

            fig, ax = im.plt.subplots(2, 3, figsize=(10, 5))

            ax[0, 0].set_title("Original Image")
            ax[0, 0].axis('off')
            ax[0, 0].imshow(np.reshape(original_images[i], (1024, 1024)))

            ax[0, 1].axis('off')
            if sm.configuration.get("ENABLE_PREPROCESSING") == "True":
                ax[0, 1].set_title("Processed Image")
                ax[0, 1].imshow(np.reshape(images[i], (1024, 1024)))

            ax[0, 2].set_title("Segmented Image")
            ax[0, 2].axis('off')
            ax[0, 2].imshow(np.reshape(segment, (1024, 1024)))

            ax[1, 0].set_title("Voids")
            ax[1, 0].axis('off')
            ax[1, 0].imshow(np.reshape(void, (1024, 1024)))

            ax[1, 1].set_title("Binder")
            ax[1, 1].axis('off')
            ax[1, 1].imshow(np.reshape(binder, (1024, 1024)))

            ax[1, 2].set_title("Aggregates")
            ax[1, 2].axis('off')
            ax[1, 2].imshow(np.reshape(aggregate, (1024, 1024)))

            im.save_plot(str(i), 'segments/')

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

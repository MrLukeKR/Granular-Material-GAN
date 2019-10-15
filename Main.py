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
import ImageTools.Preprocessor as preproc
import ImageTools.Postprocessor as postproc
import ImageTools.VoxelProcessor as vp
import ImageTools.ImageManager as im

from ImageTools.Segmentation.ThreeDimensional import StackedOtsu2D as segmentor3D
from ImageTools.Segmentation.TwoDimensional import KMeans2D as segmentor2D

from Settings import SettingsManager as sm
from Settings import FileManager as fm


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


def preprocess_image_collection(images):
    print("Pre-processing Image Collection...")
    processed_images = images

    processed_images = preproc.reshape_images(processed_images, pool=pool)
    processed_images = preproc.normalise_images(processed_images, pool=pool)
    processed_images = preproc.denoise_images(processed_images, pool=pool)
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
    fm.assign_special_folders()

# | DATA PREPARATION MODULE
    fm.data_directories = fm.prepare_directories(True)

    if sm.configuration.get("ENABLE_PREPROCESSING") == "True":
        fm.data_directories = fm.prepare_directories(False)
        for data_directory in fm.data_directories:
            fm.current_directory = data_directory.replace(sm.configuration.get("IO_DATA_ROOT_DIR"), '')

            images = im.load_images_from_directory(data_directory)
            images = preprocess_image_collection(images)

            print("Saving processed images... ", end='')
            im.save_images(images, "scan", fm.SpecialFolder.PROCESSED_DATA)
            print("done!")

    fm.data_directories = fm.prepare_directories(True)

# \-- | DATA LOADING SUB-MODULE
    for data_directory in fm.data_directories:
        images = im.load_images_from_directory(data_directory)
        fm.current_directory = data_directory.replace(sm.configuration.get("IO_DATA_ROOT_DIR"), '')

        if not fm.current_directory.endswith('/'):
            fm.current_directory += '/'
        sm.images = images

#        ind = 0
#        for image in images:
#            im.save_image(image, str(ind), 'data/core/train/image/', False)
#            ind += 1

# \-- | 2D DATA SEGMENTATION SUB-MODULE
        voids = list()
        aggregates = list()
        binders = list()
        segments = list()

        print("Segmenting images... ", end="", flush=True)
        for ind, res in enumerate(pool.map(segmentor2D.segment_image, images)):
            void, aggregate, binder, segment = res

            voids.insert(ind, void)
            aggregates.insert(ind, aggregate)
            binders.insert(ind, binder)
            segments.insert(ind, segment)
        print("done!")

        print("Post-processing Segment Collection...")

        print("\tCleaning Voids...", end="", flush=True)
        for ind, res in enumerate(pool.map(postproc.clean_segment, voids)):
            voids.insert(ind, res)
        print("done!")

        print("\tCleaning Aggregates...", end="", flush=True)
        for ind, res in enumerate(pool.map(postproc.clean_segment, aggregates)):
            aggregates.insert(ind, res)
        print("done!")

        print("\tCleaning Binders...", end="", flush=True)
        for ind, res in enumerate(pool.map(postproc.clean_segment, binders)):
            binders.insert(ind, res)
        print("done!")

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


def save_segmentation_plots(images, segments, voids, binders, aggregates):
    print("Saving plots...")

    fig, ax = im.plt.subplots(2, 3, figsize=(10, 5))
    for i in tqdm(range(len(images))):
        ax[0, 0].axis('off')
        ax[0, 0].set_title("Original Image")
        ax[0, 0].imshow(np.reshape(images[i], (1024, 1024)))

        ax[0, 1].axis('off')
        if sm.configuration.get("ENABLE_PREPROCESSING") == "True":
            ax[0, 1].set_title("Processed Image")
            ax[0, 1].imshow(np.reshape(images[i], (1024, 1024)))

        ax[0, 2].set_title("Segmented Image")
        ax[0, 2].axis('off')
        ax[0, 2].imshow(np.reshape(segments[i], (1024, 1024)))

        ax[1, 0].set_title("Voids")
        ax[1, 0].axis('off')
        ax[1, 0].imshow(np.reshape(voids[i], (1024, 1024)))

        ax[1, 1].set_title("Binder")
        ax[1, 1].axis('off')
        ax[1, 1].imshow(np.reshape(binders[i], (1024, 1024)))

        ax[1, 2].set_title("Aggregates")
        ax[1, 2].axis('off')
        ax[1, 2].imshow(np.reshape(aggregates[i], (1024, 1024)))

        im.save_plot(str(i), 'segments/')


if __name__ == "__main__":
    main()

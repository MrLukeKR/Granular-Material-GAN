import itertools

import numpy as np
import matplotlib.pyplot as plt
import ImageTools.VoxelProcessor as vp
import ImageTools.Postprocessor as pop
import ImageTools.Preprocessor as prp
import cv2

from ImageTools.ImageSaver import save_image
from ImageTools.Segmentation.TwoDimensional import Otsu2D as segmentor2D
from multiprocessing import Pool
from os import walk
from matplotlib import cm
from tqdm import tqdm
from Settings import SettingsManager as sm, FileManager as fm, MessageTools as mt
from Settings.MessageTools import print_notice

pool = Pool()

global_voxels = None
global_comparison_voxels = None
directory = None
global_leftTitle = None
global_rightTitle = None

project_images = list()
segmentedImages = list()

supported_image_formats = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif')


def segment_images():
    existing_scans = set(fm.prepare_directories(fm.SpecialFolder.SEGMENTED_SCANS))
    existing_scans = list(map(lambda x: x.split('/')[-2], existing_scans))

    fm.data_directories = list(d for d in fm.prepare_directories(fm.SpecialFolder.PROCESSED_SCANS)
                               if d.split('/')[-2] not in existing_scans)

    for data_directory in fm.data_directories:
        images = load_images_from_directory(data_directory)
        fm.current_directory = data_directory.replace(fm.get_directory(fm.SpecialFolder.PROCESSED_SCANS), '')

        if not fm.current_directory.endswith('/'):
            fm.current_directory += '/'
        sm.images = images

        #        ind = 0
        #        for image in images:
        #            im.save_image(image, str(ind), 'data/core/train/image/', False)
        #            ind += 1

        # \-- | 2D DATA SEGMENTATION SUB-MODULE
        voids = list()
        clean_voids = list()

        aggregates = list()
        clean_aggregates = list()

        binders = list()
        clean_binders = list()

        segments = list()
        clean_segments = list()

        print_notice("Segmenting images... ", mt.MessagePrefix.INFORMATION, end="")

        for ind, res in enumerate(pool.map(segmentor2D.segment_image, images)):
        #for ind, res in enumerate(map(segmentor2D.segment_image, images)):
            voids.insert(ind, res[0])
            aggregates.insert(ind, res[1])
            binders.insert(ind, res[2])
            segments.insert(ind, res[3])

        print("done!")

        if sm.configuration.get("ENABLE_POSTPROCESSING"):
            print_notice("Post-processing Segment Collection...", mt.MessagePrefix.INFORMATION)

            print_notice("\tCleaning Aggregates... ", mt.MessagePrefix.INFORMATION, end="")
#            for ind, res in enumerate(pool.map(pop.fill_holes, aggregates)):
#                clean_aggregates.insert(ind, res)

            for ind, res in enumerate(pool.map(pop.open_close_segment, aggregates)):
                clean_aggregates.insert(ind, res)

            aggregates = clean_aggregates

            print("done!")

            print_notice("\tCleaning Binders... ", mt.MessagePrefix.INFORMATION, end="")
            for ind, res in enumerate(pool.map(pop.open_close_segment, binders)):
                clean_binders.insert(ind, res)

            binders = np.logical_and(clean_binders, np.logical_not(clean_aggregates))
            print("done!")

            print_notice("\tCleaning Voids... ", mt.MessagePrefix.INFORMATION, end="")
            for ind, res in enumerate(pool.map(pop.open_close_segment, voids)):
                clean_voids.insert(ind, res)

            voids = np.logical_and(clean_voids, np.logical_or(np.logical_not(clean_aggregates), np.logical_not(clean_binders)))
            print("done!")

            # print_notice("\tCleaning Segments...", mt.MessagePrefix.INFORMATION, end="")
            # for ind, res in enumerate(pool.map(pop.close_segment, segments)):
            #   clean_segments.insert(ind, res)
            # segments = clean_segments
            # print("done!")

        print_notice("Saving segmented images... ", mt.MessagePrefix.INFORMATION, end='')
        save_images(binders, "binder", fm.SpecialFolder.SEGMENTED_SCANS)
        save_images(aggregates, "aggregate", fm.SpecialFolder.SEGMENTED_SCANS)
        save_images(voids, "void", fm.SpecialFolder.SEGMENTED_SCANS)
        save_images(segments, "segment", fm.SpecialFolder.SEGMENTED_SCANS)
        print("done!")


def apply_preprocessing_pipeline(images):
    print_notice("Pre-processing Image Collection...", mt.MessagePrefix.INFORMATION)
    processed_images = images

    processed_images = prp.reshape_images(processed_images, pool=pool)
    processed_images = prp.enhanced_contrast_images(processed_images, pool=pool)
    processed_images = prp.normalise_images(processed_images, pool=pool)
    processed_images = prp.denoise_images(processed_images, pool=pool)
    # processed_images = itp.remove_empty_scans(processed_images)
    # processed_images = itp.remove_anomalies(processed_images)
    # processed_images = itp.remove_backgrounds(processed_images)

    return processed_images


def preprocess_images():
    existing_scans = set(fm.prepare_directories(fm.SpecialFolder.PROCESSED_SCANS))
    existing_scans = [x.split('/')[-2] for x in existing_scans]

    fm.data_directories = list(d for d in fm.prepare_directories(fm.SpecialFolder.UNPROCESSED_SCANS)
                               if d.split('/')[-2] not in existing_scans)

    for data_directory in fm.data_directories:
        fm.current_directory = data_directory.replace(fm.get_directory(fm.SpecialFolder.UNPROCESSED_SCANS), '')

        images = load_images_from_directory(data_directory)
        images = apply_preprocessing_pipeline(images)

        print_notice("Saving processed images... ", mt.MessagePrefix.INFORMATION, end='')
        save_images(images, "processed_scan", fm.SpecialFolder.PROCESSED_SCANS)
        print("done!")


def save_plot(filename, save_location, root_directory, use_current_directory):
    if not isinstance(root_directory, fm.SpecialFolder):
        raise TypeError("root_directory must be of enum type 'SpecialFolder'")

    directory = fm.root_directories[root_directory.value]
    if use_current_directory:
        directory += '/' + fm.current_directory
    directory += '/' + save_location

    fm.create_if_not_exists(directory)

    file_loc = directory + '/' + filename + '.' + sm.configuration.get("IO_IMAGE_FILETYPE")

    if not fm.file_exists(file_loc):
        if sm.USE_BW:
            plt.savefig(file_loc, cmap='gray', dpi=int(sm.configuration.get("IO_OUTPUT_DPI")))
        else:
            plt.savefig(file_loc, cmap='jet', dpi=int(sm.configuration.get("IO_OUTPUT_DPI")))


def save_images(images, filename_pretext, root_directory, save_location="", use_current_directory=True):
    image_count = len(images)
    arguments = zip(images, itertools.repeat(fm.get_directory(root_directory), image_count),
                    itertools.repeat(save_location + fm.current_directory, image_count),
                    itertools.repeat(filename_pretext, image_count), itertools.repeat(len(str(len(images))), image_count),
                    range(image_count), itertools.repeat(use_current_directory, image_count))
    list_arg = list(arguments)

    with tqdm(range(image_count), "Saving images '%s'" % filename_pretext) as bar:
        for ind, res in enumerate(pool.starmap(save_image, list_arg)):
            bar.update()


def save_segmentation_plots(images, segments, voids, binders, aggregates):
    print_notice("Saving plots...", mt.MessagePrefix.INFORMATION)

    fig, ax = plt.subplots(2, 3, figsize=(10, 5))
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

        save_plot(str(i), 'segments/')


def save_voxel_image(voxel, file_name, save_location):
    directory = sm.configuration.get("IO_ROOT_DIR") + fm.current_directory + save_location

    file_loc = directory + file_name + '.' + sm.configuration.get("IO_IMAGE_FILETYPE")

    fm.create_if_not_exists(directory)

    if fm.file_exists(file_loc):
        return

    fig = vp.plot_voxel(voxel)
    plt.savefig(file_loc)
    plt.close(fig)


def parallel_save(i):
    global global_voxels, global_comparison_voxels, directory, global_leftTitle, global_rightTitle

    buff_ind = '0' * (len(str(len(global_voxels))) - len(str(i))) + str(i)

    file_loc = directory + '/' + buff_ind + '.' + sm.configuration.get("IO_IMAGE_FILETYPE")

    voxel = global_voxels[i]

    if fm.file_exists(file_loc):
        return

    if global_comparison_voxels is None:
        fig = vp.plot_voxel(voxel)
    else:
        if len(voxel.shape) == 4:
            voxel = np.squeeze(voxel)

        comp_vox = global_comparison_voxels[i]
        if len(comp_vox.shape) == 4:
            comp_vox = np.squeeze(comp_vox)

        if sm.USE_BW:
            colours = cm.gray(voxel)
        else:
            colours = cm.jet(voxel)

        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.voxels(voxel, facecolors=colours, edgecolors=colours)
        if global_leftTitle is not None:
            ax.set_title(global_leftTitle)

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.voxels(comp_vox, facecolors=colours, edgecolors=colours)
        if global_rightTitle is not None:
           ax.set_title(global_rightTitle)

    plt.savefig(file_loc)
    plt.close(fig)


def save_voxel_image_collection(voxels, root_location, save_location="",
                                comparison_voxels=None, leftTitle=None, rightTitle=None):
    global global_voxels, global_comparison_voxels, directory, global_leftTitle, global_rightTitle

    global_voxels = voxels
    global_comparison_voxels = comparison_voxels
    global_leftTitle = leftTitle
    global_rightTitle = rightTitle

    if not isinstance(root_location, fm.SpecialFolder):
        raise TypeError("root_location must be of enum type 'SpecialFolder'")

    directory = fm.root_directories[root_location.value] + fm.current_directory + save_location

    print_notice("Saving " + str(len(voxels)) + " voxel visualisations to " + directory, mt.MessagePrefix.INFORMATION)

    fm.create_if_not_exists(directory)

    pool.map(parallel_save, range(len(voxels)))

    global_voxels = None
    global_comparison_voxels = None
    directory = None
    global_leftTitle = None
    global_rightTitle = None


def save_voxel_images(voxels, voxel_category="Unknown"):
    if sm.USE_BW:
        save_voxel_image_collection(voxels, "Results/VoxelImages/" + voxel_category + "/BW/")
    else:
        save_voxel_image_collection(voxels, "Results/VoxelImages/" + voxel_category + "/RGB/")


def show_image(array):
    image_dim = len(array)
    array = np.reshape(array, newshape=(image_dim, image_dim))

    fig = plt.figure()
    if sm.USE_BW:
        plt.imshow(array, interpolation='nearest', cmap='gray')
    else:
        plt.imshow(array, interpolation='nearest', cmap='jet')
    plt.show()

    plt.close(fig)

    # currim = Image.fromarray(array * 255.0)
    # currim.show()


def display_voxel(voxel):
    vp.plot_voxel(voxel)
    plt.show()


def load_images_from_list(file_list):
    print_notice("Loading " + str(len(file_list)) + " images", mt.MessagePrefix.INFORMATION)
    file_list.sort()
    ims = list()
    t = tqdm(range(len(file_list)))
    for i in t:  # tqdm is a progress bar tool
        if not file_list[i].lower().endswith(supported_image_formats):
            continue

        t.set_description("Loading: " + file_list[i])
        t.refresh()  # to show immediately the update
        # Number of images, channels, height, width

        img = cv2.imread(file_list[i])

        if img.size != (sm.image_resolution, sm.image_resolution):
            img = cv2.resize(img, (sm.image_resolution, sm.image_resolution))

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        ims.append(img)

    print()  # Print a new line after the process bar is finished

    if len(ims) > 0:
        print_notice("Loaded " + str(len(ims)) + " images successfully!", mt.MessagePrefix.INFORMATION)
    else:
        print_notice("No images were loaded!", mt.MessagePrefix.ERROR)

    return ims


def load_images_from_directory(directory, containing_keyword=None):
    files = []

    if not directory.endswith('/'):
        directory += '/'

    for (dPaths, dNames, fNames) in walk(directory):
        files.extend([directory + '{0}'.format(i) for i in fNames])

    if containing_keyword is not None:
        files = list(f for f in files if containing_keyword in f)

    return load_images_from_list(files)


def segment_vox(data):
    img = np.reshape(data, (64, 64, 64))

    return img

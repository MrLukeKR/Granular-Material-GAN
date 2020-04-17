import itertools

import numpy as np
import matplotlib.pyplot as plt
import ImageTools.VoxelProcessor as vp
import ImageTools.Postprocessor as pop
import ImageTools.Preprocessor as prp
import cv2

from ImageTools.ImageSaver import save_image
from ImageTools.Segmentation.TwoDimensional import Otsu2D as segmentor2D
from os import walk
from matplotlib import cm
from tqdm import tqdm
from Settings import SettingsManager as sm, FileManager as fm, MessageTools as mt
from Settings.MessageTools import print_notice

global_voxels = None
global_comparison_voxels = None
directory = None
global_leftTitle = None
global_rightTitle = None

project_images = list()
segmentedImages = list()

supported_image_formats = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif')


def segment_images(multiprocessing_pool, use_rois=True):
    print_notice("Starting Segmentation Phase...", mt.MessagePrefix.INFORMATION)

    existing_scans = set(fm.prepare_directories(fm.SpecialFolder.SEGMENTED_SCANS))
    existing_scans = list(map(lambda x: x.split('/')[-2], existing_scans))

    fm.data_directories = list(d for d in fm.prepare_directories(
        fm.SpecialFolder.ROI_SCANS if use_rois else fm.SpecialFolder.PROCESSED_SCANS)
                               if d.split('/')[-2] not in existing_scans)

    if len(fm.data_directories) == 0:
        print_notice("\tNothing to segment!", mt.MessagePrefix.INFORMATION)
        return

    for data_directory in fm.data_directories:
        images = load_images_from_directory(data_directory)
        fm.current_directory = data_directory.replace(fm.compile_directory(
            fm.SpecialFolder.ROI_SCANS if use_rois else fm.SpecialFolder.PROCESSED_SCANS), '')

        if not fm.current_directory.endswith('/'):
            fm.current_directory += '/'
        sm.images = images

        # \-- | 2D DATA SEGMENTATION SUB-MODULE
        segments = list()

        print_notice("Segmenting images... ", mt.MessagePrefix.INFORMATION, end="")

        for ind, res in enumerate(multiprocessing_pool.map(segmentor2D.segment_image, images)):
            segments.insert(ind, res)

        print("done!")

        if sm.configuration.get("ENABLE_POSTPROCESSING") == "True":
            print_notice("Post-processing Segment Collection...", mt.MessagePrefix.INFORMATION)
            clean_binders = list()
            aggregates = list()

            clean_segments = list()

            print_notice("\tRemoving Blobs... ", mt.MessagePrefix.INFORMATION, end="")
            # for ind, res in enumerate(multiprocessing_pool.map(pop.remove_particles, segments)):
            #     clean_segments.insert(ind, res)
            for i in range(len(segments)):
                clean_segments.insert(i, pop.remove_particles(segments[i]))

            print("done!")

            print_notice("\tCleaning Aggregates... ", mt.MessagePrefix.INFORMATION, end="")
            # for ind, res in enumerate(multiprocessing_pool.map(pop.fill_holes, [x == 2 for x in segments])):
            #     clean_aggregates.insert(ind, res)

            for ind, res in enumerate(multiprocessing_pool.map(pop.open_close_segment, [x == 2 for x in clean_segments])):
                aggregates.insert(ind, res)

            print("done!")

            print_notice("\tCleaning Binders... ", mt.MessagePrefix.INFORMATION, end="")
            for ind, res in enumerate(multiprocessing_pool.map(pop.open_close_segment, [x == 1 for x in clean_segments])):
                clean_binders.insert(ind, res)

            binders = np.logical_and(clean_binders, np.logical_not(aggregates))
            print("done!")

            segments = list()
            print_notice("\tRecombining Segments...", mt.MessagePrefix.INFORMATION, end="")
            for i in range(len(aggregates)):
                clean_segment = aggregates[i] * 255 + (binders[i] * 127)
                segments.insert(i, clean_segment)

            print("done!")

        print_notice("Saving segmented images... ", mt.MessagePrefix.INFORMATION, end='')
        #save_images(binders, "binder", fm.SpecialFolder.SEGMENTED_SCANS, multiprocessing_pool)
        #save_images(aggregates, "aggregate", fm.SpecialFolder.SEGMENTED_SCANS, multiprocessing_pool)
        #save_images(voids, "void", fm.SpecialFolder.SEGMENTED_SCANS, multiprocessing_pool)
        save_images(segments, "segment", fm.SpecialFolder.SEGMENTED_SCANS, multiprocessing_pool)
        print("done!")


def apply_preprocessing_pipeline(images, multiprocessing_pool):
    print_notice("Pre-processing Image Collection...", mt.MessagePrefix.INFORMATION)
    processed_images = images

    processed_images = prp.reshape_images(processed_images, pool=multiprocessing_pool)
    processed_images = prp.enhanced_contrast_images(processed_images, pool=multiprocessing_pool)
    processed_images = prp.normalise_images(processed_images, pool=multiprocessing_pool)
    processed_images = prp.denoise_images(processed_images, pool=multiprocessing_pool)
    # processed_images = itp.remove_empty_scans(processed_images)
    # processed_images = itp.remove_anomalies(processed_images)
    # processed_images = itp.remove_backgrounds(processed_images)

    return processed_images


def extract_rois(multiprocessing_pool, use_segmented=False):
    print_notice("Beginning Region of Interest Extraction Phase...", mt.MessagePrefix.INFORMATION)

    existing_scans = set(fm.prepare_directories(fm.SpecialFolder.ROI_SCANS))
    existing_scans = [x.split('/')[-2] for x in existing_scans]

    if use_segmented:
        fm.data_directories = list(d for d in fm.prepare_directories(fm.SpecialFolder.SEGMENTED_SCANS)
                                   if d.split('/')[-2] not in existing_scans)
    else:
        fm.data_directories = list(d for d in fm.prepare_directories(fm.SpecialFolder.PROCESSED_SCANS)
                                   if d.split('/')[-2] not in existing_scans)

    if len(fm.data_directories) == 0:
        print_notice("\tNothing to extract!", mt.MessagePrefix.INFORMATION)
        return

    for data_directory in fm.data_directories:
        if use_segmented:
            fm.current_directory = data_directory.replace(fm.compile_directory(fm.SpecialFolder.SEGMENTED_SCANS), '')
        else:
            fm.current_directory = data_directory.replace(fm.compile_directory(fm.SpecialFolder.PROCESSED_SCANS), '')

        images = load_images_from_directory(data_directory)
        images = extract_roi(np.array(images))

        print_notice("Saving Region of Interest (ROI) images... ", mt.MessagePrefix.INFORMATION, end='')
        save_images(images, "roi", fm.SpecialFolder.ROI_SCANS, multiprocessing_pool)
        print("done!")


def extract_roi(core):
    if isinstance(core, list):
        core = np.array(core)

    roi_image_metric = sm.configuration.get("ROI_IMAGE_METRIC")
    roi_depth_metric = sm.configuration.get("ROI_DEPTH_METRIC")

    if roi_image_metric == "PERCENTAGE":
        roi_percentages = [float(x) / 100 for x in sm.configuration.get("ROI_IMAGE_DIMENSIONS").split(',')]
        roi_size = (int(roi_percentages[0] * core.shape[1]),
                    int(roi_percentages[1] * core.shape[2]))
    elif roi_image_metric == "PIXELS":
        roi_size = tuple(map(int, sm.configuration.get("ROI_IMAGE_DIMENSIONS").split(',')))
    elif roi_image_metric == "MILLIMETRES":
        raise NotImplementedError
    else:
        print_notice("Invalid value for ROI Image Metric", mt.MessagePrefix.ERROR)
        raise ValueError

    if roi_depth_metric == "ABSOLUTE":
        roi_size = (int(sm.configuration.get("ROI_DEPTH_DIMENSION")), roi_size[0], roi_size[1])
    elif roi_depth_metric == "PERCENTAGE":
        roi_percentage = float(sm.configuration.get("ROI_DEPTH_DIMENSION")) / 100
        roi_size = (int(roi_percentage * core.shape[0]), roi_size[0], roi_size[1])
    else:
        print_notice("Invalid value for ROI Depth Metric", mt.MessagePrefix.ERROR)
        raise ValueError

    start_pos = [0, 0, 0]
    end_pos = [0, 0, 0]

    centre_points = tuple(map(lambda x: x // 2, core.shape))

    for i in range(3):
        start_pos[i] = centre_points[i] - (roi_size[i] // 2)
        end_pos[i] = centre_points[i] + (roi_size[i] // 2)

    roi = core[start_pos[0]:end_pos[0], start_pos[1]:end_pos[1], start_pos[2]:end_pos[2]]

    return roi


def preprocess_images(multiprocessing_pool):
    print_notice("Beginning Preprocessing Phase...", mt.MessagePrefix.INFORMATION)

    existing_scans = set(fm.prepare_directories(fm.SpecialFolder.PROCESSED_SCANS))
    existing_scans = [x.split('/')[-2] for x in existing_scans]

    fm.data_directories = list(d for d in fm.prepare_directories(fm.SpecialFolder.UNPROCESSED_SCANS)
                               if d.split('/')[-2] not in existing_scans)

    if len(fm.data_directories) == 0:
        print_notice("\tNothing to preprocess!", mt.MessagePrefix.INFORMATION)
        return

    for data_directory in fm.data_directories:
        fm.current_directory = data_directory.replace(fm.compile_directory(fm.SpecialFolder.UNPROCESSED_SCANS), '')

        images = load_images_from_directory(data_directory)
        images = apply_preprocessing_pipeline(images, multiprocessing_pool)

        print_notice("Saving processed images... ", mt.MessagePrefix.INFORMATION, end='')
        save_images(images, "processed_scan", fm.SpecialFolder.PROCESSED_SCANS, multiprocessing_pool)
        print("done!")


def save_plot(filename, save_location, root_directory, use_current_directory):
    if not isinstance(root_directory, fm.SpecialFolder):
        raise TypeError("root_directory must be of enum type 'SpecialFolder'")

    plot_directory = fm.root_directories[root_directory.value]
    if use_current_directory:
        plot_directory += '/' + fm.current_directory
    plot_directory += '/' + save_location

    fm.create_if_not_exists(plot_directory)

    file_loc = plot_directory + '/' + filename + '.' + sm.configuration.get("IO_IMAGE_FILETYPE")

    if not fm.file_exists(file_loc):
        if sm.USE_BW:
            plt.savefig(file_loc, cmap='gray', dpi=int(sm.configuration.get("IO_OUTPUT_DPI")))
        else:
            plt.savefig(file_loc, cmap='jet', dpi=int(sm.configuration.get("IO_OUTPUT_DPI")))


def save_images(images, filename_pretext, root_directory, multiprocessing_pool, save_location="", use_current_directory=True):
    image_count = len(images)
    arguments = zip(images, itertools.repeat(fm.compile_directory(root_directory), image_count),
                    itertools.repeat(save_location + fm.current_directory, image_count),
                    itertools.repeat(filename_pretext, image_count), itertools.repeat(len(str(len(images))), image_count),
                    range(image_count), itertools.repeat(use_current_directory, image_count))
    list_arg = list(arguments)

    tqdm(multiprocessing_pool.starmap(save_image, list_arg), "Saving images...")


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
    image_directory = sm.configuration.get("IO_ROOT_DIR") + fm.current_directory + save_location

    file_loc = image_directory + file_name + '.' + sm.configuration.get("IO_IMAGE_FILETYPE")

    fm.create_if_not_exists(image_directory)

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


def save_voxel_image_collection(voxels, root_location, multiprocessing_pool, save_location="",
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

    multiprocessing_pool.map(parallel_save, range(len(voxels)))

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

        #if img.size != (sm.image_resolution, sm.image_resolution):
#            img = cv2.resize(img, (sm.image_resolution, sm.image_resolution))

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        ims.append(img)

    print()  # Print a new line after the process bar is finished

    if len(ims) > 0:
        print_notice("Loaded " + str(len(ims)) + " images successfully!", mt.MessagePrefix.INFORMATION)
    else:
        print_notice("No images were loaded!", mt.MessagePrefix.ERROR)

    return ims


def load_images_from_directory(image_directory, containing_keyword=None):
    files = []

    if not image_directory.endswith('/'):
        image_directory += '/'

    for (dPaths, dNames, fNames) in walk(image_directory):
        files.extend([image_directory + '{0}'.format(i) for i in fNames])

    if containing_keyword is not None:
        files = list(f for f in files if containing_keyword in f)

    return load_images_from_list(files)


def segment_vox(data):
    img = np.reshape(data, (64, 64, 64))

    return img

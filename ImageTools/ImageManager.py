import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import ImageTools.VoxelProcessor as vp

from multiprocessing import Pool
from ExperimentTools.MethodologyLogger import Logger
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

pool = Pool()

project_images = list()
segmentedImages = list()

supported_image_formats = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif')


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
    ind = 0

    digits = len(str(len(images)))

    for image in images:
        preamble_digits = digits - len(str(ind))
        preamble = '0' * preamble_digits
        save_image(image, root_directory, save_location, filename_pretext + '_' + preamble + str(ind), use_current_directory)
        ind += 1


def save_image(image, root_directory, save_location, filename, use_current_directory=True):
    if not isinstance(root_directory, fm.SpecialFolder):
        raise TypeError("root_directory must be of enum type 'SpecialFolder'")

    directory = fm.root_directories[root_directory.value]

    if len(save_location) > 0:
        directory += save_location

    if use_current_directory:
        directory += fm.current_directory

    fm.create_if_not_exists(directory)

    file_loc = directory + filename + '.' + sm.configuration.get("IO_IMAGE_FILETYPE")

    if not fm.file_exists(file_loc):
        if len(image.shape) != 2:
            image = np.squeeze(image, 2)
        if sm.USE_BW:
            plt.imsave(file_loc, image, cmap='gray')
        else:
            plt.imsave(file_loc, image, cmap='jet')


def save_segmentation_plots(images, segments, voids, binders, aggregates):
    Logger.print("Saving plots...")

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
    Logger.print("Loading " + str(len(file_list)) + " images")
    file_list.sort()
    ims = list()
    t = tqdm(range(len(file_list)))
    for i in t:  # tqdm is a progress bar tool
        if not file_list[i].lower().endswith(supported_image_formats):
            continue

        t.set_description("Loading: " + file_list[i])
        t.refresh()  # to show immediately the update
        # Number of images, channels, height, width

        img = Image.open(file_list[i])

        if img.size != (sm.image_resolution, sm.image_resolution):
            img = img.resize((sm.image_resolution, sm.image_resolution))

        if "RGB" in img.mode:
            r = None
            g = None
            b = None

            if img.mode == "RGB":
                r, g, b = img.split()
            elif img.mode == "RGBA":
                r, g, b, _ = img.split()

            ra = np.array(r)
            ga = np.array(g)
            ba = np.array(b)

            img = (0.299 * ra + 0.587 * ga + 0.114 * ba)

        img = np.uint8(img / np.max(img) * 255.0)

        ims.append(img)

    Logger.print()  # Print a new line after the process bar is finished

    if len(ims) > 0:
        Logger.print("Loaded " + str(len(ims)) + " images successfully!")
    else:
        Logger.print("ERROR: No images were loaded!")

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

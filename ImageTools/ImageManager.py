import os
import math
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import ImageTools.VoxelProcessor as vp
import MachineLearningTools.MachineLearningManager as mlm


from os import walk
from tqdm import tqdm
from Settings import SettingsManager as sm
from Settings import FileManager as fm
from enum import Enum

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

    if not os.path.exists(directory):
        os.makedirs(directory)

    file_loc = directory + '/' + filename + '.' + sm.configuration.get("IO_IMAGE_FILETYPE")

    if not os.path.isfile(file_loc):
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

    if not os.path.exists(directory):
        os.makedirs(directory)

    file_loc = directory + filename + '.' + sm.configuration.get("IO_IMAGE_FILETYPE")

    if not os.path.isfile(file_loc):
        if len(image.shape) != 2:
            image = np.squeeze(image, 2)
        if sm.USE_BW:
            plt.imsave(file_loc, image, cmap='gray')
        else:
            plt.imsave(file_loc, image, cmap='jet')


def save_segmentation_plots(images, segments, voids, binders, aggregates):
    print("Saving plots...")

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
    directory = sm.configuration.get("IO_OUTPUT_ROOT_DIR") + fm.current_directory + save_location

    file_loc = directory + file_name + '.' + sm.configuration.get("IO_IMAGE_FILETYPE")

    if not os.path.exists(directory):
        os.makedirs(directory)

    if os.path.isfile(file_loc):
        return

    fig = vp.plot_voxel(voxel)
    plt.savefig(file_loc)
    plt.close(fig)


def save_voxel_image_collection(voxels, save_location):
    print("Saving " + str(len(voxels)) + " voxel visualisations")
    directory = sm.configuration.get("IO_OUTPUT_ROOT_DIR") + fm.current_directory + save_location

    print(directory)

    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in tqdm(range(len(voxels))):
        file_loc = directory + str(i) + '.' + sm.configuration.get("IO_IMAGE_FILETYPE")

        if os.path.isfile(file_loc):
            continue

        fig = vp.plot_voxel(voxels[i])
        plt.savefig(file_loc)
        plt.close(fig)


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


def generate_animation(images):
    ims = []

    for img in images:
        ims.append([plt.imshow(np.reshape(img, newshape=(1024, 1024)))])
    fig = plt.figure()

    return anim.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)


def save_animation(animation, save_location, frames_per_second):
    animation.save(save_location, fps=frames_per_second, extra_args=['-vcodec', 'libx264'])


def load_images_from_list(file_list):
    print("Loading " + str(len(file_list)) + " images")
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

        if img.mode == "RGB":
            r, g, b = img.split()
            ra = np.array(r)
            ga = np.array(g)
            ba = np.array(b)

            img = (0.299 * ra + 0.587 * ga + 0.114 * ba)

        img = np.uint8(img / np.max(img) * 255.0)

        ims.append(img)

    print()  # Print a new line after the process bar is finished

    if len(ims) > 0:
        print("Loaded " + str(len(ims)) + " images successfully!")
    else:
        print("ERROR: No images were loaded!")

    return ims


def load_images_from_directory(directory):
    files = []

    if not directory.endswith('/'):
        directory += '/'

    for (dPaths, dNames, fNames) in walk(directory):
            files.extend([directory + '{0}'.format(i) for i in fNames])

    return load_images_from_list(files)


def get_noise_image(shape):
    noise = np.random.normal(0, 1, size=shape)
    noise = np.array(noise > 0).astype(np.uint8)
    noise = noise.reshape(shape)

    return noise


def segment_vox(data):
    img = np.reshape(data, (64, 64, 64))

    return img

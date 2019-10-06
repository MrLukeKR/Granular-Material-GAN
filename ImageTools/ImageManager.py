import os
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import ImageTools.VoxelProcessor as vp
import MachineLearningTools.MachineLearningManager as mlm


from os import walk
from tqdm import tqdm
from Settings import SettingsManager as sm


project_images = list()
segmentedImages = list()


def save_plot(filename, save_location):
    directory = sm.configuration.get("IO_OUTPUT_ROOT_DIR") + sm.current_directory + save_location

    if not os.path.exists(directory):
        os.makedirs(directory)

    file_loc = directory + filename + '.' + sm.configuration.get("IO_IMAGE_FILETYPE")

    if not os.path.isfile(file_loc):
        if sm.USE_BW:
            plt.savefig(file_loc, cmap='gray')
        else:
            plt.savefig(file_loc, cmap='jet')


def save_image(image, filename, save_location, use_global_save_location=True):
    directory = save_location

    if use_global_save_location:
        directory = directory + sm.configuration.get("IO_OUTPUT_ROOT_DIR") + sm.current_directory

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


def save_voxel_image(voxel, file_name, save_location):
    directory = sm.configuration.get("IO_OUTPUT_ROOT_DIR") + sm.current_directory + save_location

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
    directory = sm.configuration.get("IO_OUTPUT_ROOT_DIR") + sm.current_directory + save_location

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
    ims = list()
    t = tqdm(range(len(file_list)))
    for i in t:  # tqdm is a progress bar tool
        t.set_description("Loading: " + file_list[i])
        t.refresh()  # to show immediately the update
        # Number of images, channels, height, width
        img = Image.open(file_list[i])

        img = img.resize((sm.image_resolution, sm.image_resolution))
        img = np.asarray(img, dtype=mlm.K.floatx())

        img = np.uint8(img / img.max() * 255.0)
        img = np.reshape(img, (sm.image_resolution, sm.image_resolution, 1))
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

    files.sort()

    return load_images_from_list(files)


def get_noise_image(shape):
    noise = np.random.normal(0, 1, size=shape)
    noise = np.array(noise > 0).astype(np.uint8)
    noise = noise.reshape(shape)

    return noise


def segment_vox(data):
    img = np.reshape(data, (64, 64, 64))

    return img

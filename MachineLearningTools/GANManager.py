import os
import numpy as np
import MachineLearningTools.ML3D.UNet3D as UNet3D
import MachineLearningTools.ML2D.UNet2D as UNet2D
import Settings.SettingsManager as sm

from tqdm import tqdm
from MachineLearningTools.ML3D.ImageDataGenerator3D import ImageDataGenerator3D
from ExperimentTools.MethodologyLogger import Logger
from Settings import FileManager as fm


def train_network_2d(images):
    x_train = list()
    im_res = len(images[0])
    ind = 0

    for image in images:
        array_vox = np.array(image)
        if array_vox.shape != (im_res, im_res, sm.image_channels):
            Logger.print("Incorrect shape in [IMAGE " + str(ind) + "]: " + str(image.shape))
        else:
            x_train.append(array_vox)
        ind += 1

    x_train = np.array(x_train, dtype=float)

    Logger.print("Input dataset shape: " + str(x_train.shape))

    Logger.print("Saving training data to Numpy files (for memory efficient data usage)...")

    data_indices = list(range(len(x_train)))

    data_directory = "Data/" + fm.current_directory

    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    for ind in tqdm(data_indices):
        np.save(data_directory + str(ind), x_train[ind])

    datagen = ImageDataGenerator3D(data_indices, x_train, n_classes=3,
                                   batch_size=int(sm.configuration.get("TRAINING_BATCH_SIZE")),
                                   dim=(im_res, im_res))

    datagen.data_directory = data_directory
    config = sm.configuration

    encoder = UNet2D.create_unet(int(config.get("IMAGE_RESOLUTION")),
                                 int(config.get("IMAGE_CHANNELS")),
                                 int(config.get("LABEL_SEGMENTS")),
                                 config.get("ENABLE_BATCH_NORMALISATION") == 'True')
    # decoder = UNet.CreateUNet(voxel_resolution, image_channels)

    wnet = encoder

    # wnet.fit_generator(generator=datagen, epochs=10, steps_per_epoch=500, verbose=1)
    wnet.fit(x=x_train, y=x_train, batch_size=int(sm.configuration.get("TRAINING_BATCH_SIZE")), epochs=int(sm.configuration.get("TRAINING_EPOCHS")), verbose=1)

    return wnet


def train_network_3d(voxels):
    x_train = list()
    vox_res = len(voxels[0])
    ind = 0

    for voxel in voxels:
        array_vox = np.array(voxel)
        if array_vox.shape != (vox_res, vox_res, vox_res, sm.image_channels):
            Logger.print("Incorrect shape in [VOXEL " + str(ind) + "]: " + str(voxel.shape))
        else:
            x_train.append(array_vox)
        ind += 1

    x_train = np.array(x_train, dtype=float)

    Logger.print("Input dataset shape: " + str(x_train.shape))

    Logger.print("Saving training data to Numpy files (for memory efficient data usage)...")

    data_indices = list(range(len(x_train)))

    data_directory = "Data/" + fm.current_directory

    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    for ind in tqdm(data_indices):
        np.save(data_directory + str(ind), x_train[ind])

    datagen = ImageDataGenerator3D(data_indices, x_train, n_classes=3,
                                   batch_size=int(sm.configuration.get("TRAINING_BATCH_SIZE")),
                                   dim=(vox_res, vox_res, vox_res))

    datagen.data_directory = data_directory
    config = sm.configuration

    encoder = UNet3D.create_unet(
        int(config.get("VOXEL_RESOLUTION")),
        int(config.get("IMAGE_CHANNELS")),
        int(config.get("LABEL_SEGMENTS")),
        config.get("ENABLE_BATCH_NORMALISATION") == 'True'
    )
    # decoder = UNet.CreateUNet(voxel_resolution, image_channels)

    wnet = encoder

    # wnet.fit_generator(generator=datagen, epochs=10, steps_per_epoch=500, verbose=1)
    wnet.fit(x=x_train, y=x_train, batch_size=int(sm.configuration.get("TRAINING_BATCH_SIZE")), epochs=1, verbose=1)

    return wnet

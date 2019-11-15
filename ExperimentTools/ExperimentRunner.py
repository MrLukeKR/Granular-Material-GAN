from ExperimentTools import DatasetProcessor, MethodologyLogger
from GAN import DCGAN
from Settings import FileManager as fm, SettingsManager as sm
from ImageTools import VoxelProcessor as vp, ImageManager as im
from ExperimentTools.MethodologyLogger import Logger

import numpy as np


def run_train_test_split_experiment(aggregates, binders, split_percentage):

    pass


def run_k_fold_cross_validation_experiment(dataset_directories, k):
    data_length = len(dataset_directories)

    if k > data_length:
        print("Dataset of size [" + str(data_length) + "] is too small for [" + str(k) + "] folds.")
        print("Setting k to [" + str(data_length) + "]")

        k = min(k, data_length)

    training_sets, testing_sets = DatasetProcessor.dataset_to_k_cross_fold(dataset_directories, k)

    epochs = 5000
    batch_size = 32

    vox_res = int(sm.configuration.get("VOXEL_RESOLUTION"))
    dummy = np.zeros(shape=(1, vox_res, vox_res, vox_res, sm.image_channels))

    for fold in range(k):
        Logger.print("Running Cross Validation Fold " + str(fold + 1) + "/" + str(k))

        fold_d_losses = np.zeros((epochs * len(training_sets)))
        fold_g_losses = np.zeros((epochs * len(training_sets)))

        discriminator = DCGAN.DCGANDiscriminator(dummy, 2, 5)
        generator = DCGAN.DCGANGenerator(dummy, 2, 5)

        DCGAN.Network._discriminator = discriminator.model
        DCGAN.Network._generator = generator.model

        DCGAN.Network.create_network(dummy)

        ind = 0
        for training_set in training_sets[fold]:
            aggregates = list()
            binders = list()

            for directory in training_set:
                Logger.print("\tLoading voxels from " + directory + "... ", end='')
                fm.current_directory = directory.replace(fm.get_directory(fm.SpecialFolder.SEGMENTED_SCANS), '')

                voxel_directory = fm.get_directory(fm.SpecialFolder.VOXEL_DATA) + fm.current_directory[0:-1]

                temp_aggregates = vp.load_voxels(voxel_directory, "aggregate_" + sm.configuration.get("VOXEL_RESOLUTION"))
                temp_binders = vp.load_voxels(voxel_directory, "binder_" + sm.configuration.get("VOXEL_RESOLUTION"))

                for aggregate in temp_aggregates:
                    if not aggregate.max == 0 and not aggregate.min == 0:
                        aggregates.append(aggregate * 255)

                for binder in temp_binders:
                    if not binder.max == 0 and not binder.min == 0:
                        binders.append(binder * 255)

                Logger.print("done!")

            # im.save_voxel_image_collection(aggregates[10:15], fm.SpecialFolder.VOXEL_DATA, "figures/PostH5/aggregate")
            # im.save_voxel_image_collection(binders[10:15], fm.SpecialFolder.VOXEL_DATA, "figures/PostH5/binder")

            Logger.print("\tTraining on set " + str(ind + 1) + '/' + str(len(training_sets)) + "... ")
            d_loss, g_loss, images = DCGAN.Network.train_network(epochs, batch_size, aggregates, binders)

            fold_d_losses[ind * epochs: (ind + 1) * epochs] = np.squeeze(d_loss)
            fold_g_losses[ind * epochs: (ind + 1) * epochs] = np.squeeze(g_loss)

            ind += 1

        fig, ax = im.plt.subplots()
        ax.plot(np.array(fold_d_losses), label="Discriminator Loss")
        ax.plot(np.array(fold_g_losses), label="Generator Loss")
        ax.set(xlabel="Epochs", ylabel="Loss",
               title="Training Loss")
        ax.tick_params(axis='x', which="both", bottom=False)
        ax.legend(loc="upper right")

        directory = fm.get_directory(fm.SpecialFolder.RESULTS) + "/Figures"
        fm.create_if_not_exists(directory)
        fig.savefig(directory + '/training_' + MethodologyLogger.Logger.get_timestamp() + '.jpg')

        testing_set = testing_sets[fold]


def run_on_existing_gan(aggregates, binders):
    # discriminator, generator = mlm.load_network()
    #
    # if discriminator is None or generator is None:
    #     discriminator = DCGAN.DCGANDiscriminator(aggregates, 2, 5)
    #     generator = DCGAN.DCGANGenerator(aggregates, 2, 5)
    #
    # DCGAN.Network._discriminator = discriminator
    # DCGAN.Network._generator = generator
    #
    # network = DCGAN.Network.create_network(aggregates)
    #
    # batch_size = 32
    #
    # DCGAN.Network.train_network(int(len(aggregates) / batch_size), batch_size, aggregates, binders)
    pass

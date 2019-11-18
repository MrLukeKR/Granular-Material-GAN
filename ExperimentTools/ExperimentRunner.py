from ExperimentTools import DatasetProcessor, MethodologyLogger
from GAN import DCGAN
from Settings import FileManager as fm, SettingsManager as sm, MachineLearningManager as mlm
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

    epochs = 500
    batch_size = 32

    vox_res = int(sm.configuration.get("VOXEL_RESOLUTION"))
    dummy = np.zeros(shape=(1, vox_res, vox_res, vox_res, sm.image_channels))

    gen_filters = 128
    gen_activation_alpha = 0.2
    gen_normalisation_momentum = 0.8
    gen_levels = 3
    gen_strides = 2
    gen_kernel_size = 5

    dis_filters = 32
    dis_activation_alpha = 0.2
    dis_normalisation_momentum = 0.8
    dis_levels = 3
    dis_strides = 2
    dis_kernel_size = 5

    sql = "INSERT INTO experiment_settings (ExperimentID, NetworkType, Folds, Epochs, BatchSize, " \
          "GeneratorStrides, GeneratorKernelSize, GeneratorNumberOfLevels, GeneratorFilters, " \
          "GeneratorNormalisationMomenturm, GeneratorActivationAlpha, " \
          "DiscriminatorStrides, DiscriminatorKernelSize, DiscriminatorNumberOfLevels, DiscriminatorFilters, " \
          "DiscriminatorNormalisationMomenturm, DiscriminatorOptimisationAlpha) " \
          "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"

    experiment_id = Logger.experiment_id

    val = (experiment_id, "DCGAN (Deep Convolutional Generative Adversarial Network)", k, epochs, batch_size,
           gen_strides, gen_kernel_size, gen_levels, gen_filters, gen_normalisation_momentum, gen_activation_alpha,
           dis_strides, dis_kernel_size, dis_levels, dis_filters, dis_normalisation_momentum,
           dis_activation_alpha)

    MethodologyLogger.db_cursor.execute(sql, val)

    for fold in range(k):
        Logger.print("Running Cross Validation Fold " + str(fold + 1) + "/" + str(k))
        Logger.current_fold = fold

        fold_d_losses = np.zeros((epochs * len(training_sets)))
        fold_g_losses = np.zeros((epochs * len(training_sets)))

        discriminator = DCGAN.DCGANDiscriminator(dummy, dis_strides, dis_kernel_size, dis_filters,
                                                 dis_activation_alpha, dis_normalisation_momentum, dis_levels)

        generator = DCGAN.DCGANGenerator(dummy, gen_strides, gen_kernel_size, gen_filters, gen_activation_alpha,
                                         gen_normalisation_momentum, gen_levels)

        DCGAN.Network.discriminator = discriminator.model
        DCGAN.Network.generator = generator.model

        DCGAN.Network.create_network(dummy)

        for ind in range(len(training_sets[fold])):
            training_set = training_sets[fold][ind]
            Logger.current_set = ind

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

        mlm.save_network(DCGAN.Network.discriminator, DCGAN.Network.generator,
                         Logger.experiment_id, "Fold" + str(fold))

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

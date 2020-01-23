import math
import os

from tensorflow_core.python.client import device_lib

from ExperimentTools import DatasetProcessor, MethodologyLogger
from GAN import DCGAN
from Settings import FileManager as fm, SettingsManager as sm, MachineLearningManager as mlm, DatabaseManager as dm
from ImageTools import VoxelProcessor as vp, ImageManager as im
from ExperimentTools.MethodologyLogger import Logger
import tensorflow as tf

import numpy as np


def run_model_on_core(core_id=None):
    pass


def run_train_test_split_experiment(aggregates, binders, split_percentage):
    pass


def run_k_fold_cross_validation_experiment(dataset_directories, k, architecture):
    if not isinstance(architecture, tuple):
        raise TypeError

    data_length = len(dataset_directories)

    if k > data_length:
        print("Dataset of size [" + str(data_length) + "] is too small for [" + str(k) + "] folds.")
        print("Setting k to [" + str(data_length) + "]")

        k = min(k, data_length)

    training_sets, testing_sets = DatasetProcessor.dataset_to_k_cross_fold(dataset_directories, k)

    epochs = int(sm.configuration.get("TRAINING_EPOCHS"))
    batch_size = int(sm.configuration.get("TRAINING_BATCH_SIZE"))

    architecture_id, gen_settings, disc_settings = architecture

    for fold in range(k):
        Logger.print("Running Cross Validation Fold " + str(fold + 1) + "/" + str(k))
        Logger.current_fold = fold

        fold_d_losses = list()
        fold_d_accuracies = list()
        fold_g_losses = list()
        fold_g_mses = list

        root_dir = fm.root_directories[fm.SpecialFolder.MODEL_DATA.value]

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        filepath = root_dir + "Experiment-" + str(Logger.experiment_id) + '_' + "Fold-" + str(fold + 1) + '_'

        discriminator_location = filepath + "discriminator.h5"
        generator_location = filepath + "generator.h5"

        discriminator = mlm.create_discriminator(gen_settings)
        generator = mlm.create_generator(disc_settings)

        DCGAN.Network.discriminator = discriminator.model
        DCGAN.Network.generator = generator.model

        DCGAN.Network.create_network()

        for ind in range(len(training_sets[fold])):
            training_set = training_sets[fold][ind]
            Logger.current_set = ind

            aggregates = list()
            binders = list()

            for directory in training_set:
                dimensions, loaded_aggregates, loaded_binders = vp.load_materials(directory)
                aggregates.extend(loaded_aggregates)
                binders.extend(loaded_binders)

            # im.save_voxel_image_collection(aggregates[10:15], fm.SpecialFolder.VOXEL_DATA, "figures/PostH5/aggregate")
            # im.save_voxel_image_collection(binders[10:15], fm.SpecialFolder.VOXEL_DATA, "figures/PostH5/binder")

            Logger.print("\tTraining on set " + str(ind + 1) + '/' + str(len(training_sets[fold])) + "... ")

            d_loss, g_loss, images = DCGAN.Network.train_network(epochs, batch_size, aggregates, binders)

            directory = fm.get_directory(fm.SpecialFolder.RESULTS) + "/Figures/Experiment-" + str(Logger.experiment_id) + '/Training'
            fm.create_if_not_exists(directory)

            buff_ind = '0' * (len(str(len(training_set))) - len(str(ind))) + str(ind)

            save_training_graphs(d_loss, g_loss, directory, fold, buff_ind)

            fold_d_losses.append(d_loss[0])
            fold_d_accuracies.append(d_loss[1])
            fold_g_losses.append(g_loss[0])
            fold_g_mses.append(g_loss[1])

            generator.model.save_weights(generator_location)
            discriminator.model.save_weights(discriminator_location)

            instance_id = Logger.log_model_instance_to_database(architecture_id, generator_location, discriminator_location)
            Logger.log_model_experiment_to_database(Logger.experiment_id, instance_id)

        test_network(testing_sets, fold, DCGAN.Network.generator)


def save_training_graphs(d_loss, g_loss, directory, fold, ind):
    fig = im.plt.figure()

    gen_error_ax = fig.add_subplot(3, 1, 1)
    dis_error_ax = fig.add_subplot(3, 1, 2)
    acc_ax = fig.add_subplot(3, 1, 3)

    x = range(len(g_loss[0]))

    gen_error_ax.plot(x, g_loss[0], '-g', label="Generator Loss")
    gen_error_ax.plot(x, g_loss[1], '-b', label="Generator MSE")

    dis_error_ax.plot(x, d_loss[0], '-r', label="Discriminator Loss")

    acc_ax.plot(x, d_loss[1], '-m', label="Discriminator Accuracy")

    gen_error_ax.set_xlabel("Epochs")
    acc_ax.set_xlabel("Epochs")
    dis_error_ax.set_xlabel("Epochs")

    gen_error_ax.set_ylabel("Error")
    dis_error_ax.set_ylabel("Error")

    acc_ax.set_ylabel("Accuracy")
    gen_error_ax.legend(loc="upper right")
    dis_error_ax.legend(loc="upper right")
    acc_ax.legend(loc="upper right")

    im.plt.gcf().savefig(
        directory + '/Experiment-' + str(Logger.experiment_id) + '_Fold-' + str(fold) + '_TrainingSet-' + str(
            ind) + '.pdf')
    im.plt.close(im.plt.gcf())


def test_network(testing_sets, fold, test_generator):
    Logger.print("Testing GAN on unseen aggregate voxels...")

    for testing_set in testing_sets[fold]:
        if not isinstance(testing_set, list):
            testing_set = list(testing_sets[fold])

        test_aggregates = list()
        test_binders = list()

        for directory in testing_set:
            Logger.print("\tLoading voxels from " + directory + "... ", end='')

            dimensions, test_aggregate, test_binder = vp.load_materials(directory)
            test_aggregates.append(test_aggregate)
            test_binders.append(test_binder)

            Logger.print("done!")

            test = np.array(test_aggregates)
            test = np.expand_dims(test, 4)

            results = list(test_generator.predict(test) * 255)

            directory = fm.get_directory(fm.SpecialFolder.RESULTS) + "/Figures/Experiment-" + str(
                Logger.experiment_id) + "/Outputs"
            fm.create_if_not_exists(directory)

            DISPLAY_VOXELS = False

            vp.save_voxels(results, dimensions, fm.SpecialFolder.GENERATED_VOXEL_DATA, "Test")

            if DISPLAY_VOXELS:
                 for ind in range(len(results)):
                     fig = im.plt.figure(figsize=(10, 5))
                     ax_expected = fig.add_subplot(1, 2, 1, projection='3d')
                     ax_expected.title.set_text("Expected")

                     ax_actual = fig.add_subplot(1, 2, 2, projection='3d')
                     ax_actual.title.set_text("Actual")

                     ax_expected.voxels(test_aggregates[ind], facecolors='w', edgecolors='w')
                     ax_expected.voxels(test_binders[ind], facecolors='k', edgecolors='k')

                     ax_actual.voxels(test_aggregates[ind], facecolors='w', edgecolors='w')
                     ax_actual.voxels(np.squeeze(results[ind]), facecolors='k', edgecolors='k')

                     im.plt.gcf().savefig(directory + '/Experiment-' + str(Logger.experiment_id) + '_Fold-' + str(
                            fold) + '_Voxel-' + str(ind) + '.jpg')
                     im.plt.close(im.plt.gcf())


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

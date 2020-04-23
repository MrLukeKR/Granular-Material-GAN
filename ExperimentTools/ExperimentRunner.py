import os
from itertools import repeat

import numpy as np
from tqdm import tqdm

from GAN.DCGAN import gan_to_voxels
from ImageTools.VoxelProcessor import voxels_to_core
from Settings import MessageTools as mt
from ExperimentTools import DatasetProcessor
from GAN import DCGAN
from Settings import FileManager as fm, SettingsManager as sm, MachineLearningManager as mlm
from ImageTools import VoxelProcessor as vp, ImageManager as im
from ExperimentTools.MethodologyLogger import Logger
from ExperimentTools import DataVisualiser as dv
from Settings.MessageTools import print_notice
from ImageTools.CoreAnalysis import CoreVisualiser as cv


def run_model_on_core(core_id=None):
    pass


def run_train_test_split_experiment(aggregates, binders, split_percentage):
    pass


def run_k_fold_cross_validation_experiment(dataset_directories, k, architecture, multiprocessing_pool=None):
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

    core = str.split(testing_sets[0][0], '/')[-1]
    animation_dimensions, animation_aggregates, _ = vp.load_materials(core)
    animation_aggregates = np.expand_dims(animation_aggregates, 4)

    for fold in range(k):
        Logger.print("Running Cross Validation Fold " + str(fold + 1) + "/" + str(k))
        Logger.current_fold = fold
        database_logged = False

        fold_d_losses = list()
        fold_d_accuracies = list()
        fold_g_losses = list()
        fold_g_mses = list()

        root_dir = fm.compile_directory(fm.SpecialFolder.MODEL_DATA)

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        experiment_id = "Experiment-" + str(Logger.experiment_id)
        fold_id = "_Fold-" + str(fold)

        filepath = root_dir + experiment_id + fold_id + '_'

        discriminator_loc = filepath + "discriminator"
        generator_loc = filepath + "generator"

        discriminator = mlm.create_discriminator(gen_settings)
        generator = mlm.create_generator(disc_settings)

        DCGAN.Network.discriminator = discriminator.model
        DCGAN.Network.generator = generator.model

        DCGAN.Network.create_network()

        for ind in range(len(training_sets[fold])):
            training_set = training_sets[fold][ind]
            Logger.current_set = ind

            voxels = list()

            for directory in training_set:
                core_id = str.split(directory, '/')[-1]
                voxel_directory = fm.compile_directory(fm.SpecialFolder.VOXEL_DATA) + core_id + "/"

                temp_voxels, dimensions = vp.load_voxels(voxel_directory,
                                                         "segment_" + sm.configuration.get("VOXEL_RESOLUTION"))
                voxels.extend(temp_voxels)

            voxels = np.array(voxels, dtype=np.uint8)
            mt.print_notice("Voxel matrix uses %sGB of memory"
                            % str(round((voxels.size * voxels.itemsize) / (1024 * 1024 * 1024), 2)),
                            mt.MessagePrefix.DEBUG)

#            aggregates = np.squeeze(np.array([voxels == 255]) * 2.0 - 1.0)
#            mt.print_notice("Aggregates matrix uses %sGB of memory"
#                            % str(round((aggregates.size * aggregates.itemsize) / (1024 ** 3), 2)),
#                            mt.MessagePrefix.DEBUG)

            # Due to implementations using floats/ints sometimes resulting in either 127 or 128 for the binder
            # value, here we determine binder as "not void or aggregate"
            #binders = np.squeeze(np.array([(voxels != 0) & (voxels != 255)]) * 2.0 - 1.0)
            #mt.print_notice("Binders matrix uses %sGB of memory"
#                            % str(round((binders.size * binders.itemsize) / (1024 ** 3), 2)),
                            #mt.MessagePrefix.DEBUG)

#            mt.print_notice("Freeing voxel matrix memory... ", mt.MessagePrefix.DEBUG, end='')
#            del voxels
#            print("done!")

            Logger.print("\tTraining on set " + str(ind + 1) + '/' + str(len(training_sets[fold])) + "... ")

            directory = fm.compile_directory(fm.SpecialFolder.FIGURES) + experiment_id + "/Outputs/CoreAnimation/"
            fm.create_if_not_exists(directory)

            animation_data = (animation_aggregates, animation_dimensions, directory)

            d_loss, g_loss, images = DCGAN.Network.\
                train_network(epochs, batch_size,
                              np.squeeze(np.array([voxels == 255], dtype=bool)),
                              np.squeeze(np.array([(voxels != 0) & (voxels != 255)], dtype=bool)),
                              animation_data)

            filename = "Experiment-" + str(Logger.experiment_id)
            directory = fm.compile_directory(fm.SpecialFolder.FIGURES) + filename + '/Training'
            fm.create_if_not_exists(directory)

            buff_ind = '0' * (len(str(len(training_set))) - len(str(ind))) + str(ind)

            save_training_graphs(d_loss, g_loss, directory, fold, buff_ind)

            fold_d_losses.append(d_loss[0])
            fold_d_accuracies.append(d_loss[1])
            fold_g_losses.append(g_loss[0])
            fold_g_mses.append(g_loss[1])

            generator.model.save_weights(generator_loc)
            discriminator.model.save_weights(discriminator_loc)

            # If on the first iteration
            if not database_logged:
                instance_id = Logger.log_model_instance_to_database(architecture_id, generator_loc, discriminator_loc)[0]
                Logger.log_model_experiment_to_database(Logger.experiment_id, instance_id)
                database_logged = True

        test_network(testing_sets, fold, DCGAN.Network.generator, multiprocessing_pool)


def save_training_graphs(d_loss, g_loss, directory, fold, ind):
    fig = im.plt.figure()

    gen_error_ax = fig.add_subplot(3, 1, 1)
    dis_error_ax = fig.add_subplot(3, 1, 2)
    acc_ax = fig.add_subplot(3, 1, 3)

    x = range(len(g_loss[0]))

    dv.plot_training_data(gen_error_ax, dis_error_ax, acc_ax,
                          x, g_loss[0], g_loss[1], d_loss[0], d_loss[1])

    im.plt.gcf().savefig(
        directory + '/Experiment-' + str(Logger.experiment_id) + '_Fold-' + str(fold) + '_TrainingSet-' + str(
            ind) + '.pdf')
    im.plt.close(im.plt.gcf())


def test_network(testing_sets, fold, test_generator, multiprocessing_pool=None):
    Logger.print("Testing GAN on unseen aggregate voxels...")

    for testing_set in testing_sets[fold]:
        if not isinstance(testing_set, list):
            testing_set = list(testing_sets[fold])

        for directory in testing_set:
            core = str.split(directory, '/')[-1]
            dimensions, test_aggregate, test_binder = vp.load_materials(core)

            test_aggregate = np.expand_dims(test_aggregate, 4)

            results = gan_to_voxels(test_generator, test_aggregate)

            experiment_id = "Experiment-" + str(Logger.experiment_id)
            fold_id = "_Fold-" + str(fold)

            directory = fm.compile_directory(fm.SpecialFolder.FIGURES) + experiment_id + "/Outputs/"
            fm.create_if_not_exists(directory)

            test_aggregate = np.squeeze(test_aggregate)
            binder_core = voxels_to_core(np.array([x >= 128 for x in results]), dimensions)
            aggregate_core = voxels_to_core(test_aggregate, dimensions)

            binder_core = cv.voxels_to_mesh(binder_core)
            aggregate_core = cv.voxels_to_mesh(aggregate_core)

            cv.save_mesh(binder_core, directory + "binder.stl")
            cv.save_mesh(aggregate_core, directory + "aggregate.stl")

            results = list(results)
            vp.save_voxels(results, dimensions, directory, "GeneratedVoxels")

            print_notice("Saving Real vs. Generated voxel plots... ", end='')
            file_location = directory + experiment_id + fold_id + "_core-" + core

            result_length = range(len(results))

            if multiprocessing_pool is None:
                for ind in tqdm(result_length):
                    plot_real_vs_generated_voxels(test_aggregate[ind], test_binder[ind], results[ind], file_location, ind)
            else:
                multiprocessing_pool.starmap(plot_real_vs_generated_voxels,
                                             zip(test_aggregate, test_binder, results,
                                                 repeat(file_location, len(results)), list(result_length)))


def plot_real_vs_generated_voxels(aggregates, expected, actual, file_location, ind):
    fig = im.plt.figure(figsize=(10, 5))
    ax_expected = fig.add_subplot(1, 2, 1, projection='3d')
    ax_expected.title.set_text("Real")

    ax_actual = fig.add_subplot(1, 2, 2, projection='3d')
    ax_actual.title.set_text("Generated")

    ax_expected.voxels(aggregates, facecolors='w', edgecolors='w')
    ax_expected.voxels(expected, facecolors='k', edgecolors='k')

    actual = np.squeeze(actual)

    ax_actual.voxels(aggregates, facecolors='w', edgecolors='w')
    ax_actual.voxels(actual, facecolors='k', edgecolors='k')

    voxel_id = "_Voxel-" + str(ind)

    im.plt.gcf().savefig(file_location + voxel_id + '.pdf')
    im.plt.close(im.plt.gcf())

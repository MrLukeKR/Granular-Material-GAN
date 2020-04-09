import os
import numpy as np
from tqdm import tqdm

from Settings import MessageTools as mt
from ExperimentTools import DatasetProcessor
from GAN import DCGAN
from Settings import FileManager as fm, SettingsManager as sm, MachineLearningManager as mlm
from ImageTools import VoxelProcessor as vp, ImageManager as im
from ExperimentTools.MethodologyLogger import Logger
from ExperimentTools import DataVisualiser as dv
from Settings.MessageTools import print_notice


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
        database_logged = False

        fold_d_losses = list()
        fold_d_accuracies = list()
        fold_g_losses = list()
        fold_g_mses = list()

        root_dir = fm.compile_directory(fm.SpecialFolder.MODEL_DATA)

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        filepath = root_dir + "Experiment-" + str(Logger.experiment_id) + '_' + "Fold-" + str(fold + 1) + '_'

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
                id = str.split(directory, '/')[-1]
                voxel_directory = fm.compile_directory(fm.SpecialFolder.VOXEL_DATA) + id + "/"

                temp_voxels, dimensions = vp.load_voxels(voxel_directory,
                                                         "segment_" + sm.configuration.get("VOXEL_RESOLUTION"))
                voxels.extend(temp_voxels)

            voxels = np.array(voxels, dtype=np.uint8)
            mt.print_notice("Voxel matrix uses %sGB of memory"
                            % str(round((voxels.size * voxels.itemsize) / (1024 * 1024 * 1024), 2)),
                            mt.MessagePrefix.DEBUG)

            aggregates = np.squeeze(np.array([voxels == 255]) * 1.0)
            mt.print_notice("Aggregates matrix uses %sGB of memory"
                            % str(round((aggregates.size * aggregates.itemsize) / (1024 ** 3), 2)),
                            mt.MessagePrefix.DEBUG)

            # Due to implementations using floats/ints sometimes resulting in either 127 or 128 for the binder
            # value, here we determine binder as "not void or aggregate"
            binders = np.squeeze(np.array([(voxels != 0) & (voxels != 255)]) * 1.0)
            mt.print_notice("Binders matrix uses %sGB of memory"
                            % str(round((binders.size * binders.itemsize) / (1024 ** 3), 2)),
                            mt.MessagePrefix.DEBUG)

            mt.print_notice("Freeing voxel matrix memory... ", mt.MessagePrefix.DEBUG, end='')
            del voxels
            print("done!")

            Logger.print("\tTraining on set " + str(ind + 1) + '/' + str(len(training_sets[fold])) + "... ")

            d_loss, g_loss, images = DCGAN.Network.train_network(epochs, batch_size,
                                                                 aggregates, binders)

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

        test_network(testing_sets, fold, DCGAN.Network.generator)


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


def test_network(testing_sets, fold, test_generator):
    Logger.print("Testing GAN on unseen aggregate voxels...")

    for testing_set in testing_sets[fold]:
        if not isinstance(testing_set, list):
            testing_set = list(testing_sets[fold])

        test_aggregates = list()
        test_binders = list()
        dimensions = None

        for directory in testing_set:
            dimensions, test_aggregate, test_binder = vp.load_materials(directory)
            test_aggregates.extend(test_aggregate)
            test_binders.extend(test_binder)

        test = np.array(test_aggregates)
        test = np.expand_dims(test, 4)

        results = list(test_generator.predict(test) * 255)

        experiment_id = "Experiment-" + str(Logger.experiment_id)
        fold_id = "_Fold-" + str(fold)

        directory = fm.compile_directory(fm.SpecialFolder.FIGURES) + experiment_id + "/Outputs/"
        fm.create_if_not_exists(directory)

        vp.save_voxels(results, dimensions, directory, "Test")

        print_notice("Saving Real vs. Generated voxel plots... ", end='')
        for ind in tqdm(range(len(results))):
            fig = im.plt.figure(figsize=(10, 5))
            ax_expected = fig.add_subplot(1, 2, 1, projection='3d')
            ax_expected.title.set_text("Real")

            ax_actual = fig.add_subplot(1, 2, 2, projection='3d')
            ax_actual.title.set_text("Generated")

            ax_expected.voxels(test_aggregates[ind], facecolors='w', edgecolors='w')
            ax_expected.voxels(test_binders[ind], facecolors='k', edgecolors='k')

            ax_actual.voxels(test_aggregates[ind], facecolors='w', edgecolors='w')
            ax_actual.voxels(np.squeeze(results[ind]), facecolors='k', edgecolors='k')

            voxel_id = "_Voxel-" + str(ind)

            im.plt.gcf().savefig(directory + experiment_id + fold_id + voxel_id + '.jpg')
            im.plt.close(im.plt.gcf())

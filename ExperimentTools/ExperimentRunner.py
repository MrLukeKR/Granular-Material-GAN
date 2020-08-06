import os
import numpy as np
import matplotlib as mpl
from PIL import Image

from tqdm import tqdm
from itertools import repeat
from ExperimentTools.DataVisualiser import save_training_graphs
from ExperimentTools.DatasetProcessor import prepare_tf_set
from GAN.DCGAN import gan_to_voxels
from ImageTools.VoxelProcessor import voxels_to_core
from Settings import MessageTools as mt
from ExperimentTools import DatasetProcessor, MethodologyLogger
from GAN import DCGAN
from Settings import FileManager as fm, SettingsManager as sm, MachineLearningManager as mlm
from ImageTools import VoxelProcessor as vp, ImageManager as im
from ExperimentTools.MethodologyLogger import Logger
from Settings.MessageTools import print_notice
from ImageTools.CoreAnalysis import CoreVisualiser as cv
from matplotlib import pyplot as plt

mpl.rc('image', cmap='gray')


def run_model_on_core(core_id=None):
    pass


def run_experiment(dataset_iterator, gen_settings, disc_settings, experiment_id, batch_size, fold, epochs, dataset_size,
                   animate_with_rois=False):
    discriminator = mlm.create_discriminator(disc_settings)
    generator = mlm.create_generator(gen_settings)

    DCGAN.Network.discriminator = discriminator.model
    DCGAN.Network.generator = generator.model

    DCGAN.Network.create_network()

    base_dir = fm.compile_directory(fm.SpecialFolder.GENERATED_ASPHALT_3D_ROI_MODELS if animate_with_rois
                                    else fm.SpecialFolder.GENERATED_ASPHALT_3D_CORE_MODELS)
    directory = base_dir + experiment_id + "/CoreAnimation/"
    if sm.get_setting("ENABLE_TRAINING_ANIMATION") == "True":
        fm.create_if_not_exists(directory)

    animation_data = None

    # Use 15-3007 as this is the largest core
    if sm.get_setting("ENABLE_TRAINING_ANIMATION") == "True":
        animation_dimensions, animation_aggregates = vp.load_materials("15-3007", use_rois=animate_with_rois,
                                                                       return_binder=False)

        animation_aggregates = np.expand_dims(animation_aggregates, 4)

        animation_data = (animation_aggregates, animation_dimensions, directory)

    d_loss, g_loss = DCGAN.Network.train_network_tfdata(batch_size, dataset_iterator, fold + 1, epochs, dataset_size,
                                                        animation_data)

    return g_loss, d_loss, DCGAN.Network.generator, DCGAN.Network.discriminator

def run_train_test_split_experiment(dataset_directories, split_percentage):
    raise NotImplemented

    #TODO: Finish this and tidy up this file as much as possible
    g_loss, d_loss, trained_gen, trained_disc = run_experiment()
    pass


def run_k_fold_cross_validation_experiment(dataset_directories, k, architecture, multiprocessing_pool=None,
                                           train_with_rois=True, animate_with_rois=False):
    if not isinstance(architecture, tuple):
        raise TypeError

    data_length = len(dataset_directories)

    if k > data_length:
        print("Dataset of size [" + str(data_length) + "] is too small for [" + str(k) + "] folds.")
        print("Setting k to [" + str(data_length) + "]")

        k = min(k, data_length)

    training_sets, testing_sets = DatasetProcessor.dataset_to_k_cross_fold(dataset_directories, k)

    epochs = int(sm.get_setting("TRAINING_EPOCHS"))
    batch_size = int(sm.get_setting("TRAINING_BATCH_SIZE"))

    MethodologyLogger.Logger(fm.compile_directory(fm.SpecialFolder.LOGS), k, epochs, batch_size)

    architecture_id, gen_settings, disc_settings = architecture

    num_gpus = len(mlm.get_available_gpus())

    print_notice("GPU devices available: %s" % str(num_gpus), mt.MessagePrefix.DEBUG)

    if num_gpus == 0:
        print_notice("No GPU devices available", mt.MessagePrefix.ERROR)
        raise SystemError

    batch_size *= num_gpus

    for fold in range(k):
        print_notice("Running Cross Validation Fold " + str(fold + 1) + "/" + str(k))
        Logger.current_fold = fold
        database_logged = False

        root_dir = fm.compile_directory(fm.SpecialFolder.MODEL_DATA)

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        experiment_id = "Experiment-" + str(Logger.experiment_id)
        fold_id = "_Fold-" + str(fold)

        filepath = root_dir + experiment_id + fold_id + '_'

        discriminator_loc = filepath + "discriminator"
        generator_loc = filepath + "generator"

        # Machine Learning >>>

        training_filenames = [fm.compile_directory(fm.SpecialFolder.ROI_DATASET_DATA
                                          if train_with_rois else fm.SpecialFolder.CORE_DATASET_DATA) + x
                     + "/segment_64.tfrecord" for x in training_sets[fold]]

        testing_filenames = [fm.compile_directory(fm.SpecialFolder.ROI_DATASET_DATA
                                                   if train_with_rois else fm.SpecialFolder.CORE_DATASET_DATA) + x
                              + "/segment_64.tfrecord" for x in testing_sets[fold]]

        voxel_res = int(sm.get_setting("VOXEL_RESOLUTION"))
        voxel_dims = [voxel_res, voxel_res, voxel_res]

        train_ds, dataset_size = prepare_tf_set(training_filenames, voxel_dims, epochs, batch_size)

        train_ds_iter = iter(train_ds)

        g_loss, d_loss, trained_gen, trained_disc = run_experiment(train_ds_iter, gen_settings, disc_settings,
                                                                   experiment_id, batch_size, fold, epochs, dataset_size,
                                                                   animate_with_rois=animate_with_rois)

        filename = "Experiment-" + str(Logger.experiment_id)
        directory = fm.compile_directory(fm.SpecialFolder.FIGURES) + filename + '/Training'
        fm.create_if_not_exists(directory)

        save_training_graphs(d_loss, g_loss, directory, experiment_id, fold, epochs)

        trained_gen.save_weights(generator_loc)
        trained_disc.save_weights(discriminator_loc)

        # If on the first iteration
        if not database_logged:
            instance_id = Logger.log_model_instance_to_database(architecture_id, generator_loc, discriminator_loc)[0]
            Logger.log_model_experiment_to_database(Logger.experiment_id, instance_id)
            database_logged = True

        test_network(testing_sets, fold, DCGAN.Network.generator, batch_size, directory, multiprocessing_pool)


def test_network(testing_sets, fold, test_generator, batch_size, figure_directory=None, multiprocessing_pool=None):
    mt.print_notice("Testing GAN on unseen aggregate voxels...")

    for testing_set in testing_sets[fold]:
        if not isinstance(testing_set, list):
            testing_set = list(testing_sets[fold])

        for directory in testing_set:
            core = str.split(directory, '/')[-1]
            dimensions, test_aggregate, test_binder = vp.load_materials(core, use_rois=False)
            test_aggregate = np.expand_dims(test_aggregate, 4)

            # TODO: Make testing use TFData rather than numpy loading (faster)

            # Data must be in the range of [-1, 1] for the GAN (voxels are stored as [0, 1])
            results = gan_to_voxels(test_generator, ((test_aggregate / 255) * 2) - 1, batch_size)

            results = (results + 1) / 2

            if sm.get_setting("ENABLE_GAN_OUTPUT_HISTOGRAM") == "True":
                plt.hist(np.array((results * 255), dtype=np.uint8).flatten(), bins=range(256))
                plt.title("Histogram of GAN outputs")

                if figure_directory is not None:
                    plt.savefig(figure_directory + "/GAN_Output_Histogram.pdf",)

                if sm.get_setting("ENABLE_IMAGE_DISPLAY") == "True":
                    plt.show()

            # Rescale outputs into [0, 1], which can then be thresholded and scaled into [0, 255]

            results = results >= float(sm.get_setting("IO_GAN_OUTPUT_THRESHOLD"))
            results = results * 255

            experiment_id = "Experiment-" + str(Logger.experiment_id)
            fold_id = "_Fold-" + str(fold)

            directory = fm.compile_directory(fm.SpecialFolder.FIGURES) + experiment_id + "/Outputs/"
            fm.create_if_not_exists(directory)

            test_aggregate = np.squeeze(test_aggregate)

            # Remove overlapping aggregate and binder
            if sm.get_setting("ENABLE_FIX_GAN_OUTPUT_OVERLAP") == "True":
                results -= np.logical_and(results, test_aggregate == 255)

            binder_core = voxels_to_core(results, dimensions)
            aggregate_core = voxels_to_core(test_aggregate, dimensions)

            slice_directory = directory + "BinderSlices/"
            fm.create_if_not_exists(slice_directory)

            for ind, slice in tqdm(enumerate(binder_core), desc=mt.get_notice("Saving Generated Core Slices")):
                im = Image.fromarray(slice)

                buff_ind = (len(str(len(binder_core))) - len(str(ind))) * "0" + str(ind)
                im.save(slice_directory + buff_ind + ".png")

            binder_core = cv.voxels_to_mesh(binder_core)
            aggregate_core = cv.voxels_to_mesh(aggregate_core)

            cv.save_mesh(binder_core, directory + "generated_binder.stl")
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

import os
import tensorflow as tf
import numpy as np

from tqdm import tqdm
from itertools import repeat
from ExperimentTools.DataVisualiser import save_training_graphs
from ExperimentTools.VoxelGenerator import VoxelGenerator
from GAN.DCGAN import gan_to_voxels
from ImageTools.VoxelProcessor import voxels_to_core
from Settings import MessageTools as mt
from ExperimentTools import DatasetProcessor
from GAN import DCGAN
from Settings import FileManager as fm, SettingsManager as sm, MachineLearningManager as mlm
from ImageTools import VoxelProcessor as vp, ImageManager as im
from ExperimentTools.MethodologyLogger import Logger
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
    # animation_dimensions, animation_aggregates, _ = vp.load_materials(core)
    # animation_aggregates = np.expand_dims(animation_aggregates, 4)

    print_notice("GPU devices available: %s" % str(len(mlm.get_available_gpus())), mt.MessagePrefix.DEBUG)

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

        # Machine Learning >>>

        filenames = [fm.compile_directory(fm.SpecialFolder.DATASET_DATA) + x + "/segment_64.tfrecord" for x in training_sets[fold]]
        voxel_res = [64, 64, 64]

        train_ds = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=len(filenames))

        example = {
            'aggregate': tf.io.FixedLenFeature([], dtype=tf.string),
            'binder': tf.io.FixedLenFeature([], dtype=tf.string)
        }

        def _parse_voxel_function(example_proto):
            return tf.io.parse_single_example(example_proto, example)

        def _decode_voxel_function(serialised_example):
            aggregate = tf.io.decode_raw(serialised_example['aggregate'], tf.bool)
            binder = tf.io.decode_raw(serialised_example['binder'], tf.bool)

            segments = [aggregate, binder]

            for ind in range(2):
                segments[ind] = tf.cast(segments[ind], dtype=tf.bfloat16)

                segments[ind] = tf.reshape(segments[ind], voxel_res)

                segments[ind] = tf.expand_dims(segments[ind], -1)

            return segments

        def _rescale_voxel_values(features, labels):
            segments = [features, labels]

            for ind in range(2):
                segments[ind] = tf.math.multiply(segments[ind], 2.0)
                segments[ind] = tf.math.subtract(segments[ind], 1.0)

            return features, labels

        train_ds = train_ds.map(_parse_voxel_function)
        train_ds = train_ds.map(_decode_voxel_function)
        train_ds = train_ds.map(_rescale_voxel_values)

        train_ds = train_ds.shuffle(buffer_size=len(filenames))
        train_ds = train_ds.repeat(epochs)
        train_ds = train_ds.batch(batch_size=batch_size)
        train_ds = train_ds.prefetch(batch_size)

        ds_iter = iter(train_ds)

        discriminator = mlm.create_discriminator(gen_settings)
        generator = mlm.create_generator(disc_settings)

        DCGAN.Network.discriminator = discriminator.model
        DCGAN.Network.generator = generator.model

        DCGAN.Network.create_network()

#            directory = fm.compile_directory(fm.SpecialFolder.FIGURES) + experiment_id + "/Outputs/CoreAnimation/"
#            fm.create_if_not_exists(directory)

#            animation_data = (animation_aggregates, animation_dimensions, directory)

        #    d_loss, g_loss, images = DCGAN.Network.\
        #        train_network(epochs, batch_size,
        #                      np.squeeze(np.array([voxels == 255], dtype=bool)),
        #                      np.squeeze(np.array([(voxels != 0) & (voxels != 255)], dtype=bool)))

        d_loss, g_loss, images = DCGAN.Network.train_network_tfdata(epochs, batch_size, ds_iter)

        filename = "Experiment-" + str(Logger.experiment_id)
        directory = fm.compile_directory(fm.SpecialFolder.FIGURES) + filename + '/Training'
        fm.create_if_not_exists(directory)

        buff_ind = '0' * (len(str(k)) - len(str(fold))) + str(fold)

        save_training_graphs(d_loss, g_loss, directory, experiment_id, fold, buff_ind)

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

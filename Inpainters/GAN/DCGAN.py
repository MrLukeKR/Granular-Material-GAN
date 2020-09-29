import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from guppy import hpy

from multiprocessing import Process

from trimesh import caching

from Inpainters.GAN import AbstractGAN
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Activation, Conv3D, Conv3DTranspose as Deconv3D, BatchNormalization, LeakyReLU
from ExperimentTools.MethodologyLogger import Logger
from ImageTools.CoreAnalysis.CoreVisualiser import save_mesh, voxels_to_mesh, fix_mesh
from ImageTools.VoxelProcessor import voxels_to_core
from Settings.MessageTools import print_notice
from Settings import DatabaseManager as dm, MachineLearningManager as mlm, SettingsManager as sm, MessageTools as mt


class Network(AbstractGAN.Network):
    @property
    def adversarial(self):
        return self._adversarial

    @adversarial.setter
    def adversarial(self, value):
        self._adversarial = value

    @property
    def generator(self):
        return self._generator

    @generator.setter
    def generator(self, value):
        self._generator = value

    @property
    def discriminator(self):
        return self._discriminator

    @discriminator.setter
    def discriminator(self, value):
        self._discriminator = value

    def __init__(self, data):
        self.create_network()

    @classmethod
    def create_network(cls, data=None):
        if data is None:
            data = mlm.data_template
        print_notice("Initialising Generative Adversarial Network...")

        vox_res = int(sm.get_setting("VOXEL_RESOLUTION"))

        data_shape = (vox_res, vox_res, vox_res, 1)

        optimizer = optimizers.Adam(1e-4)

        masked_vol = Input(shape=data_shape)

        with mlm.strategy.scope():
            cls.discriminator.compile(loss='binary_crossentropy',
                                      optimizer=optimizer,
                                      metrics=['accuracy'])

            gen_missing = cls.generator(masked_vol)
            cls.discriminator.trainable = False

            valid = cls.discriminator(gen_missing)

            cls.adversarial = Model(masked_vol, [gen_missing, valid])
            cls.adversarial.compile(loss=['mse', 'binary_crossentropy'],
                                    loss_weights=[0.999, 0.001],
                                    optimizer=optimizer)

    @classmethod
    def train_step(cls, features, labels, valid, fake):
        # This is the binder generated for a given aggregate arrangement
        gen_missing = cls.generator.predict(features)
        d_loss_real = cls.discriminator.train_on_batch(labels, valid)
        d_loss_fake = cls.discriminator.train_on_batch(gen_missing, fake)

        g_loss = cls.adversarial.train_on_batch(features, [labels, valid])

        d_loss = tf.add(d_loss_real, d_loss_fake)

        return list(d_loss.numpy()), g_loss

    @classmethod
    def train_network_tfdata(cls, batch_size, dataset_iterator, epochs, total_batches, fold=None,
                             core_animation_data=None, preview_epoch_interval=None):
        print_notice("Training Generative Adversarial Network...")

        valid = tf.fill((batch_size, 1), 0.9)
        fake = tf.zeros((batch_size, 1))

        all_d_loss = list()
        all_g_loss = list()

        animation_step = int(sm.get_setting("TRAINING_ANIMATION_BATCH_STEP"))

        batch_no = 1
        epoch_no = 1

        batches_per_epoch = total_batches // epochs

        progress = tqdm(total=total_batches * batch_size, desc=mt.get_notice("Starting GAN Training"))

        overflow_notified = False
        stop_training = False

        for features, labels in dataset_iterator:
            if epoch_no > epochs and not overflow_notified:
                print_notice("Dataset could not be perfectly divided into %s epochs. "
                             "Running overflow epoch..." % str(epochs), mt.MessagePrefix.WARNING)

                overflow_notified = True

            if features.shape[0] != batch_size:
                last_valid = tf.fill((features.shape[0], 1), 0.9)
                last_fake = tf.zeros((features.shape[0], 1))

                with mlm.strategy.scope():
                    d_loss, g_loss = cls.train_step(features, labels, last_valid, last_fake)
            else:
                with mlm.strategy.scope():
                    d_loss, g_loss = cls.train_step(features, labels, valid, fake)

            progress.update(batch_size)
            progress.set_description(mt.get_notice("Epoch %d, Batch %d (%d Voxels) "
                                                   "[DIS loss: %f, acc: %.2f%%] [GEN loss: %f, mse: %f]"
                                                   % (epoch_no, batch_no,
                                                      ((epoch_no - 1) * batches_per_epoch * batch_size)
                                                      + (batch_no * batch_size),
                                                      d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1])))

            Logger.log_batch_training_to_database(epoch_no, batch_no, g_loss, d_loss, fold)

            all_d_loss.append(d_loss)
            all_g_loss.append(g_loss)

            should_preview = (preview_epoch_interval is not None
                              and epoch_no % preview_epoch_interval == 0)
            should_animate = (core_animation_data is not None
                              and len(core_animation_data) == 3
                              and batch_no % animation_step == 0)

            if should_animate or should_preview:
                generated_core = gan_to_core(cls.adversarial, core_animation_data[0], core_animation_data[1],
                                             batch_size)

                if should_preview:
                    cls.preview_results(generated_core)

                    stop_training = not cls.continue_or_exit()

                if should_animate:
                    try:
                        p = Process(target=cls.animate_gan,
                                    args=(core_animation_data, generated_core, batch_no,))
                        p.start()
                        p.join()
                    except MemoryError:
                        print_notice("Ran out of memory when creating mesh!", mt.MessagePrefix.ERROR)
                        h = hpy()
                        print(h.heap())

            if stop_training:
                mt.print_notice("Exiting GAN training early")
                break

            if batch_no >= batches_per_epoch:
                epoch_no += 1
                batch_no = 1
            else:
                batch_no += 1

        progress.close()
        return all_d_loss, all_g_loss

    @classmethod
    def animate_gan(cls, core_animation_data, generated_core, batch):
        mesh = voxels_to_mesh(generated_core)
        mesh = fix_mesh(mesh)
        save_mesh(mesh, core_animation_data[2] + '-Batch_' + str(batch) + '.stl')
        caching.Cache.clear(mesh)

    @classmethod
    def preview_results(cls, generated_core):
        # TODO: View these images in a slider like ImageJ
        raise NotImplemented

    @classmethod
    def continue_or_exit(cls):
        inp = ""

        while inp.upper() != "Y" or inp != "N":
            inp = input(mt.get_notice("Would you like to continue training? [Y/N]"))

        return inp.upper() == "Y"

    @classmethod
    def test_network(cls, testing_set):
        pass


class DCGANDiscriminator:
    _model = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def __init__(self, voxels, strides, kernel_size,
                 initial_filters, activation_alpha, normalisation_momentum, encoder_levels):
        print_notice("\tInitialising Deep Convolutional Generative Adversarial Network (Discriminator)",
                         mt.MessagePrefix.INFORMATION)

        x = len(voxels[0])
        y = len(voxels[0][0])
        z = len(voxels[0][0][0])
        w = len(voxels)

        channels = 1  # TODO: Make this variable from the settings file

        print_notice("\t\t Input size is: " + str(w) + " (" + str(x) + " * " + str(y) + " * " + str(z) + ") voxels",
                     mt.MessagePrefix.INFORMATION)

        voxel_shape = (x, y, z, channels)

        # START MODEL BUILDING
        with mlm.strategy.scope():
            model = Sequential(name="Discriminator")

            for level in range(0, encoder_levels):
                if level == 0:
                    model.add(Conv3D(initial_filters * (pow(2, level)), kernel_size=kernel_size, strides=strides,
                                     input_shape=voxel_shape, padding="same"))
                else:
                    model.add(
                        Conv3D(initial_filters * (pow(2, level)), kernel_size=kernel_size, strides=strides, padding="same"))
                model.add(LeakyReLU(alpha=activation_alpha))
                if sm.get_setting("TRAINING_USE_BATCH_NORMALISATION") == "True":
                    model.add(BatchNormalization(momentum=normalisation_momentum))

            model.add(Flatten())
            model.add(Dense(1, activation="sigmoid"))

            input_voxel = Input(shape=voxel_shape)
            verdict = model(input_voxel)

            self._model = Model(input_voxel, verdict)

        model.summary()


class DCGANGenerator:
    _model = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def __init__(self, voxels, strides, kernel_size,
                 initial_filters, activation_alpha, normalisation_momentum, encoder_levels):
        print_notice("\tInitialising Deep Convolutional Generative Adversarial Network (Generator)",
                     mt.MessagePrefix.INFORMATION)

        x = len(voxels[0])
        y = len(voxels[0][0])
        z = len(voxels[0][0][0])
        w = len(voxels)

        channels = 1  # TODO: Make this variable from the settings file

        print_notice("\t\t Input size is: " + str(w) + " (" + str(x) + " * " + str(y) + " * " + str(z) + ") voxels",
                     mt.MessagePrefix.INFORMATION)

        voxel_shape = (x, y, z, channels)

        # START MODEL BUILDING
        with mlm.strategy.scope():
            model = Sequential(name="Generator")

            for level in range(0, encoder_levels):
                if level == 0:
                   model.add(Conv3D(initial_filters * (pow(2, level)),
                                     kernel_size=kernel_size, strides=strides, input_shape=voxel_shape, padding="same"))
                else:
                   model.add(Conv3D(initial_filters * (pow(2, level)),
                                    kernel_size=kernel_size, strides=strides, padding="same"))
                model.add(LeakyReLU(alpha=activation_alpha))
                if sm.get_setting("TRAINING_USE_BATCH_NORMALISATION") == "True":
                    model.add(BatchNormalization(momentum=normalisation_momentum))

            for level in range(encoder_levels - 1, 0, -1):
                model.add(Deconv3D(initial_filters * pow(2, level - 1),
                                   kernel_size=kernel_size, strides=strides, padding="same"))
                model.add(Activation("relu"))
                if sm.get_setting("TRAINING_USE_BATCH_NORMALISATION") == "True":
                    model.add(BatchNormalization(momentum=normalisation_momentum))

            model.add(Deconv3D(channels, kernel_size=kernel_size, strides=strides, padding="same"))
            model.add(Activation("tanh"))

            input_voxel = Input(shape=voxel_shape)
            gen_missing = model(input_voxel)

            self._model = Model(input_voxel, gen_missing)

        model.summary()


def gan_to_core(network, aggregates, aggregate_dimensions, batch_size):
    results = gan_to_voxels(network, aggregates, batch_size)
    return voxels_to_core(results, aggregate_dimensions)


def gan_to_voxels_tfdata(network, aggregate_iterator):
    results = list()

    for aggregate, _ in aggregate_iterator:
        results.extend(network.predict(aggregate))

    return results


def gan_to_voxels(network, aggregates, batch_size):
    results = list()
    max_size = math.ceil(len(aggregates) / batch_size)
    for ind in tqdm(range(max_size),
                    desc=mt.get_notice("Generating synthetic binder voxels from GAN"),
                    total=max_size):
        start_ind = ind * batch_size
        end_ind = min(start_ind + batch_size, len(aggregates))

        result = network.predict(aggregates[start_ind:end_ind])

        results.extend(result)

    results = np.array(results, dtype="float32")
    return np.squeeze(results)

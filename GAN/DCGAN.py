import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from GAN import AbstractGAN
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Activation, Conv3D, Conv3DTranspose as Deconv3D, BatchNormalization, LeakyReLU
from ExperimentTools import DataVisualiser as dv
from ExperimentTools.MethodologyLogger import Logger
from ImageTools.CoreAnalysis.CoreVisualiser import save_mesh, voxels_to_mesh
from ImageTools.VoxelProcessor import voxels_to_core
from Settings.MessageTools import print_notice
from Settings import SettingsManager as sm, MessageTools as mt, MachineLearningManager as mlm, DatabaseManager as dm

#strategy = \
#    tf.distribute.experimental.MultiWorkerMirroredStrategy(tf.distribute.experimental.CollectiveCommunication.AUTO)
strategy = tf.distribute.experimental.CentralStorageStrategy()
#strategy = tf.distribute.MirroredStrategy()


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

        vox_res = int(sm.configuration.get("VOXEL_RESOLUTION"))

        data_shape = (vox_res, vox_res, vox_res, 1)

        optimizer = optimizers.Adam(1e-4)

        masked_vol = Input(shape=data_shape)

        with strategy.scope():
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

        d_loss = 0.5 * tf.add(d_loss_real, d_loss_fake)

        return d_loss, g_loss

    @classmethod
    def train_network_tfdata(cls, epochs, batch_size, dataset_iterator, core_animation_data=None):
        valid = tf.fill((batch_size, 1), 0.9)
        fake = tf.zeros((batch_size, 1))

        for epoch in range(epochs):
            batch_no = 1
            d_loss = []
            g_loss = []

            for features, labels in dataset_iterator:
                with strategy.scope():
                    d_loss, g_loss = cls.train_step(features, labels, valid, fake)

                print_notice("\rEpoch %d (Batch %d) [DIS loss: %f, acc: %.2f%%] [GEN loss: %f, mse: %f]"
                             % (epoch, batch_no, d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1]), end='')

                batch_no += 1

            print_notice("\rEpoch %d [DIS loss: %f, acc: %.2f%%] [GEN loss: %f, mse: %f]"
                         % (epoch, d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1]))

    @classmethod
    def train_network(cls, epochs, batch_size, features, labels, core_animation_data=None):
        print_notice("Preparing feature/label matrices...", mt.MessagePrefix.INFORMATION)
        if isinstance(features, list):
            features = np.asarray(features)
        if not len(features.shape) == 5:
            if len(features.shape) > 5:
                features = np.squeeze(features)
            elif len(features.shape) == 4:
                features = np.expand_dims(features, 4)

        if isinstance(labels, list):
            labels = np.asarray(labels)

        if not len(labels.shape) == 5:
            labels = np.expand_dims(np.array(labels), 4)
        print_notice("Matrices are now ready for machine learning input", mt.MessagePrefix.SUCCESS)

        Logger.print("Training network with: " + str(epochs) + " EPOCHS, " + str(batch_size) + " BATCH SIZE")

        x = []
        discriminator_losses = []
        discriminator_accuracies = []
        generator_losses = []
        generator_MSEs = []

        fig = plt.figure()

        gen_error_ax = fig.add_subplot(3, 1, 1)
        dis_error_ax = fig.add_subplot(3, 1, 2)
        acc_ax = fig.add_subplot(3, 1, 3)

        if sm.display_available:
            plt.show(block=False)

        def animate(_):
            dv.plot_training_data(generator_losses, generator_MSEs, discriminator_losses, discriminator_accuracies, x=x,
                                  gen_error_ax=gen_error_ax, dis_error_ax=dis_error_ax, acc_ax=acc_ax)
        # One sided label smoothing
        valid = np.full((batch_size, 1), 0.9)
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, len(features), batch_size)

            # This is the binder generated for a given aggregate arrangement
            if mlm.get_available_gpus() == 2:
                with tf.device('gpu:1'):
                    gen_missing = cls.generator.predict(features[idx] * 2.0 - 1.0)
            else:
                gen_missing = cls.generator.predict(features[idx] * 2.0 - 1.0)

            if mlm.get_available_gpus() == 2:
                with tf.device('gpu:0'):
                    # This trains the discriminator on real samples
                    d_loss_real = cls.discriminator.train_on_batch(labels[idx] * 2.0 - 1.0, valid)
                    # This trains the discriminator on fake samples
                    d_loss_fake = cls.discriminator.train_on_batch(gen_missing * 2.0 - 1.0, fake)
            else:
                d_loss_real = cls.discriminator.train_on_batch(labels[idx] * 2.0 - 1.0, valid)
                d_loss_fake = cls.discriminator.train_on_batch(gen_missing, fake)
            d_loss = 0.5 * tf.add(d_loss_real, d_loss_fake)

            if mlm.get_available_gpus() == 2:
                with tf.device('gpu:1'):
                    g_loss = cls.adversarial.train_on_batch(features[idx] * 2.0 - 1.0, [labels[idx] * 2.0 - 1.0, valid])
            else:
                g_loss = cls.adversarial.train_on_batch(features[idx] * 2.0 - 1.0, [labels[idx] * 2.0 - 1.0, valid])

            Logger.print("%d [DIS loss: %f, acc: %.2f%%] [GEN loss: %f, mse: %f]"
                         % (epoch, d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1]))

            discriminator_losses.append(d_loss[0])
            discriminator_accuracies.append(d_loss[1])
            generator_losses.append(g_loss[0])
            generator_MSEs.append(g_loss[1])

            if sm.display_available:
                x.append(len(x) + 1)
                animate(epoch)
                plt.draw()
                plt.pause(0.1)

            if dm.database_connected:
                sql = "INSERT INTO training (ExperimentID, Fold, Epoch, TrainingSet, DiscriminatorLoss, " \
                      "DiscriminatorAccuracy, GeneratorLoss, GeneratorMSE) " \
                      "VALUES (%s, %s, %s, %s, %s, %s, %s, %s);"

                val = (Logger.experiment_id, Logger.current_fold + 1, epoch + 1, Logger.current_set + 1,
                       float(d_loss[0]), float(d_loss[1]), float(g_loss[0]), float(g_loss[1]))

                dm.db_cursor.execute(sql, val)
                dm.db.commit()

            if core_animation_data is not None and len(core_animation_data) == 3:
                core = gan_to_core(cls.adversarial, core_animation_data[0], core_animation_data[1])
                mesh = voxels_to_mesh(core)
                save_mesh(mesh, core_animation_data[2] + 'Epoch_' + str(epoch) + '.stl')

            # im.save_voxel_image_collection(gen_missing, fm.SpecialFolder.VOXEL_DATA, "figures/postGAN/generated")
            # im.save_voxel_image_collection(labels, fm.SpecialFolder.VOXEL_DATA, "figures/postGAN/expected")

        return (discriminator_losses, discriminator_accuracies), (generator_losses, generator_MSEs)

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
        with strategy.scope():
            model = Sequential()

            for level in range(0, encoder_levels):
                if level == 0:
                    model.add(Conv3D(initial_filters * (pow(2, level)), kernel_size=kernel_size, strides=strides,
                                     input_shape=voxel_shape, padding="same"))
                else:
                    model.add(
                        Conv3D(initial_filters * (pow(2, level)), kernel_size=kernel_size, strides=strides, padding="same"))
                model.add(LeakyReLU(alpha=activation_alpha))
                if sm.configuration.get("TRAINING_USE_BATCH_NORMALISATION") == "True":
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
        with strategy.scope():
            model = Sequential()

            for level in range(0, encoder_levels):
                if level == 0:
                   model.add(Conv3D(initial_filters * (pow(2, level)),
                                     kernel_size=kernel_size, strides=strides, input_shape=voxel_shape, padding="same"))
                else:
                   model.add(Conv3D(initial_filters * (pow(2, level)),
                                    kernel_size=kernel_size, strides=strides, padding="same"))
                model.add(LeakyReLU(alpha=activation_alpha))
                if sm.configuration.get("TRAINING_USE_BATCH_NORMALISATION") == "True":
                    model.add(BatchNormalization(momentum=normalisation_momentum))

            for level in range(encoder_levels - 1, 0, -1):
                model.add(Deconv3D(initial_filters * pow(2, level - 1),
                                   kernel_size=kernel_size, strides=strides, padding="same"))
                model.add(Activation("relu"))
                if sm.configuration.get("TRAINING_USE_BATCH_NORMALISATION") == "True":
                    model.add(BatchNormalization(momentum=normalisation_momentum))

            model.add(Deconv3D(channels, kernel_size=kernel_size, strides=strides, padding="same"))
            model.add(Activation("tanh"))

            input_voxel = Input(shape=voxel_shape)
            gen_missing = model(input_voxel)

            self._model = Model(input_voxel, gen_missing)

        model.summary()


def gan_to_core(network, aggregates, aggregate_dimensions):
    results = gan_to_voxels(network, aggregates)
    return voxels_to_core(results, aggregate_dimensions)


def gan_to_voxels(network, aggregates):
    results, _ = network.predict(aggregates)
    results = results * 127.5 + 127.5
    results = np.array(results, dtype=np.uint8)
    np.put(results, [np.argwhere(results == 128)], 127)
    return np.squeeze(results)

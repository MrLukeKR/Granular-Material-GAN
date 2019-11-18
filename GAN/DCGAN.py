from ExperimentTools import MethodologyLogger
from GAN import AbstractGAN
from keras import Sequential, optimizers
from keras.models import Model
from keras.layers import Input, Flatten, Dense
from keras.layers.core import Activation
from keras.layers.convolutional import Conv3D, Deconv3D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from ExperimentTools.MethodologyLogger import Logger
from ImageTools import ImageManager as im, VoxelProcessor as vp
from Settings import FileManager as fm

import numpy as np


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
        self.create_network(data)

    @classmethod
    def create_network(cls, data):
        Logger.print("Initialising Generator Adversarial Network...")

        channels = 1
        data_shape = (len(data[0]), len(data[0][0]), len(data[0][0][0]), 1)

        optimizer = optimizers.Adam(0.0002, 0.5)

        cls.discriminator.compile(loss='binary_crossentropy',
                                  optimizer=optimizer,
                                  metrics=['accuracy'])

        masked_vol = Input(shape=data_shape)
        gen_missing = cls.generator(masked_vol)

        cls.discriminator.trainable = False

        valid = cls.discriminator(gen_missing)

        cls.adversarial = Model(masked_vol, [gen_missing, valid])
        cls.adversarial.compile(loss=['mse', 'binary_crossentropy'],
                                loss_weights=[0.999, 0.001],
                                optimizer=optimizer)

    @classmethod
    def train_network(cls, epochs, batch_size, features, labels):
        Logger.print("Training network with: " + str(epochs) + " EPOCHS, " + str(batch_size) + " BATCH SIZE")

        features = np.expand_dims(np.array(features), 5)
        labels = np.expand_dims(np.array(labels), 5)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        discriminator_losses = np.zeros((epochs, 1))
        generator_losses = np.zeros((epochs, 1))

        generated_images = list()

        for epoch in range(epochs):
            idx = np.random.randint(0, len(features), batch_size)

            # This is the binder generated for a given aggregate arrangement
            gen_missing = cls.generator.predict(features[idx])

            generated_images += gen_missing
            # This trains the discriminator on real samples
            d_loss_real = cls.discriminator.train_on_batch(labels[idx], valid)
            # This trains the discriminator on fake samples
            d_loss_fake = cls.discriminator.train_on_batch(gen_missing, fake)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = cls.adversarial.train_on_batch(features[idx], [labels[idx], valid])

            Logger.print("%d [DIS loss: %f, acc: %.2f%%] [GEN loss: %f, mse: %f]" % (epoch,
                                                                              d_loss[0],
                                                                              100 * d_loss[1],
                                                                              g_loss[0],
                                                                              g_loss[1]))

            discriminator_losses[epoch] = d_loss[0]
            generator_losses[epoch] = g_loss[0]

            sql = "INSERT INTO training (ExperimentID, Fold, Epoch, TrainingSet, DiscriminatorLoss, " \
                  "DiscriminatorAccuracy, GeneratorLoss, GeneratorMSE) " \
                  "VALUES (%s, %s, %s, %s, %s, %s, %s, %s);"

            val = (Logger.experiment_id, Logger.current_fold + 1, epoch + 1, Logger.current_set + 1,
                   float(d_loss[0]), float(d_loss[1]), float(g_loss[0]), float(g_loss[1]))

            MethodologyLogger.db_cursor.execute(sql, val)
            MethodologyLogger.db.commit()

            # im.save_voxel_image_collection(gen_missing, fm.SpecialFolder.VOXEL_DATA, "figures/postGAN/generated")
            # im.save_voxel_image_collection(labels, fm.SpecialFolder.VOXEL_DATA, "figures/postGAN/expected")

        return discriminator_losses, generator_losses, generated_images

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
        Logger.print("\tInitialising Deep Convolutional Generative Adversarial Network (Discriminator)")

        x = len(voxels[0])
        y = len(voxels[0][0])
        z = len(voxels[0][0][0])
        w = len(voxels)

        channels = 1  # TODO: Make this variable from the settings file

        Logger.print("\t\t Input size is: " + str(w) + " (" + str(x) + " * " + str(y) + " * " + str(z) + ") voxels")

        voxel_shape = (x, y, z, channels)

        # START MODEL BUILDING

        model = Sequential()

        for level in range(0, encoder_levels):
            if level == 0:
                model.add(Conv3D(initial_filters * (pow(2, level)), kernel_size=kernel_size, strides=strides,
                                 input_shape=voxel_shape, padding="same"))
            else:
                model.add(
                    Conv3D(initial_filters * (pow(2, level)), kernel_size=kernel_size, strides=strides, padding="same"))
            model.add(LeakyReLU(alpha=activation_alpha))
            model.add(BatchNormalization(momentum=normalisation_momentum))

        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))

        model.summary(print_fn=Logger.print)

        input_voxel = Input(shape=voxel_shape)
        verdict = model(input_voxel)

        self._model = Model(input_voxel, verdict)


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
        Logger.print("\tInitialising Deep Convolutional Generative Adversarial Network (Generator)")

        x = len(voxels[0])
        y = len(voxels[0][0])
        z = len(voxels[0][0][0])
        w = len(voxels)

        channels = 1 # TODO: Make this variable from the settings file

        Logger.print("\t\t Input size is: " + str(w) + " (" + str(x) + " * " + str(y) + " * " + str(z) + ") voxels")

        voxel_shape = (x, y, z, channels)


        # START MODEL BUILDING

        model = Sequential()

        for level in range(0, encoder_levels):
            if level == 0:
                model.add(Conv3D(initial_filters * (pow(2, level)), kernel_size=kernel_size, strides=strides, input_shape=voxel_shape, padding="same"))
            else:
                model.add(Conv3D(initial_filters * (pow(2, level)), kernel_size=kernel_size, strides=strides, padding="same"))
            model.add(LeakyReLU(alpha=activation_alpha))
            model.add(BatchNormalization(momentum=normalisation_momentum))

        for level in range(encoder_levels - 1, 0, -1):
            model.add(Deconv3D(initial_filters * pow(2, level - 1), kernel_size=kernel_size, strides=strides, padding="same"))
            model.add(Activation("relu"))
            model.add(BatchNormalization(momentum=normalisation_momentum))

        model.add(Deconv3D(channels, kernel_size=kernel_size, strides=strides, padding="same"))
        model.add(Activation("tanh"))

        model.summary(print_fn=Logger.print)

        input_voxel = Input(shape=voxel_shape)
        gen_missing = model(input_voxel)

        self._model = Model(input_voxel, gen_missing)

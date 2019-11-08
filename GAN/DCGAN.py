from GAN import AbstractGAN

from keras import Sequential
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Flatten, Dense
from keras.layers.core import Activation
from keras.layers.convolutional import Conv3D, Deconv3D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

import numpy as np


class Network(AbstractGAN.Network):
    _model = None
    _discriminator = None
    _generator = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

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
        print("Initialising Generator Adversarial Network...")

        x = len(data[0])
        y = len(data[0][0])
        z = len(data[0][0][0])
        w = len(data)

        channels = 1  # TODO: Make this variable from the settings file

        discriminator_optimizer = Adam(0.00001, 0.5)
        combined_optimizer = Adam(0.0001, 0.5)

        cls._discriminator.trainable = False

        cls._discriminator.compile(loss="binary_crossentropy",
                                   optimizer=discriminator_optimizer,
                                   metrics=["accuracy"])

        input_voxel = Input(shape=(x, y, z, channels))
        gen_missing = cls._generator(input_voxel)

        verdict = cls._discriminator(gen_missing)

        cls._model = Model(input_voxel, [gen_missing, verdict])
        cls._model.compile(loss=["mse", "binary_crossentropy"],
                           loss_weights=[0.999, 0.001],
                           optimizer=combined_optimizer)

        return cls._model

    @classmethod
    def train_network(cls, epochs, batch_size, training_set):
        print("Training network with: " + str(epochs) + " EPOCHS, " + str(batch_size) + " BATCH SIZE")

        features, labels = training_set
        features = np.expand_dims(np.array(features), 5)
        labels = np.expand_dims(np.array(labels), 5)

        valid = np.full((batch_size, 1), 0.9)
        invalid = np.zeros((batch_size, 1))

        disciminator_losses = np.zeros((epochs, 1))
        generator_losses = np.zeros((epochs, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, len(features), batch_size)

            gen_missing = cls._generator.predict(features[idx])

            discriminator_loss_valid = cls._discriminator.train_on_batch(labels[idx], valid)
            disciminator_loss_invalid = cls._discriminator.train_on_batch(gen_missing, invalid)
            discriminator_loss = 0.5 * np.add(disciminator_loss_invalid, discriminator_loss_valid)

            generator_loss = cls._model.train_on_batch(features[idx], [labels[idx], valid])

            print("%d [DIS loss: %f, acc: %.2f%%] [GEN loss: %f, mse: %f]" % (epoch,
                                                                              discriminator_loss[0],
                                                                              100 * discriminator_loss[1],
                                                                              generator_loss[0],
                                                                              generator_loss[1]))

            disciminator_losses[epoch] = discriminator_loss[0]
            generator_losses[epoch] = generator_loss[0]


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

    def __init__(self, voxels, strides, kernel_size):
        print("\tInitialising Deep Convolutional Generative Adversarial Network (Discriminator)")

        x = len(voxels[0])
        y = len(voxels[0][0])
        z = len(voxels[0][0][0])
        w = len(voxels)

        channels = 1  # TODO: Make this variable from the settings file

        # VARIABLES ----------------
        initial_filters = 32
        activation_alpha = 0.2
        normalisation_momentum = 0.8
        encoder_levels = 3
        # --------------------------

        print("\t\t Input size is: " + str(w) + " (" + str(x) + " * " + str(y) + " * " + str(z) + ") voxels")

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

        model.summary()

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

    def __init__(self, voxels, strides, kernel_size):
        print("\tInitialising Deep Convolutional Generative Adversarial Network (Generator)")

        x = len(voxels[0])
        y = len(voxels[0][0])
        z = len(voxels[0][0][0])
        w = len(voxels)

        channels = 1 # TODO: Make this variable from the settings file

        # VARIABLES ----------------
        initial_filters = 128
        activation_alpha = 0.2
        normalisation_momentum = 0.8
        encoder_levels = 3
        # --------------------------

        print("\t\t Input size is: " + str(w) + " (" + str(x) + " * " + str(y) + " * " + str(z) + ") voxels")

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

        model.summary()

        input_voxel = Input(shape=voxel_shape)
        gen_missing = model(input_voxel)

        self._model = Model(input_voxel, gen_missing)

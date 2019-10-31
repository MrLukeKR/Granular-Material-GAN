from GAN import AbstractGAN

from keras import Sequential
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation
from keras.layers.convolutional import Conv3D, Deconv3D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


class Network(AbstractGAN.Network):
    def __init__(self, data):
        self.create_network(data)

    @classmethod
    def train_network(cls, training_set):
        pass

    @classmethod
    def test_network(cls, testing_set):
        pass

    @classmethod
    def create_network(cls, data):
        print("Initialising Generator Adversarial Network...")
        cls.generator = DCGANGenerator(data, strides=(2, 2, 2), kernelsize=(4, 4, 4), train=True)
        cls.discriminator = DCGANDiscriminator(0.2, (2, 2, 2), (4, 4, 4), True)

        discongen = Sequential()
        discongen.add(cls.generator.model)
        discongen.trainable = False
        discongen.add(cls.discriminator.model)

        model = Sequential()
        model.add(cls.generator.model)
        cls.discriminator.trainable = False
        model.add(cls.discriminator.model)

        generator_optimisation = Adam(lr=0.01, beta_1=0.5)
        discriminator_optimisation = Adam(lr=0.000001, beta_1=0.9)

        cls.generator.model.compile(loss='binary_crossentropy', optimizer="SGD")
        discongen.compile(loss='binary_crossentropy', optimizer=generator_optimisation)

        cls.discriminator.trainable = True
        cls.discriminator.model.compile(loss='binary_crossentropy', optimizer=discriminator_optimisation)

        batchsize = 30

        return cls.discriminator, cls.generator


class DCGANDiscriminator:
    _model = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def __init__(self, leak_value, strides, kernel_size, train):
        print("\tInitialising Deep Convolutional Generative Adversarial Network (Discriminator)")

        cubesize = 64

        inputs = Input(shape=(cubesize, cubesize, cubesize, 1))

        d1 = Conv3D(filters=64, kernel_size=kernel_size,
                    strides=strides, kernel_initializer='glorot_normal',
                    bias_initializer='zeros', padding='same')(inputs)
        d1 = BatchNormalization()(d1, training=train)
        d1 = LeakyReLU(leak_value)(d1)

        d2 = Conv3D(filters=128, kernel_size=kernel_size,
                    strides=strides, kernel_initializer='glorot_normal',
                    bias_initializer='zeros', padding='same')(d1)
        d2 = BatchNormalization()(d2, training=train)
        d2 = LeakyReLU(leak_value)(d2)

        d3 = Conv3D(filters=256, kernel_size=kernel_size,
                    strides=strides, kernel_initializer='glorot_normal',
                    bias_initializer='zeros', padding='same')(d2)
        d3 = BatchNormalization()(d3, training=train)
        d3 = LeakyReLU(leak_value)(d3)

        d4 = Conv3D(filters=512, kernel_size=kernel_size,
                    strides=strides, kernel_initializer='glorot_normal',
                    bias_initializer='zeros', padding='same')(d3)
        d4 = BatchNormalization()(d4, training=train)
        d4 = LeakyReLU(leak_value)(d4)

        d5 = Conv3D(filters=1, kernel_size=kernel_size,
                    strides=(1, 1, 1), kernel_initializer='glorot_normal',
                    bias_initializer='zeros', padding='valid')(d4)
        d5 = BatchNormalization()(d5, training=train)
        d5 = Activation(activation='sigmoid')(d5)

        self.model = Model(inputs=inputs, outputs=d5)
        # model.summary()


class DCGANGenerator:
    _model = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def __init__(self, voxels, strides, kernelsize, train):
        print("\tInitialising Deep Convolutional Generative Adversarial Network (Generator)")

        x = len(voxels[0])
        y = len(voxels[0][0])
        z = len(voxels[0][0][0])
        w = len(voxels)

        channels = 1

        filters = 512

        print("\t\t Input size is: " + str(w) + " (" + str(x) + " * " + str(y) + " * " + str(z) + ") voxels")

        inputs = Input(shape=(x, y, z, channels))

        g1 = Deconv3D(filters=filters, kernel_size=kernelsize,
                      strides=strides, kernel_initializer='glorot_normal',
                      bias_initializer='zeros', padding='valid')(inputs)
        g1 = BatchNormalization()(g1, training=train)

        g1 = Activation(activation='relu')(g1)

        filters = int(filters / 2)

    # ===========================================================================

        g2 = Deconv3D(filters=filters, kernel_size=kernelsize,
                      strides=strides, kernel_initializer='glorot_normal',
                      bias_initializer='zeros', padding='same')(g1)
        g2 = BatchNormalization()(g2, training=train)
        g2 = Activation(activation='relu')(g2)

        filters = int(filters / 2)
    # ===========================================================================

        g3 = Deconv3D(filters=filters, kernel_size=kernelsize,
                      strides=strides, kernel_initializer='glorot_normal',
                      bias_initializer='zeros', padding='same')(g2)
        g3 = BatchNormalization()(g3, training=train)
        g3 = Activation(activation='relu')(g3)

        filters = int(filters / 2)
    # ===========================================================================

        g4 = Deconv3D(filters=filters, kernel_size=kernelsize,
                      strides=strides, kernel_initializer='glorot_normal',
                      bias_initializer='zeros', padding='same')(g3)
        g4 = BatchNormalization()(g4, training=train)
        g4 = Activation(activation='relu')(g4)

    # ===========================================================================

        g5 = Deconv3D(filters=1, kernel_size=kernelsize,
                      strides=strides, kernel_initializer='glorot_normal',
                      bias_initializer='zeros', padding='same')(g4)
        g5 = BatchNormalization()(g5, training=train)
        g5 = Activation(activation='sigmoid')(g5)

        self.model = Model(inputs=inputs, outputs=g5)

        # model.summary()

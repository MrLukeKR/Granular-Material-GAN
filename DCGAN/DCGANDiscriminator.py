from keras.layers import Input
from keras.layers.convolutional import Conv3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.models import Model

def Discriminator3D(leakvalue, strides, kernelsize, train):
    print("\tInitialising Deep Convolutional Generative Adversarial Network (Discriminator)")

    cubesize = 64

    inputs = Input(shape=(cubesize, cubesize, cubesize, 1))

    d1 = Conv3D(filters=64, kernel_size=kernelsize,
                  strides=strides, kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='same')(inputs)
    d1 = BatchNormalization()(d1, training=train)
    d1 = LeakyReLU(leakvalue)(d1)

    d2 = Conv3D(filters=128, kernel_size=kernelsize,
                  strides=strides, kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='same')(d1)
    d2 = BatchNormalization()(d2, training=train)
    d2 = LeakyReLU(leakvalue)(d2)

    d3 = Conv3D(filters=256, kernel_size=kernelsize,
                  strides=strides, kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='same')(d2)
    d3 = BatchNormalization()(d3, training=train)
    d3 = LeakyReLU(leakvalue)(d3)

    d4 = Conv3D(filters=512, kernel_size=kernelsize,
                  strides=strides, kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='same')(d3)
    d4 = BatchNormalization()(d4, training=train)
    d4 = LeakyReLU(leakvalue)(d4)

    d5 = Conv3D(filters=1, kernel_size=kernelsize,
                  strides=(1, 1, 1), kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='valid')(d4)
    d5 = BatchNormalization()(d5, training=train)
    d5 = Activation(activation='sigmoid')(d5)

    model = Model(inputs=inputs, outputs=d5)
    model.summary()

    return model
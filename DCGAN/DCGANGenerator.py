from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation
from keras.layers.convolutional import Deconv3D
from keras.layers.normalization import BatchNormalization


def Generator3D(images, strides, kernelsize, train):
    print("\tInitialising Deep Convolutional Generative Adversarial Network (Generator)")

    if len(images) == 0:
        return

    x, y = images[0].shape
    z = len(images)

    channels = 1

    filters = 512

    print("\t\t Input size is: (" + str(x) + "px * " + str(y) + "px) * " + str(z) + " slices")

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

    model = Model(inputs=inputs, outputs=g5)
    model.summary()

    return model

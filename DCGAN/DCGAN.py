from keras import Sequential
from keras.optimizers import Adam

from DCGAN import DCGANGenerator, DCGANDiscriminator


def initialise_network(volumes):
    print("Initialising Generator Adversarial Network...")
    generator = DCGANGenerator.Generator3D(volumes, strides=(2, 2, 2), kernelsize=(4, 4, 4), train=True)
    discriminator = DCGANDiscriminator.Discriminator3D(0.2, (2, 2, 2), (4, 4, 4), True)

    discongen = Sequential()
    discongen.add(generator)
    discongen.trainable = False
    discongen.add(discriminator)

    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)

    generatoroptimisation = Adam(lr=0.01, beta_1=0.5)
    discriminatoroptimisation = Adam(lr=0.000001, beta_1=0.9)

    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    discongen.compile(loss='binary_crossentropy', optimizer=generatoroptimisation)

    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=discriminatoroptimisation)

    batchsize = 30
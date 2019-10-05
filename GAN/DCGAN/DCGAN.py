from keras import Sequential
from keras.optimizers import Adam

from GAN.DCGAN.Discriminator import DCGANDiscriminator
from GAN.DCGAN.Generator import DCGANGenerator

generator = None
discriminator = None


def initialise_network(data):
    print("Initialising Generator Adversarial Network...")
    generator = DCGANGenerator(data, strides=(2, 2, 2), kernelsize=(4, 4, 4), train=True)
    discriminator = DCGANDiscriminator(0.2, (2, 2, 2), (4, 4, 4), True)

    discongen = Sequential()
    discongen.add(generator.model)
    discongen.trainable = False
    discongen.add(discriminator.model)

    model = Sequential()
    model.add(generator.model)
    discriminator.trainable = False
    model.add(discriminator.model)

    generator_optimisation = Adam(lr=0.01, beta_1=0.5)
    discriminator_optimisation = Adam(lr=0.000001, beta_1=0.9)

    generator.model.compile(loss='binary_crossentropy', optimizer="SGD")
    discongen.compile(loss='binary_crossentropy', optimizer=generator_optimisation)

    discriminator.trainable = True
    discriminator.model.compile(loss='binary_crossentropy', optimizer=discriminator_optimisation)

    batchsize = 30
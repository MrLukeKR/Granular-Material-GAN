import numpy as np
import ImageUtils
from DCGAN import DCGANGenerator, DCGANDiscriminator
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.python.client import device_lib
from mayavi import mlab

print("   Optimal Material Generator using Generative Adversarial Networks   ")
print("                    Developed by ***REMOVED*** (BSc)                    ")
print("In fulfilment of Doctor of Engineering at the University of Nottingham")
print("----------------------------------------------------------------------")
print()

print("Running hardware checks...")
print(device_lib.list_local_devices())

numepochs = 100

visdim = 50

# Do image loading and preprocesing
directory = "F:/Windows/Documents/18-1415/"
#directory = "/data/CT-Scans/Aggregate CT Scans/Brian Atkinson - Mustafa - Asphalt Cores/18-1415/"
myCollection = ImageUtils.ImageCollection()
myCollection.loadImagesFromDirectory(directory)

images = myCollection.thresholdImages()
images = myCollection.resizeimages(images, 0.01)

#myCollection.viewImages(images)

volumes = np.asarray(images, dtype=np.float32)
volumes = np.transpose(volumes, (1, 2, 0))

print("Visualising Air-Voids...")
mlab.contour3d(volumes, contours=2)
mlab.show()

imheight, imwidth, imlayers = volumes.shape

# Initialise GAN
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

vX, vY, vZ = volumes.shape

noise = np.random.normal(0, 1, size=(vX, vY, vZ))

mlab.contour3d(noise, contours=2)
mlab.show()
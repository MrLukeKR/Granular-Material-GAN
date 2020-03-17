from skimage import io, transform, filters
from skimage.feature import canny
from imageio import mimsave
from scipy import ndimage
from skimage.viewer import CollectionViewer
from os import walk
from tqdm import tqdm
from ExperimentTools.MethodologyLogger import Logger


import numpy as np


class ImageCollection:
    images = list()
    segmentedImages = list()
    closedImages = list()
    filledImages = list()
    threshimages = list()

    def __init__(self):
        pass

    @staticmethod
    def view_images(images):
        viewer = CollectionViewer(images)
        viewer.show()

    @staticmethod
    def constrain_images(images, dimensions):
        Logger.print("Constraining " + str(len(images)) + " images to " + str(dimensions) + "...")
        for i in tqdm(range(len(images))):
            images[i] = transform.resize(images[i], output_shape=dimensions)

        return images

    @staticmethod
    def resize_images(images, factor):
        Logger.print("Resizing " + str(len(images)) + " images by a factor of " + str(factor) + "...")
        x, y = images[0].shape
        x = int(x * factor)
        y = int(y * factor)

        for i in tqdm(range(len(images))):
            images[i] = transform.resize(images[i], output_shape=(x, y))

        return images

    def saveToGif(self):
        saveLocation = input("Where do you want to save the GIF?: ")
        mimsave(saveLocation, self.images)

    def thresholdImages(self):
        Logger.print("Thresholding " + str(len(self.images)) + " images...")
        for i in tqdm(range(len(self.images))):  # tqdm is a progress bar tool
            self.threshimages.append(self.images[i] < filters.thresholding.threshold_otsu(self.images[i]))

        return self.threshimages

    def segmentImages(self):
        Logger.print("Segmenting " + str(len(self.images)) + " images...")
        for i in tqdm(range(len(self.images))):  # tqdm is a progress bar tool
            aggregateEdges = canny(self.images[i])

            # Iterations set to 0 to require gaps to be closed infinitely until no change is found between an iteration
            closedAggregates = ndimage.binary_closing(aggregateEdges, structure=np.ones((3, 2)), iterations=0).astype(np.uint8)
            filledAggregates = ndimage.binary_fill_holes(closedAggregates).astype(np.uint8)

            self.segmentedImages.append(transform.rescale(aggregateEdges, 0.25))
            self.closedImages.append(transform.rescale(closedAggregates, 0.25))
            self.filledImages.append(transform.rescale(filledAggregates, 0.25))

        Logger.print()  # Print a new line after the process bar is finished
        Logger.print("Done")

    def applyThreshold(self): pass

    def loadImagesFromList(self, filelist):
        Logger.print("Loading " + str(len(filelist)) + " images")
        for i in tqdm(range(len(filelist))):  # tqdm is a progress bar tool
            image = io.imread(filelist[i], True)
            self.images.append(image)

        Logger.print()  # Print a new line after the process bar is finished

        if len(self.images) > 0:
            Logger.print("Loaded " + str(len(self.images)) + " images successfully!")
        else:
            Logger.print("ERROR: No images were loaded!")

    def loadImagesFromDirectory(self, directory):
        files = []

        for (dPaths, dNames, fNames) in walk(directory):
            files.extend([directory + '{0}'.format(i) for i in fNames])

        if len(files) > 0:
            self.loadImagesFromList(files)
        else:
            Logger.print("ERROR: ImageCollection file list is empty - Loading Failed")
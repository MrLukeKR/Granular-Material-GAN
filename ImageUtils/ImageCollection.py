from skimage import io, transform
from skimage.feature import canny
from imageio import mimsave
from scipy import ndimage
from skimage.viewer import CollectionViewer
from os import walk
from tqdm import tqdm
import numpy as np


class ImageCollection:
    images = list()
    segmentedImages = list()
    closedImages = list()
    filledImages = list()

    def __init__(self):
        pass

    def viewImages(self):
        viewer = CollectionViewer(self.images)
        viewer.show()

    def saveToGif(self):
        saveLocation = input("Where do you want to save the GIF?: ")
        mimsave(saveLocation, self.images)

    def segmentImages(self):
        print("Segmenting " + str(len(self.images)) + " images...")
        for i in tqdm(range(len(self.images))):  # tqdm is a progress bar tool
            aggregateEdges = canny(self.images[i])

            # Iterations set to 0 to require gaps to be closed infinitely until no change is found between an iteration
            closedAggregates = ndimage.binary_closing(aggregateEdges, structure=np.ones((3, 2)), iterations=0).astype(np.uint8)
            filledAggregates = ndimage.binary_fill_holes(closedAggregates).astype(np.uint8)

            self.segmentedImages.append(transform.rescale(aggregateEdges, 0.25))
            self.closedImages.append(transform.rescale(closedAggregates, 0.25))
            self.filledImages.append(transform.rescale(filledAggregates, 0.25))

        viewer = CollectionViewer(self.segmentedImages)
        viewer.show()

        viewer = CollectionViewer(self.closedImages)
        viewer.show()

        viewer = CollectionViewer(self.filledImages)
        viewer.show()

        print()  # Print a new line after the process bar is finished

        print("Saving images to GIF...")
        mimsave('/run/media/***REMOVED***/***REMOVED***/Preprocessing/segmentedRings.gif', self.closedImages)
        mimsave('/run/media/***REMOVED***/***REMOVED***/Preprocessing/segmentedBlobs.gif', self.filledImages)
        print("Done")


    def applyThreshold(self): pass

    def loadImagesFromList(self, filelist):
        print("Loading " + str(len(filelist)) + " images")
        for i in tqdm(range(len(filelist))):  # tqdm is a progress bar tool
            image = io.imread(filelist[i], True)
            self.images.append(image)

        print()  # Print a new line after the process bar is finished

        if len(self.images) > 0:
            print("Loaded " + str(len(self.images)) + " images successfully!")
        else:
            print("ERROR: No images were loaded!")

    def loadImagesFromDirectory(self, directory):
        files = []

        for (dPaths, dNames, fNames) in walk(directory):
            files.extend([directory + '{0}'.format(i) for i in fNames])

        if len(files) > 0:
            self.loadImagesFromList(files)
        else:
            print("ERROR: ImageCollection file list is empty - Loading Failed")
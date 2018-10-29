from skimage import io
from skimage.viewer import CollectionViewer
from os import walk
from tqdm import tqdm


class ImageCollection:
    images = list()

    def __init__(self):
        pass

    def viewImages(self):
        viewer = CollectionViewer(self.images)
        viewer.show()

    def loadImagesFromList(self, filelist):
        print("Loading " + str(len(filelist)) + " images")
        for i in tqdm(range(len(filelist))):  # tqdm is a progress bar tool
            image = io.imread(filelist[i], True)
            self.images.append(image)

        if len(self.images) > 0:
            print("Loaded " + str(len(self.images)) + " successfully!")
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
from ImageTools import ImageManager
import numpy as np


def segment_image(image):
    if len(image.shape) != 3:
        raise Exception("This segmentation method only accepts three-dimensional images (volumes). "
                        "The shape given is " + image.shape)

    # ImageManager.display_voxel(image)
    neighbourhoods = split_to_neighbourhoods(image)

    means = calculate_mean_histogram(neighbourhoods)
    medians = calculate_median_histogram(neighbourhoods)

    # 3D Otsu with Cuttlefish Algorithm
    # Based on the paper available at:
    #
    # I = Image
    # L = Gray Levels
    # N = Number of pixels
    # f(x,y) = Intensity of pixel
    # k x k = Neighbourhood function (k = 3)
    # g(x, y) = mean value
    # h(x, y) = median value


def split_to_neighbourhoods(voxel):
    k = 3
    neighbourhoods = list()
    dimensions = voxel.shape

    for x in range(0, dimensions[0]-k, k):
        for y in range(0, dimensions[1]-k, k):
            for z in range(0, dimensions[2]-k, k):
                neighbourhood = voxel[x:x+k, y:y+k, z:z+k]
                neighbourhoods.append(neighbourhood)

    return neighbourhoods


def calculate_mean_histogram(neighbourhoods):
    histogram = list()

    for n in neighbourhoods:
        histogram.append(np.mean(n))

    return histogram


def calculate_median_histogram(neighbourhoods):
    histogram = list()

    for n in neighbourhoods:
        histogram.append(np.median(n))

    return histogram

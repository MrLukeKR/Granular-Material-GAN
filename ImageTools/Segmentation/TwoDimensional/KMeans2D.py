import ImageTools.ImageManager as im
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans


def segment_image(image):
    if len(image.shape) != 2:
        raise Exception("This segmentation method only accepts two-dimensional images. "
                        "The shape given is " + image.shape)

    kmeans = KMeans(n_clusters=3).fit(np.reshape(image, (np.prod(image.shape), 1)))
    segment = np.reshape(kmeans.cluster_centers_[kmeans.labels_], image.shape)

    fig, ax = im.plt.subplots(1, 2, figsize=(10, 5))

    ax[0].set_title("Original Image")
    ax[0].imshow(np.reshape(image, (1024, 1024)))

    ax[1].set_title("Segmented Image")
    ax[1].imshow(np.reshape(segment, (1024, 1024)))

    im.plt.show()

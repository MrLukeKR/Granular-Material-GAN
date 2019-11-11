import numpy as np
from sklearn.cluster import KMeans


counter = 0


def segment_image(image):
    if len(image.shape) != 2:
        raise Exception("This segmentation method only accepts two-dimensional images. "
                        "The shape given is " + str(image.shape))

    kmeans = KMeans(n_clusters=3).fit(np.reshape(image, (np.prod(image.shape), 1)))
    segment = np.reshape(kmeans.cluster_centers_[kmeans.labels_], image.shape)

    voids = segment <= np.min(segment)
    aggregates = segment >= np.max(segment)

    binder = segment * ~(voids + aggregates)

    return voids, aggregates, binder, segment


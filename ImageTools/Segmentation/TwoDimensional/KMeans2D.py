import numpy as np
from sklearn.cluster import KMeans


def segment_image(image):
    if len(image.shape) != 2:
        raise Exception("This segmentation method only accepts two-dimensional images. "
                        "The shape given is " + str(image.shape))

    k_means = KMeans(n_clusters=3).fit(np.reshape(image, (np.prod(image.shape), 1)))
    segment = np.reshape(k_means.cluster_centers_[k_means.labels_], image.shape)

    voids = segment == np.min(segment)
    aggregates = segment == np.max(segment)
    binder = ~(aggregates + voids)

    return voids, aggregates, binder, segment


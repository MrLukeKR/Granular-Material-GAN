import ImageTools.ImageManager as im
import numpy as np
from skimage.filters import threshold_multiotsu


def segment_image(image):
    if len(image.shape) != 2:
        raise Exception("This segmentation method only accepts two-dimensional images. "
                        "The shape given is " + image.shape)

    seg = image

    fig, ax = im.plt.subplots(1, 2, figsize=(10, 3.5))

    ax[0].set_title("Original Image")
    ax[0].imshow(np.reshape(image, (1024, 1024)))

    ax[1].set_title("Segmented Image")
    ax[1].imshow(np.reshape(seg, (1024, 1024)))

    im.plt.show()

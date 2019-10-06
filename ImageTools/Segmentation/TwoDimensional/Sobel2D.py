import ImageTools.ImageManager as im
import numpy as np
from scipy import ndimage


def segment_image(image):
    if len(image.shape) != 2:
        raise Exception("This segmentation method only accepts two-dimensional images. "
                        "The shape given is " + image.shape)

    sobel_horizontal = np.array([np.array([1, 2, 1]),
                                 np.array([0, 0, 0]),
                                 np.array([-1, -2, -1])])
    sobel_vertical = np.array([np.array([-1, -0, 1]),
                               np.array([-2, 0, 2]),
                               np.array([-1, 0, 1])])

    out_h = ndimage.convolve(image, sobel_horizontal, mode='reflect')
    out_v = ndimage.convolve(image, sobel_vertical, mode='reflect')

    fig, ax = im.plt.subplots(2, 2, figsize=(10, 10))

    ax[0, 0].set_title("Original Image")
    ax[0, 0].imshow(np.reshape(image, (1024, 1024)))

    ax[1, 0].set_title("Segmented Image (Horz)")
    ax[1, 0].imshow(np.reshape(out_h, (1024, 1024)))

    ax[1, 1].set_title("Segmented Image (Vert)")
    ax[1, 1].imshow(np.reshape(out_v, (1024, 1024)))

    im.plt.show()

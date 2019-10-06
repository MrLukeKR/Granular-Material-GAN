import ImageTools.ImageManager as im
import numpy as np
from skimage.filters import threshold_otsu


def segment_image(voxel):
    if len(voxel.shape) != 3:
        raise Exception("This segmentation method only accepts three-dimensional images (volumes). "
                        "The shape given is " + voxel.shape)

    slices = deconstruct_voxel(voxel)

    for s in slices:
        if np.sum(s) == 0:
            continue

        seg = segment_slice(s)

        fig, ax = im.plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].imshow(np.reshape(s, (64, 64)))
        ax[1].imshow(np.reshape(seg, (64, 64)))

        im.plt.show()
        fig.close()


def segment_slice(slice):
    thresholds = threshold_otsu(slice, 3)

    return thresholds


def deconstruct_voxel(image):
    slices = list()
    image_size = image.shape[0]

    for z in range(0, image.shape[2]):
        slice = image[0:image_size, 0:image_size, z]
        slices.append(slice)

    return slices


def reconstruct_voxel(slices):
    pass
import numpy as np

from ImageTools.SmallestEnclosingCircle import is_in_circle


def find_background_pixels(cropped_core_slice, enclosing_circle):
    if enclosing_circle is None:
        raise ValueError("Enclosing circle does not exist!")

    for (x, y) in np.argwhere(cropped_core_slice == 0):
        cropped_core_slice[x, y] = int(is_in_circle(enclosing_circle, (x, y)))

    return cropped_core_slice

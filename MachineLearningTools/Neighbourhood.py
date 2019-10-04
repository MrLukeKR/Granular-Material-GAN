import numpy as np


def spherical_neighbour(center_index, r, shape):
    xc, yc, zc = center_index
    x = np.arange(0, 2 * r + 1)
    y = np.arange(0, 2 * r + 1)
    z = np.arange(0, 2 * r + 1)

    in_sphere = ((x[np.newaxis, :, :]-r) ** 2 + (y[:, np.newaxis, :]-r) ** 2 + (z[:, :, np.newaxis]-r) ** 2) < r ** 2
    in_sph_x, in_sph_y, in_sph_z = np.nonzero(in_sphere)

    in_sph_x += (xc - r)
    in_sph_y += (yc - r)
    in_sph_z += (zc - r)

    x_in_array = (0 <= in_sph_x) * (in_sph_x < shape[0])
    y_in_array = (0 <= in_sph_y) * (in_sph_y < shape[1])
    z_in_array = (0 <= in_sph_z) * (in_sph_z < shape[2])

    in_array = x_in_array * y_in_array * z_in_array
    return in_sph_x[in_array], in_sph_y[in_array], in_sph_z[in_array]


def gaussian_neighbour(shape, sigma=4, r=5):
    row, col, val = [], [], []
    for i, (x, y, z) in enumerate(np.ndindex(*shape)):
        neighbour_x, neighbour_y, neighbour_z = spherical_neighbour((x, y, z), r, shape)
        neighbour_value = np.exp(-((neighbour_x - x) ** 2 + (neighbour_y - y) ** 2 + (neighbour_z - z) ** 2) / sigma ** 2)
        ravel_index = np.ravel_multi_index([neighbour_x, neighbour_y, neighbour_z], shape)

        row.append(np.array([i] * len(neighbour_x)))
        col.append(ravel_index)
        # TODO: This may need a depth value somewhere
        val.append(neighbour_value)
    rows = np.hstack(row)
    cols = np.hstack(col)
    indices = np.vstack([rows, cols]).astype(np.int64)
    vals = np.hstack(val).astype(np.float)

    return indices, vals

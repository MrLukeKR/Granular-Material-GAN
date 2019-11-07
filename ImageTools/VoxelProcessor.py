import math
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import h5py

from Settings import FileManager as fm


from Settings import SettingsManager as sm


def split_voxel_collection_to_segments(voxels):
    aggregates = list()
    binders = list()
    voids = list()

    for voxel in voxels:
        a, b, v = split_voxel_to_segments(voxel)
        aggregates.append(a)
        binders.append(b)
        voids.append(v)

    return aggregates, binders, voids


def split_voxel_to_segments(voxel):
    aggregate = None
    binder = None
    void = None

    raise NotImplementedError("split_voxel_to_segments is not implemented!")


def split_to_voxels(volume_data, cubic_dimension):
    voxels = list()
    # Structure of voxel -> uint[cubicDimension][cubicDimension][cubicDimension]

    volume = np.array(volume_data,)

    # Voxel normalisation (is this necessary if we normalise the input images?
    #vol_min = volume.min(axis=(0, 1, 2), keepdims=True)
    #vol_max = volume.max(axis=(0, 1, 2), keepdims=True)
    #volume = (volume - vol_min)/(vol_max - vol_min)

    # This should be equal to a hypothetical voxelCountZ, since images are square
    voxel_count_x = len(volume) / cubic_dimension
    voxel_count_y = len(volume[0]) / cubic_dimension
    voxel_count_z = len(volume[0][0]) / cubic_dimension

    if \
            voxel_count_x != int(voxel_count_x) or \
            voxel_count_y != int(voxel_count_y) or \
            voxel_count_z != int(voxel_count_z):
        print("Voxel division resulted in a floating-point number:")

        if str(sm.configuration.get("VOXEL_RESOLVE_METHOD")).upper() == "LOSSY":
            print("\tUsing LOSSY solution")

            voxel_count_x = math.floor(voxel_count_x)
            voxel_count_y = math.floor(voxel_count_y)
        elif str(sm.configuration.get("VOXEL_RESOLVE_METHOD")).upper() == "PADDING":
            print("\tUsing PADDING solution")

            voxel_count_x = math.ceil(voxel_count_x)
            voxel_count_y = math.ceil(voxel_count_y)
        elif str(sm.configuration.get("VOXEL_RESOLVE_METHOD")).upper() == "EXTRAPOLATE":
            print("\tEXTRAPOLATE method has not been implemented yet!")

            # TODO: Write the extrapolating method of voxel segmentation

            pass
        elif str(sm.configuration.get("VOXEL_RESOLVE_METHOD")).upper() == "SHRINK":
            print("\tSHRINK method has not been implemented yet!")

            # TODO: Write the shrinking method of voxel segmentation

            pass
        elif str(sm.configuration.get("VOXEL_RESOLVE_METHOD")).upper() == "STRETCH":
            print("\tSTRETCH method has not been implemented yet!")

            # TODO: Write the stretching method of voxel segmentation

            pass

    pretext = "Separating data volume into " + str(int(voxel_count_x * voxel_count_y * voxel_count_z)) + " voxels..."

    print(pretext, end='\r', flush=True)

    voxel_count_x = int(voxel_count_x)
    voxel_count_y = int(voxel_count_y)
    voxel_count_z = int(voxel_count_z)

    for x in range(voxel_count_x):
        x_start = cubic_dimension * x
        x_end = x_start + cubic_dimension
        for z in range(voxel_count_z):
            z_start = cubic_dimension * z
            z_end = z_start + cubic_dimension
            for y in range(voxel_count_y):
                print(pretext + " Voxel [DEPTH " + str(z) + "][ROW " + str(y) + "][COL " + str(x) + "]", end='\r', flush=True)

                y_start = cubic_dimension * y
                y_end = y_start + cubic_dimension

                voxel = volume[x_start:x_end, y_start:y_end, z_start:z_end]

                if voxel.shape != (cubic_dimension, cubic_dimension, cubic_dimension):
                    resolver = str(sm.configuration.get("VOXEL_RESOLVE_METHOD")).upper()

                    print("FOUND NON-PERFECT VOXEL, RESOLVING WITH [" + resolver + "]...")

                    if resolver == "PADDING":
                        xPad = cubic_dimension - len(voxel)
                        if xPad == cubic_dimension:
                            continue

                        yPad = cubic_dimension - len(voxel[0])
                        if yPad == cubic_dimension:
                            continue

                        zPad = cubic_dimension - len(voxel[0][0])
                        if zPad == cubic_dimension:
                            continue

                        voxel = np.pad(voxel, ((0, xPad), (0, yPad), (0, zPad)), 'constant', constant_values=0)

                if voxel.shape == (cubic_dimension, cubic_dimension, cubic_dimension):
                    voxels.append(voxel)
                else:
                    print("!-- ERROR: Voxel was invalid size --!")

    print(pretext + " done!")
    return voxels


def save_voxels(voxels, location, filename):
    fm.create_if_not_exists(location)

    h5f = h5py.File(location + "/" + filename + ".h5", 'w')
    h5f.create_dataset(filename, data=voxels)
    h5f.close()


def save_voxel(voxel, location):
    if not isinstance(voxel, np.ndarray):
        raise TypeError("voxel must be of type 'np.ndarray'")


def plot_voxel(voxel):
    vox = np.array(voxel)

    x_len = len(vox)
    y_len = len(vox[0])
    z_len = len(vox[0][0])

    vox_size = max(x_len, y_len, z_len)

    if x_len < vox_size:
        print("Padding required along X axis")
    elif y_len < vox_size:
        print("Padding required along Y axis")
    elif z_len < vox_size:
        print("Padding required along Z axis")
    else:
        vox = vox.reshape((vox_size, vox_size, vox_size))

        if np.amax(vox) != 0:
            normalised = vox / np.amax(vox)
        else:
            normalised = np.zeros(vox.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if sm.USE_BW:
            colours = cm.gray(normalised)
        else:
            colours = cm.jet(normalised)
        ax.voxels(vox, facecolors=colours, edgecolors=colours)

        return fig
    return None

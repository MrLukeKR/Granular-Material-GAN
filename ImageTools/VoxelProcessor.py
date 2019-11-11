import math
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import h5py

from Settings import FileManager as fm, SettingsManager as sm
from ExperimentTools.MethodologyLogger import Logger


def split_to_voxels(volume_data, cubic_dimension):
    voxels = list()

    volume = np.array(volume_data,)

    # This should be equal to a hypothetical voxelCountZ, since images are square
    voxel_count_x = len(volume) / cubic_dimension
    voxel_count_y = len(volume[0]) / cubic_dimension
    voxel_count_z = len(volume[0][0]) / cubic_dimension

    if \
            voxel_count_x != int(voxel_count_x) or \
            voxel_count_y != int(voxel_count_y) or \
            voxel_count_z != int(voxel_count_z):
        Logger.print("Voxel division resulted in a floating-point number:")

        if str(sm.configuration.get("VOXEL_RESOLVE_METHOD")).upper() == "LOSSY":
            Logger.print("\tUsing LOSSY solution")

            voxel_count_x = math.floor(voxel_count_x)
            voxel_count_y = math.floor(voxel_count_y)
        elif str(sm.configuration.get("VOXEL_RESOLVE_METHOD")).upper() == "PADDING":
            Logger.print("\tUsing PADDING solution")

            voxel_count_x = math.ceil(voxel_count_x)
            voxel_count_y = math.ceil(voxel_count_y)
        elif str(sm.configuration.get("VOXEL_RESOLVE_METHOD")).upper() == "EXTRAPOLATE":
            Logger.print("\tEXTRAPOLATE method has not been implemented yet!")

            # TODO: Write the extrapolating method of voxel segmentation

            pass
        elif str(sm.configuration.get("VOXEL_RESOLVE_METHOD")).upper() == "SHRINK":
            Logger.print("\tSHRINK method has not been implemented yet!")

            # TODO: Write the shrinking method of voxel segmentation

            pass
        elif str(sm.configuration.get("VOXEL_RESOLVE_METHOD")).upper() == "STRETCH":
            Logger.print("\tSTRETCH method has not been implemented yet!")

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
                Logger.print(pretext + " Voxel [DEPTH " + str(z) + "][ROW " + str(y) + "][COL " + str(x) + "]", end='\r', flush=True)

                y_start = cubic_dimension * y
                y_end = y_start + cubic_dimension

                voxel = volume[x_start:x_end, y_start:y_end, z_start:z_end]

                if voxel.shape != (cubic_dimension, cubic_dimension, cubic_dimension):
                    resolver = str(sm.configuration.get("VOXEL_RESOLVE_METHOD")).upper()

                    Logger.print("FOUND NON-PERFECT VOXEL, RESOLVING WITH [" + resolver + "]...")

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
                    Logger.print("!-- ERROR: Voxel was invalid size --!")

    Logger.print(pretext + " done!")
    return voxels


def save_voxels(voxels, location, filename):
    fm.create_if_not_exists(location)

    filepath = location + "/" + filename + ".h5"

    Logger.print("Saving voxel collection to '" + filepath + "'... ", end='')
    h5f = h5py.File(filepath, 'w')
    h5f.create_dataset(filename, data=voxels)
    h5f.close()
    Logger.print("done!")


def load_voxels(location, filename):
    filepath = location + '/' + filename + ".h5"

    if not fm.file_exists(filepath):
        raise FileNotFoundError("There is no voxel at '" + filepath + "'")

    h5f = h5py.File(filepath, 'r')
    voxels = list()

    for key in h5f.keys():
        dataset = h5f.get(key)[()]
        voxels += list(voxel for voxel in dataset)

    return voxels


def plot_voxel(voxel):
    vox = np.array(voxel)

    x_len = len(vox)
    y_len = len(vox[0])
    z_len = len(vox[0][0])

    vox_size = max(x_len, y_len, z_len)

    if x_len < vox_size:
        Logger.print("Padding required along X axis")
    elif y_len < vox_size:
        Logger.print("Padding required along Y axis")
    elif z_len < vox_size:
        Logger.print("Padding required along Z axis")
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

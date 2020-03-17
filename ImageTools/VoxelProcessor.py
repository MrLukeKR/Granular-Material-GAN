import math
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import h5py

from Settings import MessageTools as mt
from Settings.MessageTools import print_notice

# DO NOT DELETE THIS! It shows as unused but it is vital to 3D projection
from mpl_toolkits.mplot3d import axes3d, Axes3D

from Settings import FileManager as fm, SettingsManager as sm
from ExperimentTools.MethodologyLogger import Logger


def load_materials(directory):
    aggregates = list()
    binders = list()

    print_notice("Loading voxels for core " + directory, mt.MessagePrefix.INFORMATION)

    fm.current_directory = directory.replace(fm.get_directory(fm.SpecialFolder.SEGMENTED_SCANS), '')

    if fm.current_directory[-1] != '/':
        fm.current_directory += '/'

    voxel_directory = fm.get_directory(fm.SpecialFolder.VOXEL_DATA) + fm.current_directory[0:-1]

    temp_voxels, dimensions = load_voxels(voxel_directory, "segment_" + sm.configuration.get("VOXEL_RESOLUTION"))

    temp_aggregates = np.where(temp_voxels == 255) * 255
    temp_binders = np.where(temp_voxels == 128) * 255

    for voxel_ind in range(len(temp_aggregates)):
        if np.min(temp_aggregates[voxel_ind]) != np.max(temp_aggregates[voxel_ind]) and \
                np.min(temp_binders[voxel_ind]) != np.max(temp_binders[voxel_ind]):
            binder = temp_binders[voxel_ind]
            aggregate = temp_aggregates[voxel_ind]

            aggregates.append(aggregate * 255)
            binders.append(binder * 255)

    print_notice("Loaded aggregates and binders", mt.MessagePrefix.SUCCESS)

    return dimensions, aggregates, binders


def voxels_to_volume(voxels, dimensions):

    pass


def volume_to_voxels(volume_data, cubic_dimension):
    voxels = list()

    volume = np.array(volume_data)

    # This should be equal to a hypothetical voxelCountZ, since images are square
    voxel_count_x = len(volume) / cubic_dimension
    voxel_count_y = len(volume[0]) / cubic_dimension
    voxel_count_z = len(volume[0][0]) / cubic_dimension

    if \
            voxel_count_x != int(voxel_count_x) or \
            voxel_count_y != int(voxel_count_y) or \
            voxel_count_z != int(voxel_count_z):
        print_notice("Voxel division resulted in a floating-point number:", mt.MessagePrefix.INFORMATION)

        if str(sm.configuration.get("VOXEL_RESOLVE_METHOD")).upper() == "LOSSY":
            print_notice("\tUsing LOSSY solution", mt.MessagePrefix.INFORMATION)

            voxel_count_x = math.floor(voxel_count_x)
            voxel_count_y = math.floor(voxel_count_y)
        elif str(sm.configuration.get("VOXEL_RESOLVE_METHOD")).upper() == "PADDING":
            print_notice("\tUsing PADDING solution", mt.MessagePrefix.INFORMATION)

            voxel_count_x = math.ceil(voxel_count_x)
            voxel_count_y = math.ceil(voxel_count_y)
        elif str(sm.configuration.get("VOXEL_RESOLVE_METHOD")).upper() == "EXTRAPOLATE":
            print_notice("\tEXTRAPOLATE method has not been implemented yet!", mt.MessagePrefix.ERROR)
            raise NotImplementedError
            # TODO: Write the extrapolating method of voxel segmentation

            pass
        elif str(sm.configuration.get("VOXEL_RESOLVE_METHOD")).upper() == "SHRINK":
            print_notice("\tSHRINK method has not been implemented yet!", mt.MessagePrefix.ERROR)
            raise NotImplementedError
            # TODO: Write the shrinking method of voxel segmentation

            pass
        elif str(sm.configuration.get("VOXEL_RESOLVE_METHOD")).upper() == "STRETCH":
            print_notice("\tSTRETCH method has not been implemented yet!", mt.MessagePrefix.ERROR)
            raise NotImplementedError
            # TODO: Write the stretching method of voxel segmentation

            pass

    dimensions = (voxel_count_x, voxel_count_y, voxel_count_z)
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
                print_notice(pretext + " Voxel [DEPTH " + str(z) + "][ROW " + str(y) + "][COL " + str(x) + "]",
                             mt.MessagePrefix.INFORMATION, end='\r')

                y_start = cubic_dimension * y
                y_end = y_start + cubic_dimension

                voxel = volume[x_start:x_end, y_start:y_end, z_start:z_end]

                if voxel.shape != (cubic_dimension, cubic_dimension, cubic_dimension):
                    resolver = str(sm.configuration.get("VOXEL_RESOLVE_METHOD")).upper()

                    print_notice("FOUND NON-PERFECT VOXEL, RESOLVING WITH [" + resolver + "]...",
                                 mt.MessagePrefix.WARNING)

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
                    print_notice("!-- ERROR: Voxel was invalid size --!", mt.MessagePrefix.ERROR)

    print(pretext + " done!")
    return voxels, dimensions


def save_voxels(voxels, dimensions, location, filename):
    fm.create_if_not_exists(location)

    filepath = location + "/" + filename + ".h5"

    print_notice("Saving voxel collection to '" + filepath + "'... ", mt.MessagePrefix.INFORMATION, end='')
    h5f = h5py.File(filepath, 'w')
    h5f.create_dataset("voxels", data=voxels)
    h5f.create_dataset("dimensions", data=dimensions)
    h5f.close()
    print("done!")


def load_voxels(location, filename):
    filepath = location + filename + ".h5"

    if not fm.file_exists(filepath):
        raise FileNotFoundError("There is no voxel at '" + filepath + "'")

    h5f = h5py.File(filepath, 'r')

    voxels = list()
    dimensions = [int(x) for x in tuple(list(h5f['dimensions']))]

    dataset = h5f['voxels']
    voxels += list(voxel for voxel in dataset)

    h5f.close()

    return voxels, dimensions


def plot_voxel(voxel):
    vox = np.array(voxel)

    if len(vox.shape) == 4:
        vox = np.squeeze(vox)

    x_len, y_len, z_len = vox.shape

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
        ax = fig.add_subplot(111, projection='3d', aspect='auto')

        if sm.USE_BW:
            colours = cm.gray(normalised)
        else:
            colours = cm.jet(normalised)
        ax.voxels(vox, facecolors=colours, edgecolors=colours)

        return fig
    return None

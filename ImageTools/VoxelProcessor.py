import math
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf

from ImageTools import ImageManager as im
from Settings import FileManager as fm, SettingsManager as sm, MessageTools as mt
from Settings.EmailManager import send_email
from Settings.MessageTools import print_notice

# DO NOT DELETE THIS! It shows as unused but it is vital to 3D projection

from Settings import FileManager as fm
from ExperimentTools.MethodologyLogger import Logger


def load_materials(core, use_rois=True, return_binder=True):
    print_notice("Loading voxels for core " + core + "... ", mt.MessagePrefix.INFORMATION, end='')

    voxel_directory = fm.compile_directory(fm.SpecialFolder.ROI_VOXEL_DATA if use_rois else fm.SpecialFolder.CORE_VOXEL_DATA)
    if voxel_directory[-1] != '/':
        voxel_directory += '/'
    voxel_directory += core + '/'

    temp_voxels, dimensions = load_voxels(voxel_directory, "segment_" + sm.get_setting("VOXEL_RESOLUTION"))

    aggregates = np.array([x == 255 for x in temp_voxels], dtype=np.uint8) * 255
    binders = np.array([(x != 255) & (x != 0) for x in temp_voxels], dtype=np.uint8) * 255

    print("done")

    if return_binder:
        return dimensions, aggregates, binders
    else:
        return dimensions, aggregates


def voxels_to_volume(voxels, dimensions):

    pass


def volume_to_voxels(volume_data, cubic_dimension):
    voxels = list()

    volume = np.array(volume_data, dtype=np.uint8)

    # This should be equal to a hypothetical voxelCountZ, since images are square
    voxel_count_x = len(volume) / cubic_dimension
    voxel_count_y = len(volume[0]) / cubic_dimension
    voxel_count_z = len(volume[0][0]) / cubic_dimension

    resolve_method = str(sm.get_setting("VOXEL_RESOLVE_METHOD")).upper()

    if \
            voxel_count_x != int(voxel_count_x) or \
            voxel_count_y != int(voxel_count_y) or \
            voxel_count_z != int(voxel_count_z):
        print_notice("Voxel division resulted in a floating-point number: (%s, %s, %s)..." %
                     (str(voxel_count_x), str(voxel_count_y), str(voxel_count_z)), mt.MessagePrefix.INFORMATION)

        if resolve_method == "LOSSY":
            print_notice("\tUsing LOSSY solution", mt.MessagePrefix.INFORMATION)

            voxel_count_x = math.floor(voxel_count_x)
            voxel_count_y = math.floor(voxel_count_y)
            voxel_count_z = math.floor(voxel_count_z)
        elif resolve_method == "PADDING":
            print_notice("\tUsing PADDING solution", mt.MessagePrefix.INFORMATION)

            voxel_count_x = math.ceil(voxel_count_x)
            voxel_count_y = math.ceil(voxel_count_y)
            voxel_count_z = math.ceil(voxel_count_z)
        elif resolve_method == "EXTRAPOLATE":
            print_notice("\tEXTRAPOLATE method has not been implemented yet!", mt.MessagePrefix.ERROR)
            raise NotImplementedError
            # TODO: Write the extrapolating method of voxel segmentation

            pass
        elif resolve_method == "SHRINK":
            print_notice("\tSHRINK method has not been implemented yet!", mt.MessagePrefix.ERROR)
            raise NotImplementedError
            # TODO: Write the shrinking method of voxel segmentation

            pass
        elif resolve_method == "STRETCH":
            print_notice("\tSTRETCH method has not been implemented yet!", mt.MessagePrefix.ERROR)
            raise NotImplementedError
            # TODO: Write the stretching method of voxel segmentation

            pass

    voxel_count_x = int(voxel_count_x)
    voxel_count_y = int(voxel_count_y)
    voxel_count_z = int(voxel_count_z)

    dimensions = (voxel_count_x, voxel_count_y, voxel_count_z)
    pretext = "Separating %s data volume into %s voxels..." % \
              (str(dimensions), str(voxel_count_x * voxel_count_y * voxel_count_z))

    print_notice(pretext, end='\r', flush=True)

    for x in range(voxel_count_x):
        x_start = cubic_dimension * x
        x_end = x_start + cubic_dimension
        for y in range(voxel_count_y):
            y_start = cubic_dimension * y
            y_end = y_start + cubic_dimension
            for z in range(voxel_count_z):
                print_notice(pretext + " Voxel [DEPTH " + str(z) + "][ROW " + str(y) + "][COL " + str(x) + "]",
                             mt.MessagePrefix.INFORMATION, end='\r')

                z_start = cubic_dimension * z
                z_end = z_start + cubic_dimension

                voxel = volume[x_start:x_end, y_start:y_end, z_start:z_end]

                if voxel.shape != (cubic_dimension, cubic_dimension, cubic_dimension):
                    print_notice("Found non-perfect voxel at (%d:%d, %d:%d, %d:%d), resolving with [%s]..."
                                 % (x_start, x_start + voxel.shape[0],
                                    y_start, y_start + voxel.shape[1],
                                    z_start, z_start + voxel.shape[2],
                                    resolve_method), mt.MessagePrefix.DEBUG)

                    if resolve_method == "PADDING":
                        x_pad = cubic_dimension - len(voxel)
                        if x_pad == cubic_dimension:
                            continue

                        y_pad = cubic_dimension - len(voxel[0])
                        if y_pad == cubic_dimension:
                            continue

                        z_pad = cubic_dimension - len(voxel[0][0])
                        if z_pad == cubic_dimension:
                            continue

                        voxel = np.pad(voxel, ((0, x_pad), (0, y_pad), (0, z_pad)), 'constant', constant_values=0)

                if voxel.shape == (cubic_dimension, cubic_dimension, cubic_dimension):
                    voxels.append(voxel)
                else:
                    print_notice("!-- ERROR: Voxel was invalid size --!", mt.MessagePrefix.ERROR)

    print(pretext + " done!")
    return voxels, dimensions


def save_voxels(voxels, dimensions, location, filename):
    fm.create_if_not_exists(location)

    filepath = location + filename + ".h5"

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


def voxels_to_core(voxels, dimensions):
    voxels = np.squeeze(voxels)
    print_notice("Building (%s) core from voxels... " % (str(dimensions)), end='')
    vox_res = int(sm.get_setting("VOXEL_RESOLUTION"))

    core = np.zeros((dimensions[0] * vox_res,
                     dimensions[1] * vox_res,
                     dimensions[2] * vox_res), dtype=np.uint8)

    ind = 0

    for x in range(dimensions[0]):
        x_min = x * vox_res
        x_max = x_min + vox_res
        for y in range(dimensions[1]):
            y_min = y * vox_res
            y_max = y_min + vox_res
            for z in range(dimensions[2]):
                z_min = z * vox_res
                z_max = z_min + vox_res
                core[x_min:x_max, y_min:y_max, z_min:z_max] = voxels[ind]
                ind += 1

    print("done")
    return core


def process_voxels(images):
    voxels = list()
    dimensions = None

    if sm.get_setting("ENABLE_VOXEL_SEPARATION") == "True":
        voxels, dimensions = volume_to_voxels(images, int(sm.get_setting("VOXEL_RESOLUTION")))

        if sm.get_setting("ENABLE_VOXEL_INPUT_SAVING") == "True":
            im.save_voxel_images(voxels, "Unsegmented")
    return voxels, dimensions


def generate_voxels(use_rois=True, multiprocessing_pool=None):
    input_dir = fm.SpecialFolder.SEGMENTED_ROI_SCANS if use_rois else fm.SpecialFolder.SEGMENTED_CORE_SCANS
    voxel_dir = fm.SpecialFolder.ROI_VOXEL_DATA if use_rois else fm.SpecialFolder.CORE_VOXEL_DATA
    dataset_dir = fm.SpecialFolder.ROI_DATASET_DATA if use_rois else fm.SpecialFolder.CORE_DATASET_DATA

    fm.data_directories = fm.prepare_directories(input_dir)

    for data_directory in fm.data_directories:
        fm.current_directory = data_directory.replace(fm.compile_directory(input_dir), '')

        voxel_directory = fm.compile_directory(voxel_dir) + fm.current_directory[0:-1] + '/'
        dataset_directory = fm.compile_directory(dataset_dir) + fm.current_directory[0:-1] + '/'

        filename = 'segment_' + sm.get_setting("VOXEL_RESOLUTION")

        if not fm.file_exists(voxel_directory + filename + ".h5") or \
                not fm.file_exists(dataset_directory + filename + ".tfrecord"):
            print_notice("Converting segments in '" + data_directory + "' to voxels...", mt.MessagePrefix.INFORMATION)

            print_notice("\tLoading segment data...", mt.MessagePrefix.INFORMATION)
            images = im.load_images_from_directory(data_directory, "segment", multiprocessing_pool)
            voxels, core_dimensions = process_voxels(images)

            if not fm.file_exists(voxel_directory + filename + ".h5"):
                print_notice("\tSaving segment voxels to HDF5...", mt.MessagePrefix.INFORMATION)
                save_voxels(voxels, core_dimensions, voxel_directory, filename)

            if not fm.file_exists(voxel_directory + filename + ".tfrecord"):
                print_notice("\tSaving serialised segment voxels to TFRecord...", mt.MessagePrefix.INFORMATION)
                vox_res = int(sm.get_setting("VOXEL_RESOLUTION"))
                array_dimensions = (np.product(core_dimensions), vox_res, vox_res, vox_res)
                aggregates = np.array([x == 255 for x in voxels], dtype=bool)
                binders = np.array([(x != 255) & (x != 0) for x in voxels], dtype=bool)
                del voxels

                # Shuffle the dataset
                shuffle_order = np.random.permutation(len(aggregates))
                aggregates = aggregates[shuffle_order]
                binders = binders[shuffle_order]

                save_voxel_tfrecord(aggregates, binders, array_dimensions, core_dimensions,
                                    dataset_directory, filename)
        # im.save_voxel_image_collection(voxels, fm.SpecialFolder.VOXEL_DATA, "figures/" + segment)
    send_email("Generating voxels from %ss is finished!" % ("ROI" if use_rois else "Core"))


def save_voxel_tfrecord(aggregates, binders, array_dimensions, core_dimensions, directory, filename):
    fm.create_if_not_exists(directory)

    with open(directory + "record_settings.conf", "w") as f:
        f.write("core_dimensions=%s\r\nvoxel_array_dimensions=%s" % (str(core_dimensions), str(array_dimensions)))

    def voxel_example(feature, label):
        return tf.train.Example(features=tf.train.Features(feature={
            'aggregate': _bytes_feature(feature),
            'binder': _bytes_feature(label)
        }))

    with tf.io.TFRecordWriter(directory + filename + ".tfrecord") as writer:
        for aggregate, binder in zip(aggregates, binders):
            tf_example = voxel_example(aggregate.tostring(), binder.tostring())
            writer.write(tf_example.SerializeToString())


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
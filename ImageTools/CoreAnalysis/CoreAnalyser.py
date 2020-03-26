import numpy as np

from skimage.morphology import skeletonize_3d
from tqdm import tqdm
from Settings import MessageTools as mt, FileManager as fm
from Settings.MessageTools import print_notice
from ImageTools import ImageManager as im


def get_core_by_id(core_id):
    core_directory = fm.compile_directory(fm.SpecialFolder.SEGMENTED_SCANS) + core_id

    if core_directory[-1] != '/':
        core_directory += '/'

    return get_core_image_stack(core_directory)


def get_core_image_stack(directory):
    return im.load_images_from_directory(directory, "segment")


def crop_to_core(core, multiprocessing_pool):
    mask_core = [x != 0 for x in core]
    flattened_core = np.zeros(np.shape(core[0]))

    for i in tqdm(range(len(core)), "Flattening core..."):
        flattened_core = np.logical_or(flattened_core, mask_core[i])

    print_notice("Cropping core to non-void content...", mt.MessagePrefix.INFORMATION)
    coords = np.argwhere(flattened_core)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    cropped_core = np.array([ct_slice[x_min:x_max + 1, y_min:y_max + 1] for ct_slice in core], dtype=np.uint8)

    # TODO: Allow conf file to determine cylindrical/cuboid
    # unique, counts = np.unique(flattened_core, return_counts=True)
    # print_notice("Flattening Results: Void = " + str(counts[0]) +
    #              ", Non-Void = " + str(counts[1]), mt.MessagePrefix.DEBUG)

    # Generate a bounding circle (acts as a flattened core mould)
    # print_notice("Generating bounding cylindrical mould...", mt.MessagePrefix.INFORMATION)
    # enclosed_circle = make_circle(coords)

    # mould_volume = ((math.pi * (enclosed_circle[2] ** 2)) * len(core))
    # print_notice("Mould volume: " + str(mould_volume) + " cubic microns", mt.MessagePrefix.DEBUG)
    # print_notice("Mould diameter: " + str(enclosed_circle[2] * 2) + " microns", mt.MessagePrefix.DEBUG)

    # Label voids outside of mould as "background" (0); Internal voids are set to 1, which allows computational
    # differentiation, but very little difference when generating images
    # print_notice("Labelling out-of-mould voids as background...", mt.MessagePrefix.INFORMATION)
    # for ind, res in enumerate(multiprocessing_pool.starmap(find_background_pixels,
    #                                                       [(x, enclosed_circle) for x in cropped_core])):
    #    np.copyto(cropped_core[ind], res)

    return cropped_core


def calculate_all(core):
    results = list()

    results.append(calculate_composition(core))
    results.append(calculate_average_void_diameter(core))

    void_network = np.squeeze(np.array([core == 0], dtype=np.bool))
    skeleton = get_skeleton(void_network)

    results.append(calculate_euler_number(skeleton))
    results.append(calculate_tortuosity(skeleton))

    return results


def calculate_composition(core):
    print_notice("Calculating Core Compositions...", mt.MessagePrefix.INFORMATION)

    unique, counts = np.unique(core, return_counts=True)

    offset = 0

    if counts.size == 3:
        offset = 1

    total = sum(counts[1-offset:])
    percentages = [x / total for x in counts[1-offset:]]

    print_notice("\tTotal Core Volume: " + str(total) + " voxels", mt.MessagePrefix.INFORMATION)

    print_notice("\tVoid Content: " + str(counts[1-offset]) + " voxels, " +
                 str(percentages[0] * 100.0) + "% of total", mt.MessagePrefix.INFORMATION)

    print_notice("\tBinder Content: " + str(counts[2-offset]) + " voxels, " +
                 str(percentages[1] * 100.0) + "% of total", mt.MessagePrefix.INFORMATION)

    print_notice("\tAggregate Content: " + str(counts[3-offset]) + " voxels, " +
                 str(percentages[2] * 100.0) + "% of total", mt.MessagePrefix.INFORMATION)

    return counts, percentages


def calculate_average_void_diameter(void_network):
    print_notice("Calculating Core Average Void Diameter...", mt.MessagePrefix.INFORMATION)

    # TODO: Calculate average void diameter
    raise NotImplementedError


def get_skeleton(core):
    print_notice("Converting core to skeleton...", mt.MessagePrefix.INFORMATION)
    return skeletonize_3d(core)


def calculate_tortuosity(core, core_is_skeleton=True):
    print_notice("Calculating Core Tortuosity...", mt.MessagePrefix.INFORMATION)

    if not core_is_skeleton:
        core = skeletonize_3d(core)

    # TODO: Calculate tortuosity
    raise NotImplementedError


def calculate_euler_number(core, core_is_skeleton=True):
    print_notice("Calculating Core Euler Number...", mt.MessagePrefix.INFORMATION)

    if not core_is_skeleton:
        core = get_skeleton(core)

    # TODO: Calculate euler number
    raise NotImplementedError

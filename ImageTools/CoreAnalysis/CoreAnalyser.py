import math

import numpy as np

from tqdm import tqdm

from ImageTools.SmallestEnclosingCircle import make_circle
from Settings import MessageTools as mt
from Settings.MessageTools import print_notice
from multiprocessing import Pool
from ImageTools.CoreAnalysis.BackgroundFinder import find_background_pixels

pool = Pool()


def crop_to_core(core):
    mask_core = [x != 0 for x in core]
    flattened_core = np.zeros(np.shape(core[0]))

    for i in tqdm(range(len(core)), "Flattening core to determine cylindrical volume"):
        flattened_core = np.logical_or(flattened_core, mask_core[i])

    unique, counts = np.unique(flattened_core, return_counts=True)
    print_notice("Flattening Results: Void = " + str(counts[0]) +
                 ", Non-Void = " + str(counts[1]), mt.MessagePrefix.DEBUG)

    print_notice("Cropping core to non-void content...", mt.MessagePrefix.INFORMATION)
    coords = np.argwhere(flattened_core)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    cropped_core = np.array([ct_slice[x_min:x_max + 1, y_min:y_max + 1] for ct_slice in core])

    # Generate a bounding circle (acts as a flattened core mould)
    print_notice("Generating bounding cylindrical mould...", mt.MessagePrefix.INFORMATION)
    enclosed_circle = make_circle(coords)

    mould_volume = ((math.pi * (enclosed_circle[2] ** 2)) * len(core))
    print_notice("Mould volume: " + str(mould_volume) + " cubic microns", mt.MessagePrefix.DEBUG)
    print_notice("Mould diameter: " + str(enclosed_circle[2] * 2) + " microns", mt.MessagePrefix.DEBUG)

    # Label voids outside of mould as "background" (0); Internal voids are set to 1, which allows computational
    # differentiation, but very little difference when generating images
    print_notice("Labelling out-of-mould voids as background...", mt.MessagePrefix.INFORMATION)
    for ind, res in enumerate(pool.starmap(find_background_pixels, [(x, enclosed_circle) for x in cropped_core])):
        np.copyto(cropped_core[ind], res)

    pool.close()

    return cropped_core


def calculate_all(core):
    results = list()

    core = crop_to_core(core)
    void_network = (core == 0)

    results.append(calculate_composition(core))
    results.append(calculate_average_void_diameter(core))
    results.append(calculate_euler_number(core))
    results.append(calculate_tortuosity(void_network))

    return results


def calculate_composition(core):
    print_notice("Calculating Core Compositions...", mt.MessagePrefix.INFORMATION)

    unique, counts = np.unique(core, return_counts=True)

    total = sum(counts[1:])

    print_notice("\tTotal Core Volume: " + str(total) + " cubic microns", mt.MessagePrefix.INFORMATION)

    print_notice("\tVoid Content: " + str(counts[1]) + " cubic microns, " +
                 str((counts[1] / total) * 100.0) + "% of total", mt.MessagePrefix.INFORMATION)

    print_notice("\tBinder Content: " + str(counts[2]) + " cubic microns, " +
                 str((counts[2] / total) * 100.0) + "% of total", mt.MessagePrefix.INFORMATION)

    print_notice("\tAggregate Content: " + str(counts[3]) + " cubic microns, " +
                 str((counts[3] / total) * 100.0) + "% of total", mt.MessagePrefix.INFORMATION)

    return counts


def calculate_tortuosity(core):
    print_notice("Calculating Core Tortuosity...", mt.MessagePrefix.INFORMATION)
    raise NotImplementedError


def calculate_euler_number(core):
    print_notice("Calculating Core Euler Number...", mt.MessagePrefix.INFORMATION)
    raise NotImplementedError


def calculate_average_void_diameter(core):
    print_notice("Calculating Core Average Void Diameter...", mt.MessagePrefix.INFORMATION)
    raise NotImplementedError

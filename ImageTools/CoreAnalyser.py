from tqdm import tqdm

from ImageTools.SmallestEnclosingCircle import make_circle, is_in_circle
from Settings import MessageTools as mt
from Settings.MessageTools import print_notice
import numpy as np


def crop_to_core(core):
    mask_core = [x != 0 for x in core]
    flattened_core = np.zeros(np.shape(core[0]))

    for i in tqdm(range(len(core)), "Flattening core to determine cylindrical volume"):
        flattened_core = np.logical_or(flattened_core, mask_core[i])

    unique, counts = np.unique(flattened_core, return_counts=True)
    print("Flattening Results: " + str(unique[0]) + " - " + str(counts[0]) +
          ", " + str(unique[1]) + " - " + str(counts[1]))

    print_notice("Cropping core to non-void content...", mt.MessagePrefix.INFORMATION)
    coords = np.argwhere(flattened_core)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    cropped_core = np.array([ct_slice[x_min:x_max + 1, y_min:y_max + 1] for ct_slice in core])

    # Generate a bounding circle (acts as a flattened core mould)
    print_notice("Generating bounding cylindrical mould...", mt.MessagePrefix.INFORMATION)
    enclosing_circle = make_circle(coords)

    # Label voids outside of mould as "background" (-1)
    print_notice("Labelling out-of-mould voids as background...", mt.MessagePrefix.INFORMATION)
    for i in range(len(cropped_core)):
        voids = np.argwhere(cropped_core[i] == 0)
        for (x, y) in voids:
            if not is_in_circle(enclosing_circle, (x, y)):
                cropped_core[i, x, y] = -1

    return cropped_core


def calculate_all(core):
    results = list()

    core = crop_to_core(core)

    results.append(calculate_composition(core))
    results.append(calculate_tortuosity(core))
    results.append(calculate_euler_number(core))
    results.append(calculate_average_void_diameter(core))

    return results


def calculate_composition(core):
    print_notice("Calculating Core Compositions...", mt.MessagePrefix.INFORMATION)
    raise NotImplementedError


def calculate_tortuosity(core):
    print_notice("Calculating Core Tortuosity...", mt.MessagePrefix.INFORMATION)
    raise NotImplementedError


def calculate_euler_number(core):
    print_notice("Calculating Core Euler Number...", mt.MessagePrefix.INFORMATION)
    raise NotImplementedError


def calculate_average_void_diameter(core):
    print_notice("Calculating Core Average Void Diameter...", mt.MessagePrefix.INFORMATION)
    raise NotImplementedError

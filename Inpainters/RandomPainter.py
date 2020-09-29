import numpy as np

from numpy import random
from Settings import MessageTools as mt
from ImageTools import VoxelProcessor as vp


def get_aggregate_skeleton(core_id, use_rois=False):
    # Load aggregate and binder (unused) voxels from core by its ID
    dimensions, aggregate, _ = vp.load_materials(core_id, use_rois=use_rois)
    # Make the matrix explicitly 4-Dimensional (1 channel -- intensity)
    aggregate = np.expand_dims(aggregate, 4)

    return aggregate


def paint_by_exact_avc(avc_percent, aggregate_core, use_roi=True):
    # Invert the core to make voids white
    aggregate_core = np.invert(aggregate_core)
    # Get available empty locations
    locs = np.flatnonzero(aggregate_core)
    # Randomly shuffle the indices to make painting random
    np.random.shuffle(locs)

    # Get the total volume of the given core
    if use_roi:
        total_volume = np.product(aggregate_core.shape)
    else:
        # TODO: Make alternative version for using full cores
        raise NotImplemented

    # Count non-white voxels as available space to place mastic
    available_volume = len(locs)
    # Get the percentage of unused voxels from the total number of voxels in the core
    available_percent = available_volume / total_volume
    # Get the percentage of this free space that should become mastic
    mastic_percent = (available_percent - avc_percent) / available_percent
    # Convert this percentage back into a volume, in number of voxels
    mastic_volume = available_volume * mastic_percent
    # Convert binary values to 8-bit image intensities
    aggregate_core *= 255

    # Paint the core randomly by placing a mastic voxel at randomly selected available locations
    # for the total amount of mastic_volume
    for loc in locs[:mastic_volume]:
        aggregate_core[np.unravel_index(loc, aggregate_core.shape)] = 127

    # Return back to the original intensity of segments, where aggregates are brightest and voids are darkest
    aggregate_core = np.invert(aggregate_core)

    return aggregate_core


def paint_by_probabilistic_avc(avc_percent, aggregate_core, use_roi=True):
    # Convert percentages to decimals
    if avc_percent > 1:
        mt.print_notice("Air-Void Content was not between 0 and 1. The value has been divided by 100, please check"
                        "that this is an appropriate action.", mt.MessagePrefix.WARNING)
        avc_percent /= 100
    # Invert the core to make voids white
    aggregate_core = np.invert(aggregate_core)
    # Get available empty locations
    locs = np.flatnonzero(aggregate_core)
    # Convert binary values to 8-bit image intensities
    aggregate_core *= 255

    # For each location, place a mastic voxel based on the AVC percentage being used as a probability of placement
    for loc in locs:
        # Generate a random number from 0 to 1
        rnd = random.rand()
        # If the random number is less than the avc, paint mastic there
        if rnd <= avc_percent:
            aggregate_core[np.unravel_index(loc, aggregate_core.shape)] = 127

    return aggregate_core

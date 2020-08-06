import os
import numpy as np
import kimimaro
import pytrax as pt
import porespy.networks as psn

from tqdm import tqdm
from Settings import MessageTools as mt, FileManager as fm, DatabaseManager as dm, SettingsManager as sm
from Settings.MessageTools import print_notice
from ImageTools import ImageManager as im
from mpl_toolkits import mplot3d


def get_core_by_id(core_id, use_rois=True, multiprocessing_pool=None):
    core_directory = fm.compile_directory(fm.SpecialFolder.SEGMENTED_ROI_SCANS if use_rois
                                          else fm.SpecialFolder.SEGMENTED_CORE_SCANS) + core_id

    if core_directory[-1] != '/':
        core_directory += '/'

    return get_core_image_stack(core_directory, multiprocessing_pool)


def get_core_image_stack(directory, multiprocessing_pool=None):
    return im.load_images_from_directory(directory, "segment", multiprocessing_pool)


def crop_to_core(core):
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
    void_network = np.squeeze(np.array([core == 0], dtype=np.bool))
    pore_network = get_pore_network(void_network)

    results = list()
    results.append(calculate_composition(core))
    results.append(calculate_average_void_diameter(pore_network, True))
    results.append(calculate_tortuosity(pore_network, True))
    results.append(calculate_euler_number(pore_network, True))

    return results


def calculate_aggregate_gradation(aggregates):
    print_notice("Calculating Aggregate Gradation...", mt.MessagePrefix.INFORMATION)
    aggregates = list()

    # TODO: Calculate gradations
    return aggregates


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


def calculate_average_void_diameter(void_network, core_is_pore_network):
    print_notice("Calculating Core Average Void Diameter...", mt.MessagePrefix.INFORMATION)

    if not core_is_pore_network:
        void_network = get_pore_network(void_network)

    avd = np.average(void_network["pore.diameter"])

    conversion = float(sm.get_setting("PIXELS_TO_MM"))

    avd_mm = avd * conversion

    print_notice("\tAverage Void Diameter (Volume) = %fmm (%f Pixels)" % (avd_mm, avd))  # TODO: Convert pixels to mm

    return avd_mm


def get_skeleton(core, suppress_messages=False):
    if not suppress_messages:
        print_notice("Converting core to skeleton...", mt.MessagePrefix.INFORMATION)

    if isinstance(core, list):
        core = np.array(core, dtype=np.uint8)

    return kimimaro.skeletonize(core, progress=True, parallel=0, parallel_chunk_size=64)


def get_pore_network(core):
    pore_network = psn.snow(core, marching_cubes_area=False)

    return pore_network


def calculate_tortuosity(core):
    print_notice("Calculating Core Tortuosity...", mt.MessagePrefix.INFORMATION)

    core = np.array(core, dtype=np.uint8)

    rw = pt.RandomWalk(core)
    rw.run(nt=1e5, nw=1e4, same_start=False, stride=100, num_proc=12)

    path = fm.compile_directory(fm.SpecialFolder.FIGURES) + "/Analysis/Tortuosity"

    rw.export_walk(image=rw.im, sample=1, path=path)
    rw.plot_msd()

    return rw.data['Mean_tau']


def calculate_euler_number(core, core_is_pore_network=True):
    print_notice("Calculating Core Euler Number...", mt.MessagePrefix.INFORMATION)

    if not core_is_pore_network:
        print_notice("\tConverting to pore network...", mt.MessagePrefix.INFORMATION)
        core = np.array([x == 0 for x in core], np.bool)
    print_notice("\tConverting to skeleton...", mt.MessagePrefix.INFORMATION)

    skeleton = get_skeleton(core, True)

    # TODO: Calculate Euler number
    euler = core.euler_number

    print_notice("\tEuler number = " + str(euler))

    return euler


def update_database_core_analyses(ignore_blacklist=False):
    print_notice("Updating core analyses in database...", mt.MessagePrefix.INFORMATION)

    ct_directory = fm.compile_directory(fm.SpecialFolder.UNPROCESSED_SCANS)

    ct_ids = [name for name in os.listdir(ct_directory)]

    dm.db_cursor.execute("USE ct_scans;")

    included_calculations = "MeasuredAirVoidContent, MasticContent, AverageVoidDiameter, Tortuosity"

    for ct_id in ct_ids:
        sql = "SELECT " + included_calculations + " FROM asphalt_cores WHERE ID=%s"
        if not ignore_blacklist:
            sql += " AND Blacklist = 0"
        values = (ct_id,)

        dm.db_cursor.execute(sql, values)
        res = dm.db_cursor.fetchone()

        if res is not None and any(x is None for x in res):
            core = get_core_by_id(ct_id)
            counts, percentages = calculate_composition(core)
            void_network = np.array([x == 0 for x in core], np.bool)

            # gradation = ca.calculate_aggregate_gradation(np.array([x == 2 for x in core], np.bool))
            avd = calculate_average_void_diameter(void_network, False)
            # euler_number = ca.calculate_euler_number(void_network)

            tortuosity = calculate_tortuosity(void_network)

            sql = "UPDATE asphalt_cores SET " \
                  "MeasuredAirVoidContent=%s, MasticContent=%s, AverageVoidDiameter=%s, Tortuosity=%s WHERE ID=%s"

            values = (float(percentages[0]), float(percentages[1]), float(avd), float(tortuosity), ct_id)
            dm.db_cursor.execute(sql, values)

    dm.db_cursor.execute("USE ***REMOVED***_Phase1;")
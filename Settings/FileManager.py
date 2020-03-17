from glob import glob
from os import walk, path, makedirs

from Settings import SettingsManager as sm, MessageTools as mt
from Settings.MessageTools import print_notice
from enum import Enum
from anytree import Node, RenderTree, Walker, search


class SpecialFolder(Enum):
    ROOT = 0
    PROCESSED_SCANS = 1
    UNPROCESSED_SCANS = 2
    SEGMENTED_SCANS = 3
    EXPERIMENTS = 4
    RESULTS = 5
    VOXEL_DATA = 6
    MODEL_DATA = 7
    DATASET_DATA = 8
    LOGS = 9
    GENERATED_VOXEL_DATA = 10
    GENERATED_CORE_DATA = 11
    SCAN_DATA = 12
    ROI_SCANS = 13
    FIGURES = 14


data_directories = []
current_directory = str()

directory_tree = Node(SpecialFolder.ROOT)

directory_ids = {
    SpecialFolder.ROOT: "IO_ROOT_DIR",
    SpecialFolder.EXPERIMENTS: "IO_EXPERIMENT_ROOT_DIR",
    SpecialFolder.SCAN_DATA: "IO_SCAN_ROOT_DIR",
    SpecialFolder.RESULTS: "IO_RESULTS_ROOT_DIR",

    SpecialFolder.PROCESSED_SCANS: "IO_PROCESSED_SCAN_ROOT_DIR",
    SpecialFolder.UNPROCESSED_SCANS: "IO_UNPROCESSED_SCAN_ROOT_DIR",
    SpecialFolder.SEGMENTED_SCANS: "IO_SEGMENTED_SCAN_ROOT_DIR",
    SpecialFolder.ROI_SCANS: "IO_ROI_SCAN_ROOT_DIR",

    SpecialFolder.VOXEL_DATA: "IO_VOXEL_DATA_ROOT_DIR",
    SpecialFolder.MODEL_DATA: "IO_MODEL_ROOT_DIR",
    SpecialFolder.DATASET_DATA: "IO_DATASET_ROOT_DIR",
    SpecialFolder.LOGS: "IO_LOG_ROOT_DIR",

    SpecialFolder.GENERATED_VOXEL_DATA: "IO_GENERATED_VOXEL_ROOT_DIR",
    SpecialFolder.GENERATED_CORE_DATA: "IO_GENERATED_CORE_ROOT_DIR",
    SpecialFolder.FIGURES: "IO_FIGURES_ROOT_DIR"
}


def initialise_directory_tree():
    print_notice("Initialising directory tree...", mt.MessagePrefix.DEBUG)

    experiments = Node(SpecialFolder.EXPERIMENTS, parent=directory_tree)
    scans = Node(SpecialFolder.SCAN_DATA, parent=directory_tree)
    results = Node(SpecialFolder.RESULTS, parent=experiments)

    for folder in [SpecialFolder.PROCESSED_SCANS,
                   SpecialFolder.UNPROCESSED_SCANS,
                   SpecialFolder.SEGMENTED_SCANS,
                   SpecialFolder.ROI_SCANS]:
        Node(folder, parent=scans)

    for folder in [SpecialFolder.VOXEL_DATA,
                   SpecialFolder.MODEL_DATA,
                   SpecialFolder.DATASET_DATA,
                   SpecialFolder.LOGS]:
        Node(folder, parent=experiments)

    for folder in [SpecialFolder.GENERATED_CORE_DATA,
                   SpecialFolder.GENERATED_VOXEL_DATA,
                   SpecialFolder.FIGURES]:
        Node(folder, parent=results)

    for pre, fill, node in RenderTree(directory_tree):
        print_notice("%s%s" % (pre, node.name), mt.MessagePrefix.DEBUG)


def compile_directory(child_directory, force_add_scan_type=False, add_scan_type_if_leaf=True):
    if not isinstance(child_directory, SpecialFolder):
        print_notice("Given directory is not of type SpecialFolder!", mt.MessagePrefix.ERROR)

    node = search.find(directory_tree, filter_=lambda n: n.name == child_directory)

    compiled_directory = ""

    for directory in node.ancestors:
        compiled_directory += get_directory(directory.name) + '/'
    compiled_directory += get_directory(child_directory) + '/'

    if force_add_scan_type or (add_scan_type_if_leaf and node.is_leaf):
        compiled_directory += sm.configuration.get("IO_SCAN_TYPE") + '/'

    return compiled_directory


def check_folder_type(special_folder):
    if not isinstance(special_folder, SpecialFolder):
        raise TypeError("special_folder must be of enum type SpecialFolder")


def get_settings_id(special_folder):
    check_folder_type(special_folder)

    if special_folder in directory_ids:
        return directory_ids.get(special_folder)
    else:
        return "NONE"


def assign_special_folders():
    missing_key = False

    for folder in SpecialFolder:
        if sm.configuration.get(get_settings_id(folder)) is None:
            print(str(folder) + " is not in the configuration file!")
            missing_key = True
            continue

        directory = sm.configuration.get(get_settings_id(folder))

        if len(directory) > 0:
            create_if_not_exists(compile_directory(folder, False))
        else:
            print_notice("No directory information was found for " + str(folder), mt.MessagePrefix.ERROR)
            raise ValueError

    if missing_key:
        raise KeyError


def get_directory(special_folder):
    check_folder_type(special_folder)

    return sm.configuration.get(directory_ids.get(special_folder))


def compile_directories(special_folder):
    check_folder_type(special_folder)

    return [f.replace('\\', '/')
            for f in glob(compile_directory(special_folder) + "**/", recursive=True)]


def prepare_directories(special_folder):
    check_folder_type(special_folder)

    data_dirs = compile_directories(special_folder)

    to_remove = set()

    for data_directory in data_dirs:
        for dPaths, dNames, fNames in walk(data_directory):
            if len(fNames) == 0:
                to_remove.add(data_directory)

    for directory in to_remove:
        data_dirs.remove(directory)

    return data_dirs


def create_if_not_exists(directory):
    if not path.exists(directory):
        try:
            print_notice("Creating directory '%s'" % directory, mt.MessagePrefix.DEBUG)
            makedirs(directory)
        except FileExistsError:
            pass


def file_exists(filepath):
    return path.isfile(filepath)

from glob import glob
from os import walk

from Settings import SettingsManager as sm
from enum import Enum

data_directories = []
current_directory = str()
root_directories = [""]


class SpecialFolder(Enum):
    NONE = 0
    PROCESSED_SCANS = 1
    UNPROCESSED_SCANS = 2
    SEGMENTED_SCANS = 3
    VOID_SEGMENTS = 4
    AGGREGATE_SEGMENTS = 5
    BINDER_SEGMENTS = 6
    EXPERIMENTS = 7
    RESULTS = 8
    VOXEL_DATA = 9


def get_settings_id(special_folder):
    if not isinstance(special_folder, SpecialFolder):
        raise TypeError("special_folder must be of enum type SpecialFolder")

    if special_folder == SpecialFolder.PROCESSED_SCANS:
        return "IO_PROCESSED_SCAN_ROOT_DIR"
    elif special_folder == SpecialFolder.UNPROCESSED_SCANS:
        return "IO_UNPROCESSED_SCAN_ROOT_DIR"
    elif special_folder == SpecialFolder.SEGMENTED_SCANS:
        return "IO_SEGMENTED_SCAN_ROOT_DIR"
    elif special_folder == SpecialFolder.VOID_SEGMENTS:
        return "IO_SEGMENTED_VOID_ROOT_DIR"
    elif special_folder == SpecialFolder.AGGREGATE_SEGMENTS:
        return "IO_SEGMENTED_AGGREGATE_ROOT_DIR"
    elif special_folder == SpecialFolder.BINDER_SEGMENTS:
        return "IO_SEGMENTED_BINDER_ROOT_DIR"
    elif special_folder == SpecialFolder.EXPERIMENTS:
        return "IO_EXPERIMENT_ROOT_DIR"
    elif special_folder == SpecialFolder.RESULTS:
        return "IO_RESULTS_ROOT_DIR"
    elif special_folder == SpecialFolder.VOXEL_DATA:
        return "IO_VOXEL_DATA_ROOT_DIR"
    else:
        return "NONE"


def has_directory_been_processed(unprocessed_directory, processed_directory):

    return False


def assign_special_folders():
    for folder in SpecialFolder:
        root_directories.insert(folder.value, sm.configuration.get(get_settings_id(folder)))


def get_directories(special_folder):
    if not isinstance(special_folder, SpecialFolder):
        raise TypeError("special_folder must be of enum type SpecialFolder")
    return [f.replace('\\', '/')
            for f in glob(root_directories[special_folder.value] + "**/", recursive=True)]


def prepare_directories(special_folder):
    if not isinstance(special_folder, SpecialFolder):
        raise TypeError("special_folder must be of enum type SpecialFolder")

    data_dir = get_directories(special_folder)

    to_remove = set()

    for data_directory in data_dir:
        for dPaths, dNames, fNames in walk(data_directory):
            if len(fNames) == 0:
                to_remove.add(data_directory)

    for directory in to_remove:
        data_dir.remove(directory)

    return data_dir

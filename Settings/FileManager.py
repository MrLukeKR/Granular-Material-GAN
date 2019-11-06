from glob import glob
from os import walk

from Settings import SettingsManager as sm
from enum import Enum

data_directories = []
current_directory = str()
root_directories = [""]


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


def get_settings_id(special_folder):
    if not isinstance(special_folder, SpecialFolder):
        raise TypeError("special_folder must be of enum type SpecialFolder")

    if special_folder == SpecialFolder.ROOT:
        return "IO_ROOT_DIR"
    elif special_folder == SpecialFolder.PROCESSED_SCANS:
        return "IO_PROCESSED_SCAN_ROOT_DIR"
    elif special_folder == SpecialFolder.UNPROCESSED_SCANS:
        return "IO_UNPROCESSED_SCAN_ROOT_DIR"
    elif special_folder == SpecialFolder.SEGMENTED_SCANS:
        return "IO_SEGMENTED_SCAN_ROOT_DIR"
    elif special_folder == SpecialFolder.EXPERIMENTS:
        return "IO_EXPERIMENT_ROOT_DIR"
    elif special_folder == SpecialFolder.RESULTS:
        return "IO_RESULTS_ROOT_DIR"
    elif special_folder == SpecialFolder.VOXEL_DATA:
        return "IO_VOXEL_DATA_ROOT_DIR"
    elif special_folder == SpecialFolder.MODEL_DATA:
        return "IO_MODEL_ROOT_DIR"
    elif special_folder == SpecialFolder.DATASET_DATA:
        return "IO_DATASET_ROOT_DIR"
    else:
        return "NONE"


def assign_special_folders():
    root_directories.insert(SpecialFolder.ROOT.value, sm.configuration.get(get_settings_id(SpecialFolder.ROOT)))

    for folder in SpecialFolder:
        if folder != SpecialFolder.ROOT:
            root_directories.insert(
                folder.value,
                root_directories[SpecialFolder.ROOT.value] + sm.configuration.get(get_settings_id(folder))
            )


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

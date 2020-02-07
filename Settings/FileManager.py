from glob import glob
from os import walk, path, makedirs

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
    LOGS = 9
    GENERATED_VOXEL_DATA = 10
    SCAN_DATA = 11


def check_folder_type(special_folder):
    if not isinstance(special_folder, SpecialFolder):
        raise TypeError("special_folder must be of enum type SpecialFolder")


def get_settings_id(special_folder):
    check_folder_type(special_folder)

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
    elif special_folder == SpecialFolder.GENERATED_VOXEL_DATA:
        return "IO_GENERATED_VOXEL_ROOT_DIR"
    elif special_folder == SpecialFolder.VOXEL_DATA:
        return "IO_VOXEL_DATA_ROOT_DIR"
    elif special_folder == SpecialFolder.MODEL_DATA:
        return "IO_MODEL_ROOT_DIR"
    elif special_folder == SpecialFolder.DATASET_DATA:
        return "IO_DATASET_ROOT_DIR"
    elif special_folder == SpecialFolder.LOGS:
        return "IO_LOG_ROOT_DIR"
    elif special_folder == SpecialFolder.SCAN_DATA:
        return "IO_SCAN_DIR"
    else:
        return "NONE"


def assign_special_folders():
    root_directories.insert(SpecialFolder.ROOT.value, sm.configuration.get(get_settings_id(SpecialFolder.ROOT)))

    missing_key=False

    for folder in SpecialFolder:
        if sm.configuration.get(get_settings_id(folder)) is None:
            print(str(folder) + " is not in the configuration file!")
            missing_key = True
            continue

        if folder != SpecialFolder.ROOT:
            root_directories.insert(
                folder.value,
                root_directories[SpecialFolder.ROOT.value] + sm.configuration.get(get_settings_id(folder))
            )

    if missing_key:
        raise KeyError

    for directory in root_directories:
        if len(directory) > 0:
            create_if_not_exists(directory)


def get_directory(special_folder):
    check_folder_type(special_folder)

    return root_directories[special_folder.value]


def get_directories(special_folder):
    check_folder_type(special_folder)

    return [f.replace('\\', '/')
            for f in glob(root_directories[special_folder.value] + "**/", recursive=True)]


def prepare_directories(special_folder):
    check_folder_type(special_folder)

    data_dir = get_directories(special_folder)

    to_remove = set()

    for data_directory in data_dir:
        for dPaths, dNames, fNames in walk(data_directory):
            if len(fNames) == 0:
                to_remove.add(data_directory)

    for directory in to_remove:
        data_dir.remove(directory)

    return data_dir


def create_if_not_exists(directory):
    if not path.exists(directory):
        try:
            makedirs(directory)
        except FileExistsError:
            pass


def file_exists(filepath):
    return path.isfile(filepath)

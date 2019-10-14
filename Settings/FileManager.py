from glob import glob
from os import walk

from Settings import SettingsManager as sm
from enum import Enum

data_directories = []
current_directory = str()
root_directories = [""]


class SpecialFolder(Enum):
    NONE = 0
    PROCESSED_DATA = 1
    UNPROCESSED_DATA = 2
    EXPERIMENT_OUTPUT = 3


def has_directory_been_processed(unprocessed_directory, processed_directory):

    return False


def assign_special_folders():
    root_directories.insert(SpecialFolder.UNPROCESSED_DATA.value, sm.configuration.get("IO_DATA_ROOT_DIR"))
    root_directories.insert(SpecialFolder.PROCESSED_DATA.value, sm.configuration.get("IO_PROCESSED_DATA_ROOT_DIR"))
    root_directories.insert(SpecialFolder.EXPERIMENT_OUTPUT.value, sm.configuration.get("IO_OUTPUT_ROOT_DIR"))


def get_processed_directories():
    return [f.replace('\\', '/')
            for f in glob(sm.configuration.get("IO_PROCESSED_DATA_ROOT_DIR") + "**/", recursive=True)]


def get_unprocessed_directories():
    return [f.replace('\\', '/')
            for f in glob(sm.configuration.get("IO_DATA_ROOT_DIR") + "**/", recursive=True)]


def prepare_directories(use_processed=False):
    if use_processed:
        data_dir = get_processed_directories()
    else:
        data_dir = get_unprocessed_directories()

    to_remove = set()

    for data_directory in data_dir:
        for dPaths, dNames, fNames in walk(data_directory):
            if len(fNames) == 0:
                to_remove.add(data_directory)

    for directory in to_remove:
        data_dir.remove(directory)

    return data_dir

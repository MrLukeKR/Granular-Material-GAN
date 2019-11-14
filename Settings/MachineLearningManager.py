import os

from ExperimentTools.MethodologyLogger import Logger
from keras.models import load_model
from Settings import FileManager as fm


# Models are saved as discriminator.h5 and generator.h5, under an experimental ID
def load_network():
    root_dir = fm.root_directories[fm.SpecialFolder.MODEL_DATA.value]
    Logger.print("Loading Generative Adversarial Network...")

    discriminator = None
    generator = None

    if fm.file_exists(root_dir + "discriminator.h5"):
        Logger.print("\tLoading Discriminator... ", end='')
        discriminator = load_model(root_dir + "discriminator.h5")
        Logger.print("done!")
    else:
        Logger.print("Discriminator network is missing!")

    if fm.file_exists(root_dir + "generator.h5"):
        Logger.print("\tLoading Generator... ", end='')
        generator = load_model(root_dir + "generator.h5")
        Logger.print("done!")
    else:
        Logger.print("Generator network is missing!")

    return discriminator, generator


def save_network(discriminator, generator):
    Logger.print("Saving Generative Adversarial Network Model...")

    root_dir = fm.root_directories[fm.SpecialFolder.MODEL_DATA.value]

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    discriminator.model.save(root_dir + "discriminator.h5")
    generator.model.save(root_dir + "generator.h5")


def prepare_dataset(voxels, training_split):
    if training_split >= 1:
        raise ValueError("training_split must be less than 1")
    elif training_split <= 0:
        raise ValueError("training_split must be greater than 0")

    # TODO: Shuffle voxels
    # TODO: Split to training/testing

    training_set = list()
    testing_set = list()

    return training_set, testing_set


def save_dataset(training_set, testing_set):

    pass
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


def save_network(discriminator, generator, experimentID, filename=None):
    Logger.print("Saving Generative Adversarial Network Model...")

    root_dir = fm.root_directories[fm.SpecialFolder.MODEL_DATA.value]

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    filepath = root_dir + "Experiment-" + str(experimentID) + '_' + filename + '_'

    discriminator.save(filepath + "discriminator.h5")
    generator.save(filepath + "generator.h5")


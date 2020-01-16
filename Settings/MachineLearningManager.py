import math
import os

from tensorflow.python.client import device_lib
from ExperimentTools.MethodologyLogger import Logger
from tensorflow.keras.models import load_model
from Settings import FileManager as fm, MessageTools as mt
import tensorflow as tf

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def isfloatnumeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_clean_input(prompt, min=None, max=None):
    user_input = ""

    while not isfloatnumeric(user_input):
        print(prompt)
        user_input = input("Enter a value > ")

        if isfloatnumeric(user_input) and min is not None or max is not None:
            if max is None:
                max = math.inf

            if min is None:
                min = -math.inf

            if float(user_input) < min or float(user_input) > max:
                user_input = ""

        print("")

    return user_input


def safe_get_gpu(gpu_id):
    if gpu_id < get_available_gpus():
        return tf.device('gpu:' + str(gpu_id))
    else:
        return None


def design_gan_architecture():
    generator_settings = dict()
    discriminator_settings = dict()

    generator_settings["strides"] = get_clean_input("Enter GENERATOR strides", 0)
    generator_settings["kernel_size"] = get_clean_input("Enter GENERATOR kernel size", 0)
    generator_settings["levels"] = get_clean_input("Enter GENERATOR levels", 0)
    generator_settings["filters"] = get_clean_input("Enter GENERATOR filters", 0)
    generator_settings["normalisation_momentum"] = get_clean_input("Enter GENERATOR normalisation momentum [0 - 1]", 0, 1)
    generator_settings["activation_alpha"] = get_clean_input("Enter GENERATOR activation alpha [0 - 1]", 0, 1)

    discriminator_settings["strides"] = get_clean_input("Enter DISCRIMINATOR strides", 0)
    discriminator_settings["kernel_size"] = get_clean_input("Enter DISCRIMINATOR kernel size", 0)
    discriminator_settings["levels"] = get_clean_input("Enter DISCRIMINATOR levels", 0)
    discriminator_settings["filters"] = get_clean_input("Enter DISCRIMINATOR filters", 0)
    discriminator_settings["normalisation_momentum"] = get_clean_input("Enter DISCRIMINATOR normalisation momentum [0 - 1]", 0, 1)
    discriminator_settings["activation_alpha"] = get_clean_input("Enter DISCRIMINATOR activation alpha [0 - 1]", 0, 1)

    if Logger.log_model_to_database(generator_settings, discriminator_settings):
        mt.print_notice("Saved model to database!", mt.MessagePrefix.INFORMATION)
    else:
        mt.print_notice("This model already exists in the database!", mt.MessagePrefix.WARNING)

    return generator_settings, discriminator_settings


# Models are saved as discriminator.h5 and generator.h5, under an experimental ID
def load_network(discriminator_location, generator_location):
    Logger.print("Loading Generative Adversarial Network...")

    discriminator = None
    generator = None

    if fm.file_exists(discriminator_location):
        Logger.print("\tLoading Discriminator... ", end='')
        discriminator = load_model(discriminator_location)
        Logger.print("done!")
    else:
        Logger.print("Discriminator network is missing!")

    if fm.file_exists(generator_location):
        Logger.print("\tLoading Generator... ", end='')
        generator = load_model(generator_location)
        Logger.print("done!")
    else:
        Logger.print("Generator network is missing!")

    return discriminator, generator
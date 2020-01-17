import math
import os

from tensorflow.python.client import device_lib
from ExperimentTools.MethodologyLogger import Logger
from tensorflow.keras.models import load_model
from Settings import FileManager as fm, MessageTools as mt, DatabaseManager as dm, SettingsManager as sm
from Settings.MessageTools import print_notice
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
    class dummy_context_mgr():
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    if gpu_id < len(get_available_gpus()):
        try:
            return tf.device('gpu:' + str(gpu_id))
        except AttributeError:
            return dummy_context_mgr()
    else:
        return dummy_context_mgr()


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


def load_model_from_database(modelID=None):
    cursor = dm.db_cursor

    query = "SELECT ID FROM ***REMOVED***_Phase1.model_instances;"

    cursor.execute(query)
    models = cursor.fetchall()

    if cursor.rowcount == 0:
        print_notice("There are no models in the database", mt.MessagePrefix.WARNING)
        exit(0)
    elif cursor.rowcount == 1:
        choice = 0
    else:
        print("The following models are available:")
        for model in models:
            query = "SELECT * FROM ***REMOVED***_Phase1.model_instances WHERE ID = " + str(model[0]) + ";"

            cursor.execute(query)
            settings = cursor.fetchall()

            if len(settings) == 0:
                continue

            print("Model [" + str(model[0]) + "] @ " + str(model[1]) + " ", end='')
            print(settings)

        choice = ""

        while not choice.isnumeric():
            choice = input("Enter the experiment ID to load > ")

            if choice.isnumeric() and int(choice) not in models:
                print_notice("That experiment ID does not exist", mt.MessagePrefix.WARNING)
                choice = ""

    print_notice("Loading model [" + models[choice] + "]", mt.MessagePrefix.INFORMATION)
    model_location_prefix = sm.configuration.get("IO_ROOT_DIR") + sm.configuration.get("IO_MODEL_ROOT_DIR")

    models = [model for model in os.listdir(model_location_prefix)
              if os.path.isfile(os.path.join(model_location_prefix, model))
              and "Experiment-" + str(choice) in model]

    joint_models = list()

    for model in models:
        filename_parts = model.split("_")
        prefix = model.replace(filename_parts[-1], "")

        generator = prefix + "generator.h5"
        discriminator = prefix + "discriminator.h5"

    if generator in models and discriminator in models and not (generator, discriminator) in joint_models:
        joint_models.append((generator, discriminator))

    if len(joint_models) > 1:
        print_notice("Multiple models are available with this experiment:", mt.MessagePrefix.INFORMATION)
        ids = range(len(joint_models))

        for id in ids:
            prefix = joint_models[id][0].split("_")[-1]
            prefix = joint_models[id][0].replace('_' + prefix, "")
            print("[" + str(id) + "] " + prefix)

        choice = ""
        while not choice.isnumeric():
            choice = input("Which model would you like to load? > ")
            if choice.isnumeric() and int(choice) not in ids:
                print_notice("That model does not exist!", mt.MessagePrefix.WARNING)
                choice = ""

    elif len(joint_models) == 0:
        print_notice("No models were found for this experiment!", mt.MessagePrefix.WARNING)
        exit(0)
    else:
        choice = 0

    selected_model = joint_models[int(choice)]

    print_notice("Loading model '" + selected_model[0].replace('_' + selected_model[0].split("_")[-1], "") + "'...", mt.MessagePrefix.INFORMATION)
    loaded_discriminator, loaded_generator = load_network(model_location_prefix + selected_model[1],
                                                              model_location_prefix + selected_model[0])

    if loaded_discriminator is not None and loaded_generator is not None:
        print_notice("Model successfully loaded", mt.MessagePrefix.SUCCESS)

        loaded_generator.summary()
        loaded_discriminator.summary()
    else:
        print_notice("Error loading model!", mt.MessagePrefix.ERROR)
        raise ValueError


def load_architecture_from_database(architectureID=None):
    cursor = dm.db_cursor

    query = "SELECT * FROM ***REMOVED***_Phase1.model_architectures;"

    cursor.execute(query)
    models = cursor.fetchall()

    choice = 0

    if cursor.rowcount == 0:
        print_notice("There are no architectures in the database", mt.MessagePrefix.WARNING)
        return
    elif cursor.rowcount != 1:
        print("The following architectures are available:")
        for model in models:
            query = "SELECT * FROM ***REMOVED***_Phase1.model_architectures WHERE ID = " + str(model[0]) + ";"

            cursor.execute(query)
            settings = cursor.fetchall()

            if len(settings) == 0:
                continue

            print("Model [" + str(model[0]) + "] @ " + str(model[1:]) + " ", end='')
            print(settings)

        choice = ""

        while not choice.isnumeric():
            choice = input("Enter the architecture ID to load > ")

            if choice.isnumeric() and int(choice) not in models:
                print_notice("That architecture ID does not exist", mt.MessagePrefix.WARNING)
                choice = ""

    model_choice = models[choice]

    print_notice("Loading architecture [" + str(model_choice[0]) + "] - '" + model_choice[1] + "'...", mt.MessagePrefix.INFORMATION)
    print_notice("\tGenerator:", mt.MessagePrefix.INFORMATION)
    print_notice("\t\tStrides\t" + str(model_choice[2]), mt.MessagePrefix.INFORMATION)
    print_notice("\t\tKernel Size\t" + str(model_choice[3]), mt.MessagePrefix.INFORMATION)
    print_notice("\t\tNumber of Levels\t" + str(model_choice[4]), mt.MessagePrefix.INFORMATION)
    print_notice("\t\tFilters\t" + str(model_choice[5]), mt.MessagePrefix.INFORMATION)
    print_notice("\t\tNormalisation Momentum\t" + str(model_choice[6]), mt.MessagePrefix.INFORMATION)
    print_notice("\t\tActivation Alpha\t" + str(model_choice[7]), mt.MessagePrefix.INFORMATION)

    print_notice("\tDiscriminator:", mt.MessagePrefix.INFORMATION)
    print_notice("\t\tStrides\t" + str(model_choice[8]), mt.MessagePrefix.INFORMATION)
    print_notice("\t\tKernel Size\t" + str(model_choice[9]), mt.MessagePrefix.INFORMATION)
    print_notice("\t\tNumber of Levels\t" + str(model_choice[10]), mt.MessagePrefix.INFORMATION)
    print_notice("\t\tFilters\t" + str(model_choice[11]), mt.MessagePrefix.INFORMATION)
    print_notice("\t\tNormalisation Momentum\t" + str(model_choice[12]), mt.MessagePrefix.INFORMATION)
    print_notice("\t\tActivation Alpha\t" + str(model_choice[13]), mt.MessagePrefix.INFORMATION)

    gen_settings = dict()
    disc_settings = dict()

    gen_settings["strides"] = int(model_choice[2])
    gen_settings["kernel_size"] = int(model_choice[3])
    gen_settings["levels"] = int(model_choice[4])
    gen_settings["filters"] = int(model_choice[5])
    gen_settings["normalisation_momentum"] = float(model_choice[6])
    gen_settings["activation_alpha"] = float(model_choice[7])

    disc_settings["strides"] = int(model_choice[8])
    disc_settings["kernel_size"] = int(model_choice[9])
    disc_settings["levels"] = int(model_choice[10])
    disc_settings["filters"] = int(model_choice[11])
    disc_settings["normalisation_momentum"] = float(model_choice[12])
    disc_settings["activation_alpha"] = float(model_choice[13])

    return gen_settings, disc_settings


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
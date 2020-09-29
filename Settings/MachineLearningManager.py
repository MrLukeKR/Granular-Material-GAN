import math

from tensorflow.python.client import device_lib
from ExperimentTools.MethodologyLogger import Logger
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from Inpainters.GAN import DCGAN
from Settings import DatabaseManager as dm, FileManager as fm, SettingsManager as sm, MessageTools as mt
from Settings.MessageTools import print_notice
import tensorflow as tf
import numpy as np


# >>> TENSORFLOW TRAINING SPEEDUP & EFFICIENCY SETTINGS
# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(tf.distribute.experimental.CollectiveCommunication.AUTO)
strategy = tf.distribute.MirroredStrategy()
# tf.config.optimizer.set_jit(True)
# <<<

vox_res = None
data_template = None


def initialise():
    global vox_res, data_template
    vox_res = int(sm.get_setting("VOXEL_RESOLUTION"))
    data_template = np.zeros(shape=(1, vox_res, vox_res, vox_res, sm.image_channels))

    if sm.get_setting("ENABLE_HALF_PRECISION_TRAINING") == "True":
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def isfloatnumeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_clean_input(prompt, min_value=None, max_value=None):
    user_input = ""

    while not isfloatnumeric(user_input):
        print(prompt)
        user_input = input("Enter a value > ")

        if isfloatnumeric(user_input) and min_value is not None or max_value is not None:
            if max_value is None:
                max_value = math.inf

            if min_value is None:
                min_value = -math.inf

            if float(user_input) < min_value or float(user_input) > max_value:
                user_input = ""

        print("")

    return user_input


def safe_get_gpu(gpu_id):
    class dummy_context_mgr:
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

    generator_settings["strides"] = int(get_clean_input("Enter GENERATOR strides", 0))
    generator_settings["kernel_size"] = int(get_clean_input("Enter GENERATOR kernel size", 0))
    generator_settings["levels"] = int(get_clean_input("Enter GENERATOR levels", 0))
    generator_settings["filters"] = int(get_clean_input("Enter GENERATOR filters", 0))
    generator_settings["normalisation_momentum"] = float(get_clean_input("Enter GENERATOR normalisation momentum [0 - 1]", 0, 1))
    generator_settings["activation_alpha"] = float(get_clean_input("Enter GENERATOR activation alpha [0 - 1]", 0, 1))

    discriminator_settings["strides"] = int(get_clean_input("Enter DISCRIMINATOR strides", 0))
    discriminator_settings["kernel_size"] = int(get_clean_input("Enter DISCRIMINATOR kernel size", 0))
    discriminator_settings["levels"] = int(get_clean_input("Enter DISCRIMINATOR levels", 0))
    discriminator_settings["filters"] = int(get_clean_input("Enter DISCRIMINATOR filters", 0))
    discriminator_settings["normalisation_momentum"] = float(get_clean_input("Enter DISCRIMINATOR normalisation momentum [0 - 1]", 0, 1))
    discriminator_settings["activation_alpha"] = float(get_clean_input("Enter DISCRIMINATOR activation alpha [0 - 1]", 0, 1))

    arch_id = Logger.log_model_to_database(generator_settings, discriminator_settings)

    if arch_id is not None:
        mt.print_notice("Saved model to database!", mt.MessagePrefix.INFORMATION)
    else:
        mt.print_notice("This model already exists in the database!", mt.MessagePrefix.WARNING)

    return arch_id, generator_settings, discriminator_settings


def get_model_instance(instance_id):
    query = "SELECT * FROM ***REMOVED***_Phase1.model_instances WHERE ID = " + str(instance_id) + ";"
    db_cursor = dm.get_cursor()

    db_cursor.execute(query)
    return db_cursor.fetchone()


def get_model_instances():
    query = "SELECT ID FROM ***REMOVED***_Phase1.model_instances;"
    db_cursor = dm.get_cursor()

    db_cursor.execute(query)
    return [x[0] for x in db_cursor.fetchall()]


def get_model_architecture(architecture_id):
    query = "SELECT * FROM ***REMOVED***_Phase1.model_architectures WHERE ID = " + str(architecture_id) + ";"
    db_cursor = dm.get_cursor()

    db_cursor.execute(query)
    return db_cursor.fetchone()


def load_model_from_database(model_id=None):
    choice = ""

    if model_id is None:
        models = get_model_instances()

        if len(models) == 0:
            print_notice("There are no models in the database", mt.MessagePrefix.WARNING)
            exit(0)
        elif len(models) == 1:
            choice = 0
        else:
            print("The following models are available:")
            for model in models:
                instance = get_model_instance(model)

                if len(instance) == 0:
                    continue

                settings = get_model_architecture(instance[1])

                if len(settings) == 0:
                    continue

                print("Model [" + str(instance[0]) + "] -> " + str(settings) + " ")
                print('\t' + str(instance))

            choice = ""

            while not choice.isnumeric():
                choice = input("Enter the experiment ID to load > ")

                if choice.isnumeric() and int(choice) not in models:
                    print_notice("That experiment ID does not exist", mt.MessagePrefix.WARNING)
                    choice = ""

        print_notice("Loading model [" + str(choice) + "]", mt.MessagePrefix.INFORMATION)
        instance = get_model_instance(choice)
        generator_loc = instance[2]
        discriminator_loc = instance[3]

        architecture = load_architecture_from_database(instance[1])
        model = load_network((generator_loc, discriminator_loc), architecture)

        return model, architecture


def load_architecture_from_database(architecture_id=None):
    db_cursor = dm.get_cursor()

    query = "SELECT * FROM ***REMOVED***_Phase1.model_architectures ORDER BY ID ASC;"

    db_cursor.execute(query)
    models = db_cursor.fetchall()

    choice = 0

    if db_cursor.rowcount == 0:
        print_notice("There are no architectures in the database", mt.MessagePrefix.WARNING)
        return None, None, None
    elif db_cursor.rowcount != 1:
        print_notice("The following architectures are available:")
        for ind, model in enumerate(models):
            if len(model) == 0:
                continue

            print_notice("[{:^3s}] Model {:^3s} - \"{:s}\" ".format(str(ind), str(model[0]), str(model[1])))
            _, _, _ = architecture_to_settings(model)


        choice = ""

        while not choice.isnumeric():
            choice = input("Enter the architecture ID to load > ")

            if choice.isnumeric() and (0 >= int(choice) > len(models)):
                print_notice("That architecture ID does not exist", mt.MessagePrefix.WARNING)
                choice = ""

    model_choice = models[int(choice)]

    print_notice("Loading [%s]: Architecture %s - '%s'..." % (choice, str(model_choice[0]), model_choice[1]))

    architecture_id, gen_settings, disc_settings = architecture_to_settings(model_choice, True)

    return architecture_id, gen_settings, disc_settings


def architecture_to_settings(model_choice, suppress_output=False):
    gen_settings = dict()
    disc_settings = dict()

    architecture_id = int(model_choice[0])

    setting_ind = 3

    for setting_collection in [gen_settings, disc_settings]:
        if not suppress_output:
            if setting_ind == 3:
                print("\t\tGenerator:")
            else:
                print("\t\tDiscriminator:")

        for setting in [("Strides", "strides"), ("Kernel Size", "kernel_size"), ("Number of Levels", "levels"),
                        ("Filters", "filters")]:
            setting_collection[setting[1]] = int(model_choice[setting_ind])
            if not suppress_output:
                print("\t\t\t{:<25s}\t{:>5s}".format(setting[0], str(setting_collection[setting[1]])))
            setting_ind += 1

        for setting in [("Normalisation Momentum", "normalisation_momentum"), ("Activation Alpha", "activation_alpha")]:
            setting_collection[setting[1]] = float(model_choice[setting_ind])
            if not suppress_output:
                print("\t\t\t{:<25s}\t{:>5s}".format(setting[0], str(setting_collection[setting[1]])))
            setting_ind += 1

    return architecture_id, gen_settings, disc_settings


def create_discriminator(settings, template=None):
    global data_template

    if template is None:
        template = data_template

    with safe_get_gpu(0):
        return DCGAN.DCGANDiscriminator(template, settings["strides"], settings["kernel_size"],
                                        settings["filters"], settings["activation_alpha"],
                                        settings["normalisation_momentum"], settings["levels"])


def create_generator(settings, template=None):
    global data_template

    if template is None:
        template = data_template

    with safe_get_gpu(1):
        return DCGAN.DCGANGenerator(template, settings["strides"], settings["kernel_size"], settings["filters"],
                                    settings["activation_alpha"], settings["normalisation_momentum"],
                                    settings["levels"])


# Models are saved as discriminator.ckpt and generator.ckpt, under an experimental ID
def load_network(locations, architectures):
    if not isinstance(locations, tuple):
        raise TypeError

    generator_location = locations[0]
    discriminator_location = locations[1]

    generator_architecture = architectures[1]
    discriminator_architecture = architectures[2]

    print_notice("Loading Generative Adversarial Network...", mt.MessagePrefix.INFORMATION)

    discriminator = None
    generator = None

    if fm.file_exists(discriminator_location):
        print_notice("\tLoading Discriminator... ", mt.MessagePrefix.INFORMATION, end='')
        discriminator = create_discriminator(discriminator_architecture)
        discriminator.model.load_weights(discriminator_location)
        print_notice("done!", mt.MessagePrefix.INFORMATION)
    else:
        print_notice("Discriminator network is missing!", mt.MessagePrefix.ERROR)

    if fm.file_exists(generator_location):
        print_notice("\tLoading Generator... ", mt.MessagePrefix.INFORMATION, end='')
        generator = create_generator(generator_architecture)
        generator.model.load_weights(generator_location)
        print_notice("done!", mt.MessagePrefix.INFORMATION)
    else:
        print_notice("Generator network is missing!", mt.MessagePrefix.ERROR)

    return generator, discriminator
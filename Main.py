# Utilities >>>
import numpy as np
# <<< Utilities

# Image Processing >>>
from ImageTools import VoxelProcessor as vp, ImageManager as im
from ImageTools.CoreAnalysis import CoreAnalyser as ca
from ImageTools.CoreAnalysis.CoreVisualiser import plot_core
from Settings import SettingsManager as sm
from Settings import FileManager as fm
from Settings import DatabaseManager as dm
# <<< Image Processing

# Experiments >>>
from ExperimentTools import MethodologyLogger, ExperimentRunner
from ExperimentTools.MethodologyLogger import Logger

from Settings import MachineLearningManager as mlm
# <<< Experiments
from Settings import MessageTools as mt
from Settings.MessageTools import print_notice

model_loaded = None
architecture_loaded = None


def print_introduction():
    print("   Optimal Material Generator using Generative Adversarial Networks   ")
    print("                    Developed by ***REMOVED*** (BSc)                    ")
    print("In fulfilment of Doctor of Engineering at the University of Nottingham")
    print("----------------------------------------------------------------------")
    print()


def process_voxels(images):
    voxels = list()
    dimensions = None

    if sm.configuration.get("ENABLE_VOXEL_SEPARATION") == "True":
        voxels, dimensions = vp.volume_to_voxels(images, int(sm.configuration.get("VOXEL_RESOLUTION")))

        if sm.configuration.get("ENABLE_VOXEL_INPUT_SAVING") == "True":
            im.save_voxel_images(voxels, "Unsegmented")
    return voxels, dimensions


def setup():
    sm.load_settings()
    fm.assign_special_folders()

    dm.connect_to_database()
    dm.initialise_database()

    mlm.initialise()

    print_introduction()


def get_core_image_stack(directory):
    binder_stack = im.load_images_from_directory(directory, "binder")
    aggregate_stack = im.load_images_from_directory(directory, "aggregate")

    core = [x // 2 + y for (x, y) in zip(binder_stack, aggregate_stack)]

    return core


def generate_voxels():
    fm.data_directories = fm.prepare_directories(fm.SpecialFolder.SEGMENTED_SCANS)

    for data_directory in fm.data_directories:
        printed_status = False
        fm.current_directory = data_directory.replace(fm.get_directory(fm.SpecialFolder.SEGMENTED_SCANS), '')

        voxel_directory = fm.get_directory(fm.SpecialFolder.VOXEL_DATA) + fm.current_directory[0:-1]

        for segment in ("aggregate", "binder"):
            filename = segment + '_' + sm.configuration.get("VOXEL_RESOLUTION")

            if fm.file_exists(voxel_directory + '/' + filename + ".h5"):
                continue

            if not printed_status:
                Logger.print("Converting segments in '" + data_directory + "' to voxels...")
                printed_status = True

            Logger.print("\tLoading " + segment + " data...\r\n\t\t", end='')
            images = im.load_images_from_directory(data_directory, segment)
            voxels, dimensions = process_voxels(images)

            Logger.print("\t\tSaving " + segment + " voxels...\r\n\t\t", end='')
            vp.save_voxels(voxels, dimensions, voxel_directory, filename)
            # im.save_voxel_image_collection(voxels, fm.SpecialFolder.VOXEL_DATA, "figures/" + segment)


def experiment_menu():
    if architecture_loaded is None and model_loaded is None:
        print_notice("Please load an architecture or model first!", mt.MessagePrefix.WARNING)
        return

    print("- Experiment Menu -")
    print("")
    print("[1] K-Cross Fold Validation (Batch Training)")
    print("[2] K-Cross Fold Validation (Entire Dataset Generator)")
    print("[3] Single Model (Batch Training)")
    print("[4] Single Model (Entire Dataset Generator)")

    user_input = input("Enter your menu choice > ")
    
    if user_input.isnumeric() and 4 >= int(user_input) > 0:
        MethodologyLogger.Logger(fm.get_directory(fm.SpecialFolder.LOGS))
    if user_input == "1":
        ExperimentRunner.run_k_fold_cross_validation_experiment(fm.data_directories, 10, architecture_loaded)
    elif user_input == "2":
        raise NotImplementedError
    elif user_input == "3":
        raise NotImplementedError
    elif user_input == "4":
        raise NotImplementedError


def core_analysis_menu():
    print("- Core Analysis Menu -")
    print("[1] Perform all calculations")
    print("[2] Calculate Core Composition (AVC, Mastic Content)")
    print("[3] Calculate Tortuosity")
    print("[4] Calculate Euler Number")
    print("[5] Calculate Average Void Diameter")
    print("")
    print("[ENTER] Return to Main Menu")

    user_input = input("Enter a menu option > ")

    core_id = core_selection_menu()
    core_directory = fm.get_directory(fm.SpecialFolder.SEGMENTED_SCANS) + core_id

    if core_directory[-1] != '/':
        core_directory += '/'

    core = get_core_image_stack(core_directory)

    if user_input == "1":
        ca.calculate_all(core)
    elif user_input == "2":
        ca.calculate_composition(core)
    elif user_input == "3":
        skeleton = ca.get_skeleton(core)
        plot_core(skeleton)
        ca.calculate_tortuosity(skeleton)
    elif user_input == "4":
        skeleton = ca.get_skeleton(core)
        ca.calculate_euler_number(skeleton)
    elif user_input == "5":
        ca.calculate_average_void_diameter(core)


def core_selection_menu():
    print_notice("The following cores are available in the database:", mt.MessagePrefix.INFORMATION)

    cores = dm.get_cores_from_database()

    for core in cores:
        print_notice("[%s]\t Air Void Content: %s\tBitumen Content: %s\t Notes: %s"
                     % (core[0], core[2], core[3], core[4]), mt.MessagePrefix.INFORMATION)

    choice = ""

    valid_ids = [x[0] for x in cores]

    while choice not in valid_ids:
        choice = input("Enter the core ID to run the model on > ")
        if choice not in valid_ids:
            print_notice("'" + choice + "' is not in the database!", mt.MessagePrefix.WARNING)

    return choice


def run_model_menu():
    choice = core_selection_menu()

    dimensions, aggregates, binders = vp.load_materials(choice)

    aggregates = np.expand_dims(aggregates, 4)
    binders = np.expand_dims(binders, 4)

    core_model = np.empty(dimensions)
    generator = model_loaded[0].model

    print_notice("Running aggregate voxels through binder generator...", mt.MessagePrefix.INFORMATION)
    generated_binders = generator.predict(aggregates)

    if len(generated_binders) > 0:
        print_notice("Successfully generated binder voxels!", mt.MessagePrefix.SUCCESS)
    else:
        print_notice("No voxels were generated!", mt.MessagePrefix.WARNING)

    aggregates = [np.array(x).astype(np.float32) for x in aggregates]
    generated_binders = [np.array(x).astype(np.float32) for x in generated_binders]
    binders = [np.array(x).astype(np.float32) for x in binders]

    for x in range(len(generated_binders)):
        generated_binders[x] -= (generated_binders[x] * aggregates[x])
        generated_binders[x] = (generated_binders[x] >= 0.9) * 0.1
        generated_binders[x] += aggregates[x]

    for x in range(len(binders)):
        binders[x] += (aggregates[x] * 2)

    # TODO: Add this voxel set to the database, in order to save the file with a meaningful ID
    im.save_voxel_image_collection(generated_binders, fm.SpecialFolder.GENERATED_VOXEL_DATA, "TestImage",
                                   binders, "Generated", "Actual")
    vp.save_voxels(generated_binders, dimensions, fm.get_directory(fm.SpecialFolder.GENERATED_VOXEL_DATA), "Test")


def main_menu():
    global model_loaded, architecture_loaded

    if architecture_loaded:
        print_notice("Architecture Loaded: " + str(architecture_loaded), mt.MessagePrefix.INFORMATION)
        if model_loaded is None:
            print_notice("No Model Loaded", mt.MessagePrefix.INFORMATION)
        else:
            print_notice("Model Loaded: " + str(model_loaded), mt.MessagePrefix.INFORMATION)
    else:
        print_notice("No Architecture Loaded", mt.MessagePrefix.INFORMATION)

    print("")
    print("!-- ADMIN TOOLS --!")
    print("[CLEARDB] Reinitialise database")
    print("")
    print("- Main Menu -")
    print("[1] Create New Architecture")
    print("[2] Load Existing Architecture")
    print("[3] Load Existing Model Instance")
    print("[4] Train Model")
    print("[5] Run Model")
    print("[6] Core Analysis Tools")

    print("")
    print("[EXIT] End program")

    user_input = input("Enter a menu option > ")

    if user_input.upper() == "CLEARDB":
        dm.reinitialise_database()
    elif user_input == "1":
        arch_id, gen, disc = mlm.design_gan_architecture()
        architecture_loaded = (arch_id, gen, disc)

        print("Would you like to train a model with this architecture? (Can be done later from the main menu)")
        user_input = input("CHOICE [Y/N] > ")

        if user_input[0].upper() == "Y":
            experiment_menu()
    elif user_input == "2":
        arch_id, gen, disc = mlm.load_architecture_from_database()
        if arch_id is not None and gen is not None and disc is not None:
            architecture_loaded = (arch_id, gen, disc)
    elif user_input == "3":
        model_loaded, architecture_loaded = mlm.load_model_from_database()
    elif user_input == "4":
        experiment_menu()
    elif user_input == "5":
        if model_loaded is not None:
            run_model_menu()
        else:
            print_notice("Please load a model first!", mt.MessagePrefix.WARNING)
    elif user_input == "6":
        core_analysis_menu()

    print("")
    return user_input


def main():
    setup()

    print("Please wait while data collections are processed...")

# | DATA PREPARATION MODULE
    if sm.configuration.get("ENABLE_PREPROCESSING") == "True":
        im.preprocess_images()

# \-- | DATA LOADING SUB-MODULE
    if sm.configuration.get("ENABLE_SEGMENTATION") == "True":
        im.segment_images()

    generate_voxels()
# \-- | SEGMENT-TO-VOXEL CONVERSION SUB-MODULE

    while main_menu() != "EXIT":
        continue

        # | GENERATIVE ADVERSARIAL NETWORK MODULE


if __name__ == "__main__":
    main()

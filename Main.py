# Utilities >>>
import os

import numpy as np
from multiprocessing.spawn import freeze_support
from multiprocessing.pool import Pool
# <<< Utilities

# Image Processing >>>
# import pymesh

from ImageTools import VoxelProcessor as vp, ImageManager as im
from ImageTools.CoreAnalysis import CoreAnalyser as ca, CoreVisualiser as cv
# <<< Image Processing

# Data Visualisation >>>
from ImageTools import DataVisualiser as dv
# <<< Data Visualisation

# Experiments >>>
from ExperimentTools import MethodologyLogger, ExperimentRunner
# <<< Experiments

# Settings >>>
from Settings import \
    MachineLearningManager as mlm, \
    MessageTools as mt, \
    SettingsManager as sm, \
    FileManager as fm, \
    DatabaseManager as dm

from Settings.MessageTools import print_notice
# <<< Settings

model_loaded = None
architecture_loaded = None
multiprocessing_pool = None


def print_introduction():
    print("   Optimal Material Generator using Generative Adversarial Networks   ")
    print("                    Developed by ***REMOVED*** (BSc)                    ")
    print("In fulfilment of Doctor of Engineering at the University of Nottingham")
    print("----------------------------------------------------------------------")
    print()


def update_database_core_analyses():
    print_notice("Updating core analyses in database...", mt.MessagePrefix.INFORMATION)

    unprocessed_ct_directory = fm.compile_directory(fm.SpecialFolder.UNPROCESSED_SCANS)

    ct_ids = [name for name in os.listdir(unprocessed_ct_directory)]

    dm.db_cursor.execute("USE ct_scans;")

    included_calculations = "AirVoidContent, MasticContent, AverageVoidDiameter"

    for ct_id in ct_ids:
        sql = "SELECT " + included_calculations + " FROM asphalt_cores WHERE ID=%s"
        values = (ct_id,)

        dm.db_cursor.execute(sql, values)
        res = dm.db_cursor.fetchone()

        if any(x is None for x in res):
            core = ca.get_core_by_id(ct_id)
            counts, percentages = ca.calculate_composition(core)
            void_network = np.array([x == 0 for x in core], np.bool)

            # gradation = ca.calculate_aggregate_gradation(np.array([x == 2 for x in core], np.bool))
            avd = ca.calculate_average_void_diameter(void_network)
            # euler_number = ca.calculate_euler_number(void_network)

            sql = "UPDATE asphalt_cores SET AirVoidContent=%s, MasticContent=%s, AverageVoidDiameter=%s WHERE ID=%s"

            values = (float(percentages[0]), float(percentages[1]), float(avd), ct_id)
            dm.db_cursor.execute(sql, values)

    dm.db_cursor.execute("USE ***REMOVED***_Phase1;")


def process_voxels(images):
    voxels = list()
    dimensions = None

    if sm.configuration.get("ENABLE_VOXEL_SEPARATION") == "True":
        voxels, dimensions = vp.volume_to_voxels(images, int(sm.configuration.get("VOXEL_RESOLUTION")))

        if sm.configuration.get("ENABLE_VOXEL_INPUT_SAVING") == "True":
            im.save_voxel_images(voxels, "Unsegmented")
    return voxels, dimensions


def setup():
    print_introduction()

    sm.load_settings()
    fm.initialise_directory_tree()
    fm.assign_special_folders()

    dm.connect_to_database()
    dm.initialise_database()

    mlm.initialise()


def generate_voxels():
    fm.data_directories = fm.prepare_directories(fm.SpecialFolder.SEGMENTED_SCANS)

    for data_directory in fm.data_directories:
        fm.current_directory = data_directory.replace(fm.compile_directory(fm.SpecialFolder.SEGMENTED_SCANS), '')

        voxel_directory = fm.compile_directory(fm.SpecialFolder.VOXEL_DATA) + fm.current_directory[0:-1] + '/'

        filename = 'segment_' + sm.configuration.get("VOXEL_RESOLUTION")

        if fm.file_exists(voxel_directory + filename + ".h5"):
            continue

        print_notice("Converting segments in '" + data_directory + "' to voxels...", mt.MessagePrefix.INFORMATION)

        print_notice("\tLoading segment data...", mt.MessagePrefix.INFORMATION)
        images = im.load_images_from_directory(data_directory, "segment")
        voxels, dimensions = process_voxels(images)

        print_notice("\tSaving segment voxels...", mt.MessagePrefix.INFORMATION)
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
        MethodologyLogger.Logger(fm.compile_directory(fm.SpecialFolder.LOGS))
    if user_input == "1":
        core_ids, split = data_selection_menu()

        directories = [x[2] for x in dm.get_cores_from_database() if x[0] in core_ids]

        fold_count = int(input("How many folds? > "))

        # Do train phase
        ExperimentRunner.run_k_fold_cross_validation_experiment(directories[:int(split[0])], fold_count, architecture_loaded)

        # TODO: Do test phase
        # ExperimentRunner.test_network(directories[int(split[0] + 1):], )
    elif user_input == "2":
        # TODO: K-Cross fold validation with iterative generator
        raise NotImplementedError
    elif user_input == "3":
        # TODO: Single model training (batch)
        raise NotImplementedError
    elif user_input == "4":
        # TODO: Single model training (iterative generator)
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
    core = ca.get_core_by_id(core_id)
    core = ca.crop_to_core(core, multiprocessing_pool)

    if user_input == "1":
        ca.calculate_all(core)
    elif user_input == "2":
        ca.calculate_composition(core)
    elif user_input == "3":
        skeleton = ca.get_skeleton(core)
        cv.plot_core(skeleton)
        ca.calculate_tortuosity(skeleton)
    elif user_input == "4":
        #skeleton = ca.get_skeleton(core)
        ca.calculate_euler_number(core, False)
    elif user_input == "5":
        ca.calculate_average_void_diameter(core)


def core_selection_menu():
    print_notice("The following cores are available in the database:", mt.MessagePrefix.INFORMATION)

    cores = dm.get_cores_from_database()

    for core in cores:
        print_notice("[%s]"
                     "\tAir Void Content: %s"
                     "\tMastic Content: %s"
                     "\tTortuosity: %s"
                     "\tEuler Number: %s"
                     "\tAverage Void Diameter: %s"
                     "\tNotes: %s"
                     % (core[0], core[3], core[4], core[5], core[6], core[7], core[8]), mt.MessagePrefix.INFORMATION)

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


def data_selection_menu():
    valid = False

    cores = dm.get_cores_from_database()
    core_ids = list()

    while not valid:
        print("[1] All cores")
        print("[2] Select cores by ID")
        print("[3] Select cores by air void percentage")
        print("[4] Select cores by mastic content percentage")

        user_input = input("Input your choice > ")
        valid = True

        if user_input == "1":
            core_ids = [core[0] for core in cores]
        elif user_input == "2":
            raise NotImplementedError
        elif user_input == "3":
            air_voids = set([core[2] for core in cores])
            print(air_voids)
        elif user_input == "4":
            mastic_content = set([core[3] for core in cores])
            print(mastic_content)
        else:
            valid = False
            print_notice("Not a valid input choice!", mt.MessagePrefix.ERROR)

    if len(core_ids) == 0:
        print_notice("No cores available for the given selection criteria", mt.MessagePrefix.ERROR)
        return None, (0, 0)

    available_for_training = len(core_ids) - 1
    user_input = 0

    if available_for_training > 1:
        user_input = input("How many cores from the available set should be used as the training set? (1-%s) > "
                           % str(available_for_training))
    elif available_for_training <= 0:
        print_notice("There are not enough cores in this set to perform validation!", mt.MessagePrefix.ERROR)
        return None, (0, 0)

    split = (user_input, str(len(core_ids) - int(user_input)))

    print_notice("Using %s cores for training, with %s cores for validation" % split, mt.MessagePrefix.INFORMATION)

    return core_ids, split


def core_category_menu():
    valid = False

    while not valid:
        print("Which core category would you like to export from?")
        print("[1] Physical Cores")
        print("[2] Generated Cores")

        user_input = input("Enter your menu choice > ")
        valid = True

        if user_input in {"1", "2"}:
            return int(user_input)
        else:
            valid = False
            print_notice("Not a valid menu choice!", mt.MessagePrefix.ERROR)
            return -1


def core_visualisation_menu():
    valid = False

    while not valid:
        print("[1] Export core to 3D Object File (STL)")
        print("[2] Export core to slice stack animation (Unprocessed/Processed/Segmented/ROI)")
        print("[3] Export image processing plots")
        print("[4] Export segmentation plots")
        print("[5] Export voxel plots")

        user_input = input("Input your choice > ")
        valid = True
        core = None

        if user_input in ["1", "2", "4"]:
            response = core_category_menu()

            if response == 1:
                core_id = core_selection_menu()
                core = ca.get_core_by_id(core_id)
            elif response == 2:
                # TODO: Load generated core by ID
                core = None  # TODO: Replace None with 3D matrix of core components
                raise NotImplementedError
            else:
                print_notice("Not a valid menu choice!", mt.MessagePrefix.ERROR)
                raise ValueError

        if user_input == "1":
            core_mesh = cv.voxels_to_mesh(core)
            core_mesh = cv.simplify_mesh(core_mesh)
            model_dir = fm.compile_directory(fm.SpecialFolder.REAL_ASPHALT_3D_MODELS) + str(core_id) + '.stl'
            core_mesh.export(model_dir)
            # pymesh.save_mesh(model_dir, core_mesh)  # TODO: Make 3D model directories

        elif user_input == "2":
            # TODO: Export core to slice fly-through animation
            raise NotImplementedError
        elif user_input == "3":
            # TODO: Export image processing images
            raise NotImplementedError
        elif user_input == "4":
            # TODO: Export segmentation images
            raise NotImplementedError
        elif user_input == "5":
            # TODO: Export voxel images
            raise NotImplementedError
        else:
            valid = False
            print_notice("Not a valid option!", mt.MessagePrefix.ERROR)


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
    print("[7] Core Visualisation Tools")

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
    elif user_input == "7":
        core_visualisation_menu()

    print("")
    return user_input


def main():
    global multiprocessing_pool

    multiprocessing_pool = Pool()
    setup()

    print_notice("Please wait while data collections are processed...", mt.MessagePrefix.INFORMATION)

# | DATA PREPARATION MODULE
    if sm.configuration.get("ENABLE_PREPROCESSING") == "True":
        im.preprocess_images(multiprocessing_pool)

    im.extract_rois(multiprocessing_pool)

# \-- | DATA LOADING SUB-MODULE
    if sm.configuration.get("ENABLE_SEGMENTATION") == "True":
        im.segment_images(multiprocessing_pool)

    generate_voxels()
# \-- | SEGMENT-TO-VOXEL CONVERSION SUB-MODULE

    update_database_core_analyses()

    while main_menu() != "EXIT":
        continue

        # | GENERATIVE ADVERSARIAL NETWORK MODULE


if __name__ == "__main__":
    freeze_support()
    main()

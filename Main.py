# Utilities >>>
from multiprocessing.pool import Pool
from multiprocessing.spawn import freeze_support
import random

import numpy as np

# Experiments >>>
from ExperimentTools import ExperimentRunner
from ExperimentTools.DataVisualiser import save_training_graphs
from ImageTools import ImageManager as im, VoxelProcessor as vp
from ImageTools.CoreAnalysis import CoreAnalyser as ca, CoreVisualiser as cv
# Settings >>>
from ImageTools.CoreAnalysis.CoreAnalyser import update_database_core_analyses, get_core_by_id, calculate_composition, \
    crop_to_core
from ImageTools.CoreAnalysis.CoreVisualiser import model_all_cores
from ImageTools.ImageManager import load_images_from_directory, apply_preprocessing_pipeline, save_images, \
    segment_images
from ImageTools.VoxelProcessor import generate_voxels
from Settings import DatabaseManager as dm, FileManager as fm, MachineLearningManager as mlm, SettingsManager as sm, \
    MessageTools as mt, EmailManager as em
from Settings.EmailManager import send_email
from Settings.MessageTools import print_notice

# <<< Utilities

model_loaded = None
architecture_loaded = None
multiprocessing_pool = None


def print_introduction():
    print("   Optimal Material Generator using Generative Adversarial Networks   ")
    print("                    Developed by ***REMOVED*** (BSc)                    ")
    print("In fulfilment of Doctor of Engineering at the University of Nottingham")
    print("----------------------------------------------------------------------")
    print()


def setup():
    print_introduction()

    sm.load_auth()
    dm.connect_to_database()
    dm.initialise_database()

    em.initialise()

    fm.initialise_directory_tree()
    fm.assign_special_folders()

    dm.populate_ct_scan_database()

    mlm.initialise()


def experiment_menu():
    global model_loaded

    if architecture_loaded is None and model_loaded is None:
        print_notice("Please load an architecture or model first!", mt.MessagePrefix.WARNING)
        return

    print("- Experiment Menu -")
    print("")
    print("[1] K-Cross Fold Validation")
    print("[2] Train/Test Split")

    user_input = input("Enter a menu option > ")

    core_ids = data_selection_menu()
    random.shuffle(core_ids)

    use_rois = sm.get_setting("USE_REGIONS_OF_INTEREST") == "True"

    if user_input == "1":
        fold_count = int(input("How many folds? > "))

        # Do train phase
        ExperimentRunner.run_k_fold_cross_validation_experiment(core_ids, fold_count,
                                                                architecture_loaded, multiprocessing_pool,
                                                                train_with_rois=use_rois, animate_with_rois=False)
    elif user_input == "2":
        split = ""

        while not (str.isdigit(split) and 0 < int(split) <= (len(core_ids) - 1)):
            split = input("How many cores should be used for training? [1-%s] > " % str(len(core_ids) - 1))

        split = int(split)
        ExperimentRunner.run_train_test_split_experiment(core_ids, split, architecture_loaded, multiprocessing_pool,
                                                         train_with_rois=use_rois, animate_with_rois=False)


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
    core = ca.crop_to_core(core)
    pores = np.array([x == 0 for x in core], dtype=np.bool)

    if user_input == "1":
        ca.calculate_all(core)
    elif user_input == "2":
        ca.calculate_composition(core)
    elif user_input == "3":
        # pores = ca.get_pore_network(core)
        tortuosity = ca.calculate_tortuosity(pores)
        print_notice("Tortuosity: " + str(tortuosity), mt.MessagePrefix.SUCCESS)
    elif user_input == "4":
        # skeleton = ca.get_skeleton(core)
        ca.calculate_euler_number(core, False)
    elif user_input == "5":
        ca.calculate_average_void_diameter(pores)


def core_selection_menu():
    print_notice("The following cores are available in the database:", mt.MessagePrefix.INFORMATION)

    cores = dm.get_cores_from_database()

    for core in cores:
        print_notice("[%s]"
                     "\tTarget Air Void Content: %s"
                     "\tMeasured Air Void Content: %s"
                     "\tMastic Content: %s"
                     "\tTortuosity: %s"
                     "\tEuler Number: %s"
                     "\tAverage Void Diameter: %s"
                     "\tNotes: %s"
                     % (core[0], str(float(core[3]) * 100) + "%", str(float(core[4]) * 100) + "%",
                        str(float(core[5]) * 100) + "%", core[6], core[7], core[8], core[9]), mt.MessagePrefix.INFORMATION)

    choice = ""

    valid_ids = [x[0] for x in cores]

    while choice not in valid_ids:
        choice = input("Enter the core ID to run the model on > ")
        if choice not in valid_ids:
            print_notice("'" + choice + "' is not in the database!", mt.MessagePrefix.WARNING)

    return choice


def run_model_menu():
    global model_loaded

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

        user_input = input("Enter a menu option > ")
        valid = True

        if user_input == "1":
            core_ids = [core[0] for core in cores]
        elif user_input == "2":
            raise NotImplementedError
        elif user_input == "3":
            air_voids = list(set([core[3] for core in cores]))
            inp = ""
            while not (str.isdigit(inp) and 0 <= int(inp) < len(air_voids)):
                for ind, avc in enumerate(air_voids):
                    available = sum([core[3] == avc for core in cores])
                    print("[%s] %s%% AVC (%s available)" % (str(ind), str(avc), str(available)))
                inp = input("Enter an AVC selection > ")

                if not (str.isdigit(inp) and 0 <= int(inp) < len(air_voids)):
                    print_notice("Invalid selection", mt.MessagePrefix.ERROR)

            inp = int(inp)
            core_ids = [core[0] for core in cores if core[3] == air_voids[inp]]
            print_notice("%s x %s%% Air Void Content cores selected:" % (str(len(core_ids)), str(air_voids[inp])))
            for core in core_ids:
                print("\t\t\t\t%s" % core)

        elif user_input == "4":
            mastic_content = set([core[5] for core in cores])
            print(mastic_content)
        else:
            valid = False
            print_notice("Not a valid input choice!", mt.MessagePrefix.ERROR)

    if len(core_ids) == 0:
        print_notice("No cores available for the given selection criteria", mt.MessagePrefix.ERROR)
        return None, (0, 0)

    available_for_training = len(core_ids) - 1

    if available_for_training <= 0:
        print_notice("There are not enough cores in this set to perform validation!", mt.MessagePrefix.ERROR)
        return None

    return core_ids


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
        core_id = None

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

            model_dir = fm.compile_directory(fm.SpecialFolder.REAL_ASPHALT_3D_MODELS) + str(core_id) + '.stl'
            core_mesh.export(model_dir)  # TODO: Make 3D model directories

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


def data_visualisation_menu():
    valid = False

    while not valid:
        print("[1] Plot Experiment Training Data")

        user_input = input("Input your choice > ")
        valid = True

        if user_input == "1":
            info = dm.get_experiment_information()
            for experiment in info:
                print_notice("Experiment ID: %s\tTimestamp: %s\tFolds: %s\tEpochs: %s\tBatch Size: %s"
                             "\tTraining Records: %s" % experiment)

            experiment_ids = [str(x[0]) for x in info]
            experiment_id = ''

            while experiment_id not in experiment_ids:
                experiment_id = input("Enter experiment ID > ")

            train_data = dm.get_training_data(experiment_id)
            disc_loss = [x[6] for x in train_data]
            disc_accuracy = [x[7] for x in train_data]
            gen_loss = [x[8] for x in train_data]
            gen_mse = [x[9] for x in train_data]

            epochs = max([x[4] for x in train_data])

            # TODO: Get unique fold IDs and save a graph per fold
            fold_id = train_data[0][3]

            if fold_id == 0:
                fold_id = None

            save_training_graphs((disc_loss, disc_accuracy), (gen_loss, gen_mse),
                                 fm.compile_directory(fm.SpecialFolder.FIGURES) + 'Experiment-' + experiment_id + '/Training/',
                                 experiment_id, fold_id, epochs, animate=True)

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
    print("[8] Data Visualisation Tools")

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
    elif user_input == "8":
        data_visualisation_menu()

    print("")
    return user_input


def main():
    global multiprocessing_pool

    multiprocessing_pool = Pool()
    setup()

    print_notice("Testing pre-processing pipeline", mt.MessagePrefix.DEBUG)

    test_processing = False

    if test_processing:
        fm.current_directory = "/run/media/***REMOVED***/Experiments/Doctorate/Phase1/data/CT-Scans/01_Unprocessed/Aggregate-CT-Scans/15-2974/"
        data_directory = fm.current_directory

        print_notice("Preprocessing %s..." % data_directory)
        images = load_images_from_directory(data_directory, multiprocessing_pool=multiprocessing_pool)
        images = apply_preprocessing_pipeline(images, multiprocessing_pool)

        print_notice("Saving processed images... ", mt.MessagePrefix.INFORMATION, end='')
        save_images(images, "processed_scan", fm.SpecialFolder.PROCESSED_SCANS, multiprocessing_pool)
        print("done!")

    segment_dir = fm.SpecialFolder.SEGMENTED_CORE_SCANS

    fm.current_directory = "/run/media/***REMOVED***/Experiments/Doctorate/Phase1/data/CT-Scans/02_Processed/Aggregate-CT-Scans/15-2974/"
    data_directory = fm.current_directory

    segment_images(data_directory, segment_dir, multiprocessing_pool)

    print_notice("Exiting early for manual image checking...")
    exit(0)

    print_notice("Please wait while data collections are processed...", mt.MessagePrefix.INFORMATION)

    # | DATA PREPARATION MODULE
    if sm.get_setting("ENABLE_PREPROCESSING") == "True":
        im.preprocess_images(multiprocessing_pool)

    im.extract_rois(multiprocessing_pool)

    # \-- | DATA LOADING SUB-MODULE
    if sm.get_setting("ENABLE_SEGMENTATION") == "True":
        im.segment_all_images(multiprocessing_pool, True)
        im.segment_all_images(multiprocessing_pool, False)

    generate_voxels(True, multiprocessing_pool)
    generate_voxels(False, multiprocessing_pool)

    # \-- | SEGMENT-TO-VOXEL CONVERSION SUB-MODULE

    update_database_core_analyses()

    if sm.get_setting("ENABLE_3D_MODEL_GENERATION") == "True":
        model_all_cores(multiprocessing_pool, use_rois=False)
        model_all_cores(multiprocessing_pool, use_rois=True)

    while main_menu() != "EXIT":
        continue

        # | GENERATIVE ADVERSARIAL NETWORK MODULE


if __name__ == "__main__":
    freeze_support()
    #try:
    main()
    #except:
#        send_email("Software Failed")

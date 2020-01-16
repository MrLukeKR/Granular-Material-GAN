# Utilities >>>
import os
from multiprocessing import Pool
# <<< Utilities

# Image Processing >>>
import ImageTools.Preprocessor as preproc
import ImageTools.Postprocessor as postproc
import ImageTools.VoxelProcessor as vp
import ImageTools.ImageManager as im

from ImageTools.Segmentation.TwoDimensional import KMeans2D as segmentor2D

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

pool = None
model_loaded = None
architecture_loaded = None


def print_introduction():
    print("   Optimal Material Generator using Generative Adversarial Networks   ")
    print("                    Developed by ***REMOVED*** (BSc)                    ")
    print("In fulfilment of Doctor of Engineering at the University of Nottingham")
    print("----------------------------------------------------------------------")
    print()


def apply_preprocessing_pipeline(images):
    Logger.print("Pre-processing Image Collection...")
    processed_images = images

    processed_images = preproc.reshape_images(processed_images, pool=pool)
    processed_images = preproc.normalise_images(processed_images, pool=pool)
    processed_images = preproc.denoise_images(processed_images, pool=pool)
    # processed_images = itp.remove_empty_scans(processed_images)
    # processed_images = itp.remove_anomalies(processed_images)
    # processed_images = itp.remove_backgrounds(processed_images)

    return processed_images


def process_voxels(images):
    voxels = list()

    if sm.configuration.get("ENABLE_VOXEL_SEPARATION") == "True":
        voxels, dimensions = vp.volume_to_voxels(images, int(sm.configuration.get("VOXEL_RESOLUTION")))

        if sm.configuration.get("ENABLE_VOXEL_INPUT_SAVING") == "True":
            im.save_voxel_images(voxels, "Unsegmented")
    return voxels, dimensions


def setup():
    global pool

    sm.load_settings()
    fm.assign_special_folders()

    dm.connect_to_database()
    dm.initialise_database()

    pool = Pool()
    print_introduction()


def segment_images():
    existing_scans = set(fm.prepare_directories(fm.SpecialFolder.SEGMENTED_SCANS))
    existing_scans = list(map(lambda x: x.split('/')[-2], existing_scans))

    fm.data_directories = list(d for d in fm.prepare_directories(fm.SpecialFolder.PROCESSED_SCANS)
                               if d.split('/')[-2] not in existing_scans)

    for data_directory in fm.data_directories:
        images = im.load_images_from_directory(data_directory)
        fm.current_directory = data_directory.replace(fm.get_directory(fm.SpecialFolder.PROCESSED_SCANS), '')

        if not fm.current_directory.endswith('/'):
            fm.current_directory += '/'
        sm.images = images

        #        ind = 0
        #        for image in images:
        #            im.save_image(image, str(ind), 'data/core/train/image/', False)
        #            ind += 1

        # \-- | 2D DATA SEGMENTATION SUB-MODULE
        voids = list()
        clean_voids = list()

        aggregates = list()
        clean_aggregates = list()

        binders = list()
        clean_binders = list()

        segments = list()
        clean_segments = list()

        Logger.print("Segmenting images... ", end="", flush=True)
        for ind, res in enumerate(pool.map(segmentor2D.segment_image, images)):
            voids.insert(ind, res[0])
            aggregates.insert(ind, res[1])
            binders.insert(ind, res[2])
            segments.insert(ind, res[3])
        Logger.print("done!")

        Logger.print("Post-processing Segment Collection...")

        ENABLE_POSTPROCESSING = False

        if ENABLE_POSTPROCESSING:
            Logger.print("\tCleaning Voids...", end="", flush=True)
            for ind, res in enumerate(pool.map(postproc.clean_segment, voids)):
                clean_voids.insert(ind, res)
            voids = clean_voids
            Logger.print("done!")

            Logger.print("\tCleaning Aggregates...", end="", flush=True)
            for ind, res in enumerate(pool.map(postproc.clean_segment, aggregates)):
                clean_aggregates.insert(ind, res)
            aggregates = clean_aggregates
            Logger.print("done!")

            Logger.print("\tCleaning Binders...", end="", flush=True)
            for ind, res in enumerate(pool.map(postproc.clean_segment, binders)):
                clean_binders.insert(ind, res)
            binders = clean_binders
            Logger.print("done!")

            Logger.print("\tCleaning Segments...", end="", flush=True)
            for ind, res in enumerate(pool.map(postproc.clean_segment, segments)):
                clean_segments.insert(ind, res)
            segments = clean_segments
            Logger.print("done!")

        Logger.print("Saving segmented images... ", end='')
        im.save_images(binders, "binder", fm.SpecialFolder.SEGMENTED_SCANS)
        im.save_images(aggregates, "aggregate", fm.SpecialFolder.SEGMENTED_SCANS)
        im.save_images(voids, "void", fm.SpecialFolder.SEGMENTED_SCANS)
        im.save_images(segments, "segment", fm.SpecialFolder.SEGMENTED_SCANS)
        Logger.print("done!")


def preprocess_images():
    existing_scans = set(fm.prepare_directories(fm.SpecialFolder.PROCESSED_SCANS))
    existing_scans = list(map(lambda x: x.split('/')[-2], existing_scans))

    fm.data_directories = list(d for d in fm.prepare_directories(fm.SpecialFolder.UNPROCESSED_SCANS)
                               if d.split('/')[-2] not in existing_scans)

    for data_directory in fm.data_directories:
        fm.current_directory = data_directory.replace(fm.get_directory(fm.SpecialFolder.UNPROCESSED_SCANS), '')

        images = im.load_images_from_directory(data_directory)
        images = apply_preprocessing_pipeline(images)

        Logger.print("Saving processed images... ", end='')
        im.save_images(images, "processed_scan", fm.SpecialFolder.PROCESSED_SCANS)
        Logger.print("done!")


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


def load_architecture_from_database(architectureID=None):
    pass


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
    loaded_discriminator, loaded_generator = mlm.load_network(model_location_prefix + selected_model[1],
                                                              model_location_prefix + selected_model[0])

    if loaded_discriminator is not None and loaded_generator is not None:
        print_notice("Model successfully loaded", mt.MessagePrefix.SUCCESS)

        loaded_generator.summary()
        loaded_discriminator.summary()
    else:
        print_notice("Error loading model!", mt.MessagePrefix.ERROR)
        raise ValueError


def experiment_menu():
    if model_loaded is None:
        print_notice("Please load a model first!", mt.MessagePrefix.WARNING)
        return

    print("- Experiment Menu -")
    print("")
    print("[1] K-Cross Fold Validation (Batch Training)")
    print("[2] K-Cross Fold Validation (Entire Dataset Generator)")
    print("[3] Single Model (Batch Training)")
    print("[4] Single Model (Entire Dataset Generator)")

    user_input = input("")

    MethodologyLogger.Logger(fm.get_directory(fm.SpecialFolder.LOGS))
    if user_input == "1":
        ExperimentRunner.run_k_fold_cross_validation_experiment(fm.data_directories, 10)
    elif user_input == "2":
        raise NotImplementedError
    elif user_input == "3":
        raise NotImplementedError
    elif user_input == "4":
        raise NotImplementedError


def core_analysis_menu():
    print("- Core Analysis Menu -")
    print("[1] Perform all calculations")
    print("[2] Calculate Air Voice Content")

    print("")
    print("[ANYTHING ELSE] Return to Main Menu")

    user_input = input("Enter a menu option > ")

    if user_input == "1":
        raise NotImplementedError
    elif user_input == "2":
        raise NotImplementedError


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
        gen, disc = mlm.design_gan_architecture()
        model_loaded = (gen, disc)
    elif user_input == "2":
        load_architecture_from_database()
    elif user_input == "3":
        load_model_from_database()
    elif user_input == "4":
        experiment_menu()
    elif user_input == "5":
        if model_loaded is not None:
            raise NotImplementedError
        else:
            print_notice("Please load a model first!", mt.MessagePrefix.WARNING)
    elif user_input == "5":
        core_analysis_menu()

    print("")
    return user_input


def main():
    global pool

    setup()

    print("Please wait while data collections are processed...")

# | DATA PREPARATION MODULE
    if sm.configuration.get("ENABLE_PREPROCESSING") == "True":
        preprocess_images()

# \-- | DATA LOADING SUB-MODULE
    if sm.configuration.get("ENABLE_SEGMENTATION") == "True":
        segment_images()

    generate_voxels()
# \-- | SEGMENT-TO-VOXEL CONVERSION SUB-MODULE

    while main_menu() != "EXIT":
        continue

        # | GENERATIVE ADVERSARIAL NETWORK MODULE


if __name__ == "__main__":
    main()

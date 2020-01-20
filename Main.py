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
    dimensions = None

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

    mlm.initialise()

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
        arch_id, gen, disc = mlm.design_gan_architecture()
        architecture_loaded = (arch_id, gen, disc)

        print("Would you like to train a model with this architecture? (Can be done later from the main menu)")
        user_input = input("CHOICE [Y/N] > ")

        if user_input[0].upper() == "Y":
            experiment_menu()
    elif user_input == "2":
        arch_id, gen, disc = mlm.load_architecture_from_database()
        architecture_loaded = (arch_id, gen, disc)
    elif user_input == "3":
        model_loaded, architecture_loaded = mlm.load_model_from_database()
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

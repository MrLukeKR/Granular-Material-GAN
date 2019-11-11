from GAN import DCGAN

# File I/O >>>
from glob import glob
from os import walk
# <<< File I/O

# Utilities >>>
from multiprocessing import Pool
# <<< Utilities

# Image Processing >>>
import ImageTools.Preprocessor as preproc
import ImageTools.Postprocessor as postproc
import ImageTools.VoxelProcessor as vp
import ImageTools.ImageManager as im

from ImageTools.Segmentation.ThreeDimensional import StackedOtsu2D as segmentor3D
from ImageTools.Segmentation.TwoDimensional import KMeans2D as segmentor2D

from Settings import SettingsManager as sm
from Settings import FileManager as fm
# <<< Image Processing

# Machine Learning >>>
import Settings.MachineLearningManager as mlm
# <<< Machine Learning

# Experiments >>>
from ExperimentTools import MethodologyLogger, ExperimentRunner, DatasetProcessor
# <<< Experiments

pool = None


def print_introduction():
    print("   Optimal Material Generator using Generative Adversarial Networks   ")
    print("                    Developed by ***REMOVED*** (BSc)                    ")
    print("In fulfilment of Doctor of Engineering at the University of Nottingham")
    print("----------------------------------------------------------------------")
    print()


def preprocess_image_collection(images):
    print("Pre-processing Image Collection...")
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
        voxels = vp.split_to_voxels(images, int(sm.configuration.get("VOXEL_RESOLUTION")))

        if sm.configuration.get("ENABLE_VOXEL_INPUT_SAVING") == "True":
            im.save_voxel_images(voxels, "Unsegmented")
    return voxels


def main():
    global pool

    pool = Pool()
    print_introduction()

    sm.load_settings()
    fm.assign_special_folders()

# | DATA PREPARATION MODULE
    if sm.configuration.get("ENABLE_PREPROCESSING") == "True":
        existing_scans = set(fm.prepare_directories(fm.SpecialFolder.PROCESSED_SCANS))
        existing_scans = list(map(lambda x: x.split('/')[-2], existing_scans))

        fm.data_directories = list(d for d in fm.prepare_directories(fm.SpecialFolder.UNPROCESSED_SCANS)
                                   if d.split('/')[-2] not in existing_scans)

        for data_directory in fm.data_directories:
            fm.current_directory = data_directory.replace(fm.get_directory(fm.SpecialFolder.UNPROCESSED_SCANS), '')

            images = im.load_images_from_directory(data_directory)
            images = preprocess_image_collection(images)

            print("Saving processed images... ", end='')
            im.save_images(images, "processed_scan", fm.SpecialFolder.PROCESSED_SCANS)
            print("done!")

# \-- | DATA LOADING SUB-MODULE

    if sm.configuration.get("ENABLE_SEGMENTATION") == "True":
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

            print("Segmenting images... ", end="", flush=True)
            for ind, res in enumerate(pool.map(segmentor2D.segment_image, images)):
                voids.insert(ind, res[0])
                aggregates.insert(ind, res[1])
                binders.insert(ind, res[2])
                segments.insert(ind, res[3])
            print("done!")

            print("Post-processing Segment Collection...")

            ENABLE_POSTPROCESSING=False

            if ENABLE_POSTPROCESSING:
                print("\tCleaning Voids...", end="", flush=True)
                for ind, res in enumerate(pool.map(postproc.clean_segment, voids)):
                    clean_voids.insert(ind, res)
                voids = clean_voids
                print("done!")

                print("\tCleaning Aggregates...", end="", flush=True)
                for ind, res in enumerate(pool.map(postproc.clean_segment, aggregates)):
                    clean_aggregates.insert(ind, res)
                aggregates = clean_aggregates
                print("done!")

                print("\tCleaning Binders...", end="", flush=True)
                for ind, res in enumerate(pool.map(postproc.clean_segment, binders)):
                    clean_binders.insert(ind, res)
                binders = clean_binders
                print("done!")

                print("\tCleaning Segments...", end="", flush=True)
                for ind, res in enumerate(pool.map(postproc.clean_segment, segments)):
                    clean_segments.insert(ind, res)
                segments = clean_segments
                print("done!")

            print("Saving segmented images... ", end='')
            im.save_images(binders, "binder", fm.SpecialFolder.SEGMENTED_SCANS)
            im.save_images(aggregates, "aggregate", fm.SpecialFolder.SEGMENTED_SCANS)
            im.save_images(voids, "void", fm.SpecialFolder.SEGMENTED_SCANS)
            im.save_images(segments, "segment", fm.SpecialFolder.SEGMENTED_SCANS)
            print("done!")

# \-- | DATA REPRESENTATION CONVERSION SUB-MODULE
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
                    print("Converting segments in '" + data_directory + "' to voxels...")
                    printed_status = True

                print("\tLoading " + segment + " data...\r\n\t\t", end='')
                images = im.load_images_from_directory(data_directory, segment)
                voxels = process_voxels(images)

                print("\t\tSaving " + segment + " voxels...\r\n\t\t", end='')
                vp.save_voxels(voxels, voxel_directory, filename)


            # im.save_voxel_image_collection(voxels, fm.SpecialFolder.VOXEL_DATA, "/Unsegmented/")

        experiment = MethodologyLogger.Logger("", "")

# | GENERATIVE ADVERSARIAL NETWORK MODULE
        ExperimentRunner.run_k_fold_cross_validation_experiment(fm.data_directories, 10)


if __name__ == "__main__":
    main()

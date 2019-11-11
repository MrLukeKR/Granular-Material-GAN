from ExperimentTools import DatasetProcessor
from GAN import DCGAN
from Settings import FileManager as fm, SettingsManager as sm
from ImageTools import VoxelProcessor as vp


def run_train_test_split_experiment(aggregates, binders, split_percentage):

    pass


def run_k_fold_cross_validation_experiment(dataset_directories, k):
    training_sets, testing_set = DatasetProcessor.dataset_to_k_cross_fold(dataset_directories, k)

    for fold in range(k):
        aggregates = list()
        binders = list()

        print("Running Cross Validation Fold " + str(fold + 1) + "/" + str(k))
        for training_set in training_sets[fold]:
            for directory in training_set:
                print("\tLoading voxels from " + directory + "... ", end='')
                fm.current_directory = directory.replace(fm.get_directory(fm.SpecialFolder.SEGMENTED_SCANS), '')

                voxel_directory = fm.get_directory(fm.SpecialFolder.VOXEL_DATA) + fm.current_directory[0:-1]

                aggregates.append(vp.load_voxels(voxel_directory, "aggregate_" + sm.configuration.get("VOXEL_RESOLUTION")))

                binders.append(vp.load_voxels(voxel_directory, "binder_" + sm.configuration.get("VOXEL_RESOLUTION")))
                print("done!")

            discriminator = DCGAN.DCGANDiscriminator(aggregates, 2, 5)
            generator = DCGAN.DCGANGenerator(aggregates, 2, 5)

            DCGAN.Network._discriminator = discriminator
            DCGAN.Network._generator = generator

            network = DCGAN.Network.create_network(aggregates)

            batch_size = 32

            DCGAN.Network.train_network(int(len(aggregates) / batch_size), batch_size, aggregates, binders)


def run_on_existing_gan(aggregates, binders):
    # discriminator, generator = mlm.load_network()
    #
    # if discriminator is None or generator is None:
    #     discriminator = DCGAN.DCGANDiscriminator(aggregates, 2, 5)
    #     generator = DCGAN.DCGANGenerator(aggregates, 2, 5)
    #
    # DCGAN.Network._discriminator = discriminator
    # DCGAN.Network._generator = generator
    #
    # network = DCGAN.Network.create_network(aggregates)
    #
    # batch_size = 32
    #
    # DCGAN.Network.train_network(int(len(aggregates) / batch_size), batch_size, aggregates, binders)
    pass

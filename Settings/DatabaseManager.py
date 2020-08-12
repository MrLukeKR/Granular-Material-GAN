import os
import mysql.connector

from Settings import FileManager as fm, SettingsManager as sm, MessageTools as mt
from Settings.MessageTools import print_notice

db = None
db_cursor = None
database_connected = False


def reinitialise_database():
    global db, db_cursor, database_connected

    if not database_connected:
        print_notice("Database is not connected!", mt.MessagePrefix.ERROR)
        exit(-1)

    print_notice("Deleting databases... ", mt.MessagePrefix.INFORMATION, end='')
    db_cursor.execute("DROP DATABASE ct_scans;")
    db_cursor.execute("DROP DATABASE ***REMOVED***_Phase1;")
    print("done!")

    initialise_database()


def connect_to_database():
    global db, db_cursor, database_connected

    try:
        db = mysql.connector.connect(
            host=sm.auth.get("DB_HOST"),
            port=sm.auth.get("DB_PORT"),
            user=sm.auth.get("DB_USER"),
            passwd=sm.auth.get("DB_PASS")
        )

        db_cursor = db.cursor()
        db.autocommit = True
        database_connected = True
    except Exception as exc:
        print(exc)
        exit(-1)


def get_cores_from_database(ignore_blacklist=False):
    db_cursor.execute("USE ct_scans;")

    sql = "SELECT * FROM asphalt_cores"
    sql += " WHERE Blacklist = 0" if not ignore_blacklist else ";"

    db_cursor.execute(sql)

    cores = db_cursor.fetchall()

    db_cursor.execute("USE ***REMOVED***_Phase1;")

    return cores


def get_experiment_information():
    sql = "SELECT experiments.ID, experiments.Timestamp, experiments.Folds, experiments.Epochs, experiments.BatchSize," \
          "COUNT(training.ExperimentID) FROM experiments, training " \
          "WHERE experiments.ID = training.ExperimentID GROUP BY experiments.ID;"

    db_cursor.execute(sql)

    return db_cursor.fetchall()


def get_training_data(experiment_id):
    sql = "SELECT * FROM training WHERE ExperimentID=" + experiment_id + ';'
    db_cursor.execute(sql)

    return db_cursor.fetchall()


def populate_ct_scan_database():
    db_cursor.execute("CREATE DATABASE IF NOT EXISTS ct_scans;")
    db_cursor.execute("USE ct_scans;")
    db_cursor.execute("CREATE TABLE IF NOT EXISTS asphalt_cores"
                      "(ID varchar(10) NOT NULL,"
                      "ScanDirectory VARCHAR(256) NOT NULL,"
                      "ModelFileLocation VARCHAR(256) NULL,"
                      "TargetAirVoidContent DOUBLE NULL,"
                      "MeasuredAirVoidContent DOUBLE NULL,"
                      "MasticContent DOUBLE NULL,"
                      "Tortuosity DOUBLE NULL,"
                      "EulerNumber DOUBLE NULL,"
                      "AverageVoidDiameter DOUBLE NULL,"
                      "Blacklist BOOLEAN NOT NULL DEFAULT FALSE,"
                      "Notes VARCHAR(1024) NULL,"
                      "PRIMARY KEY (ID));")

    db_cursor.execute("CREATE TABLE IF NOT EXISTS generated_asphalt_cores"
                      "(ID INT(11) AUTO_INCREMENT NOT NULL,"
                      "GeneratorModelID INT(11 ) NOT NULL,"
                      "VoxelDirectory VARCHAR(256) NOT NULL,"
                      "ModelFileLocation VARCHAR(256) NULL,"
                      "AirVoidContent DOUBLE NULL,"
                      "MasticContent DOUBLE NULL,"
                      "Tortuosity DOUBLE NULL,"
                      "EulerNumber DOUBLE NULL,"
                      "AverageVoidDiameter DOUBLE NULL,"
                      "Notes VARCHAR(1024) NULL,"
                      "PRIMARY KEY (ID),"
                      "FOREIGN KEY (GeneratorModelID) REFERENCES ***REMOVED***_Phase1.model_instances(ID));")

    unprocessed_ct_directory = fm.compile_directory(fm.SpecialFolder.UNPROCESSED_SCANS)

    ct_ids = [name for name in os.listdir(unprocessed_ct_directory)]

    unprocessed_ct_directory = unprocessed_ct_directory.replace(fm.compile_directory(fm.SpecialFolder.ROOT), '')

    for ct_id in ct_ids:
        directory = unprocessed_ct_directory + ct_id
        sql = "INSERT INTO asphalt_cores(ID, ScanDirectory) " \
              "VALUES (%s, %s) ON DUPLICATE KEY UPDATE ScanDirectory=%s;"
        values = (ct_id, directory, directory)

        db_cursor.execute(sql, values)


def initialise_settings():
    db_cursor.execute("USE ***REMOVED***_Phase1;")

    db_cursor.execute("CREATE TABLE IF NOT EXISTS settings"
                      "(Name VARCHAR(64) NOT NULL,"
                      "Value VARCHAR(64) NULL,"
                      "PRIMARY KEY(Name));")

    db_cursor.execute("INSERT IGNORE INTO settings VALUES "
                      "('IO_ROOT_DIR', NULL),"
                      "('IO_SCAN_ROOT_DIR', NULL),"
                      "('IO_EXPERIMENT_ROOT_DIR', NULL),"
                      "('IO_SCAN_TYPE', NULL),"
                      
                      "('IO_UNPROCESSED_SCAN_ROOT_DIR', 'Unprocessed'),"
                      "('IO_PROCESSED_SCAN_ROOT_DIR', 'Processed'),"
                      "('IO_ROI_SCAN_ROOT_DIR', 'Regions-Of-Interest'),"
                      
                      "('IO_SEGMENTED_SCAN_ROOT_DIR', 'Segmented'),"
                      "('IO_SEGMENTED_ROI_SCAN_DIR', 'Regions-Of-Interest'),"
                      "('IO_SEGMENTED_CORE_SCAN_DIR', 'Cores'),"
                      
                      "('IO_RESULTS_ROOT_DIR', 'Results'),"
                      "('IO_FIGURES_ROOT_DIR', 'Figures'),"
                      "('IO_VOXEL_DATA_ROOT_DIR', 'Voxels'),"
                      "('IO_ROI_VOXEL_DATA_DIR', 'Regions-Of-Interest'),"
                      "('IO_CORE_VOXEL_DATA_DIR', 'Cores'),"
                      "('IO_3D_MODEL_ROOT_DIR', '3D-Models'),"
                      "('IO_GENERATED_VOXEL_ROOT_DIR', 'Generated-Voxels'),"
                      "('IO_GENERATED_CORE_ROOT_DIR', 'Generated-Cores'),"
                      "('IO_ASPHALT_3D_MODEL_DIR', 'Real-Asphalt'),"
                      "('IO_GENERATED_ASPHALT_3D_MODEL_DIR', 'Generated-Asphalt'),"
                      "('IO_ASPHALT_CORE_MODELS', 'Cores'),"
                      "('IO_ASPHALT_ROI_MODELS', 'Regions-Of-Interest'),"
                      "('IO_GENERATED_ASPHALT_CORE_MODELS', 'Cores'),"
                      "('IO_GENERATED_ASPHALT_ROI_MODELS', 'Regions-Of-Interest'),"
                      "('IO_MODEL_ROOT_DIR', 'Models'),"
                      "('IO_DATASET_ROOT_DIR', 'Datasets'),"
                      "('IO_ROI_DATASET_DIR', 'Regions-Of-Interest'),"
                      "('IO_CORE_DATASET_DIR', 'Cores'),"
                      "('IO_LOG_ROOT_DIR', 'Logs'),"
                      "('IO_IMAGE_FILETYPE', 'pdf'),"
                      "('IO_OUTPUT_DPI', '500'),"
                      "('IO_GAN_OUTPUT_THRESHOLD', '254'),"

                      "('ROI_IMAGE_METRIC', 'PIXELS'),"
                      "('ROI_IMAGE_DIMENSIONS', '800,800'),"
                      "('ROI_DEPTH_METRIC', 'PERCENTAGE'),"
                      "('ROI_DEPTH_DIMENSION', '80'),"

                      "('USE_REGIONS_OF_INTEREST', 'True'),"

                      "('VOXEL_MESH_STEP_SIZE', '6'),"
                      "('VOXEL_RESOLUTION', '64'),"
                      "('VOXEL_RESOLVE_METHOD', 'PADDING'),"
                      
                      "('IMAGE_CHANNELS', '1'),"
                      "('LABEL_SEGMENTS', '3'),"

                      "('PIXELS_TO_MM', '10'),"
                      "('SCAN_STACK_STRIP_PERCENT', '5'),"
                      "('MAXIMUM_BLOB_AREA', '150'),"

                      "('TRAINING_BATCH_SIZE', '32'),"
                      "('TRAINING_EPOCHS', '1000'),"
                      "('TRAINING_EPOCH_STEPS', '500'),"
                      "('TRAINING_USE_BATCH_NORMALISATION', 'False'),"  
                      "('TRAINING_ANIMATION_BATCH_STEP', '10'),"

                      "('EMAIL_NOTIFICATION_RECIPIENT', NULL),"

                      "('ENABLE_FIX_GAN_OUTPUT_OVERLAP', 'False'),"
                      "('ENABLE_IMAGE_SAVING', 'False'),"
                      "('ENABLE_VOXEL_INPUT_SAVING', 'False'),"
                      "('ENABLE_VOXEL_OUTPUT_SAVING', 'False'),"
                      "('ENABLE_IMAGE_DISPLAY', 'False'),"
                      "('ENABLE_VOXEL_SEPARATION', 'True'),"
                      "('ENABLE_SEGMENTATION', 'True'),"
                      "('ENABLE_PREPROCESSING', 'True'),"
                      "('ENABLE_POSTPROCESSING', 'True'),"
                      "('ENABLE_GAN_TRAINING', 'False'),"
                      "('ENABLE_TRAINING_ANIMATION', 'True'),"
                      "('ENABLE_VOXEL_PLOT_GENERATION', 'True'),"
                      "('ENABLE_GAN_OUTPUT_HISTOGRAM', 'True'),"
                      "('ENABLE_GAN_GENERATION', 'False');")

    db_cursor.execute("SELECT Name FROM settings WHERE Value IS NULL;")
    uninitialised_settings = db_cursor.fetchall()

    if len(uninitialised_settings) > 0:
        print_notice("There are uninitialised settings in the database!", mt.MessagePrefix.ERROR)
        print_notice("\tPlease assign values to the following settings:")
        for setting in uninitialised_settings:
            print("\t\t%s" % setting)
        exit(-1)


def initialise_machine_learning_database():
    db_cursor.execute("CREATE DATABASE IF NOT EXISTS ***REMOVED***_Phase1;")
    db_cursor.execute("USE ***REMOVED***_Phase1;")

    db_cursor.execute("CREATE TABLE IF NOT EXISTS model_architectures"
                      "(ID INT AUTO_INCREMENT NOT NULL,"
                      "Description VARCHAR(128) NOT NULL,"
                      "NetworkType VARCHAR(256) NOT NULL,"
                      "GeneratorStrides VARCHAR(15) NOT NULL,"
                      "GeneratorKernelSize VARCHAR(15) NOT NULL,"
                      "GeneratorNumberOfLevels INT NOT NULL,"
                      "GeneratorFilters VARCHAR(15) NOT NULL,"
                      "GeneratorNormalisationMomentum VARCHAR(15) NOT NULL,"
                      "GeneratorActivationAlpha VARCHAR(15) NOT NULL,"
                      "DiscriminatorStrides VARCHAR(15) NOT NULL,"
                      "DiscriminatorKernelSize VARCHAR(15) NOT NULL,"
                      "DiscriminatorNumberOfLevels INT NOT NULL,"
                      "DiscriminatorFilters VARCHAR(15) NOT NULL,"
                      "DiscriminatorNormalisationMomentum VARCHAR(15) NOT NULL,"
                      "DiscriminatorActivationAlpha VARCHAR(15) NOT NULL,"
                      "PRIMARY KEY(ID),"
                      "UNIQUE (NetworkType, GeneratorStrides, GeneratorKernelSize, GeneratorNumberOfLevels,"
                      "GeneratorFilters, GeneratorNormalisationMomentum, GeneratorActivationAlpha, "
                      "DiscriminatorStrides, DiscriminatorKernelSize, DiscriminatorNumberOfLevels, "
                      "DiscriminatorFilters, DiscriminatorNormalisationMomentum, DiscriminatorActivationAlpha));")

    db_cursor.execute("CREATE TABLE IF NOT EXISTS model_instances"
                      "(ID INT(11) AUTO_INCREMENT NOT NULL,"
                      "ArchitectureID INT NOT NULL,"
                      "GeneratorFilePath VARCHAR(256) NOT NULL,"
                      "DiscriminatorFilePath VARCHAR(256) NOT NULL,"
                      "PRIMARY KEY (ID),"
                      "UNIQUE(GeneratorFilePath),"
                      "UNIQUE(DiscriminatorFilePath),"
                      "FOREIGN KEY (ArchitectureID) REFERENCES model_architectures(ID));")

    db_cursor.execute("CREATE TABLE IF NOT EXISTS experiments "
                      "(ID INT AUTO_INCREMENT,"
                      "Timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,"
                      "Epochs INT NULL,"
                      "Folds INT NULL,"
                      "BatchSize INT NULL,"
                      "CPUCores INT NOT NULL,"
                      "CPUSpeed VARCHAR(64) NOT NULL,"
                      "GPUVRAMSize VARCHAR(64) NULL,"
                      "GPUCUDACores INT NULL,"
                      "RAMSize VARCHAR(64) NOT NULL,"
                      "PRIMARY KEY(ID));")

    db_cursor.execute("CREATE TABLE IF NOT EXISTS model_experiments"
                      "(ExperimentID INT NOT NULL,"
                      "ModelInstanceID INT NOT NULL,"
                      "PRIMARY KEY(ExperimentID, ModelInstanceID),"
                      "FOREIGN KEY (ExperimentID) REFERENCES experiments(ID),"
                      "FOREIGN KEY (ModelInstanceID) REFERENCES model_instances(ID));")

    db_cursor.execute("CREATE TABLE IF NOT EXISTS results "
                      "(ID INT AUTO_INCREMENT,"
                      "Timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,"
                      "ExperimentID INT NOT NULL,"
                      "FinalDiscriminatorLoss DOUBLE NOT NULL,"
                      "FinalDiscriminatorAccuracy DOUBLE NOT NULL,"
                      "FinalGeneratorLoss DOUBLE NOT NULL,"
                      "FinalGeneratorMSE DOUBLE NOT NULL,"
                      "PRIMARY KEY(ID),"
                      "FOREIGN KEY (ExperimentID) REFERENCES experiments(ID));")

    db_cursor.execute("CREATE TABLE IF NOT EXISTS training "
                      "(ID INT AUTO_INCREMENT,"
                      "Timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,"
                      "ExperimentID INT NOT NULL,"
                      "Fold INT NOT NULL,"
                      "Epoch INT NOT NULL,"
                      "Batch INT NOT NULL,"
                      "DiscriminatorLoss DOUBLE NOT NULL,"
                      "DiscriminatorAccuracy DOUBLE NOT NULL,"
                      "GeneratorLoss DOUBLE NOT NULL,"
                      "GeneratorMSE DOUBLE NOT NULL,"
                      "PRIMARY KEY(ID),"
                      "FOREIGN KEY (ExperimentID) REFERENCES experiments(ID));")


def initialise_database():
    global db, db_cursor, database_connected

    if not database_connected:
        print_notice("Database is not connected!", mt.MessagePrefix.ERROR)
        exit(-1)

    print_notice("Initialising database... ", mt.MessagePrefix.INFORMATION)
    try:
        initialise_settings()
        initialise_machine_learning_database()
    except Exception as exc:
        print_notice(str(exc), mt.MessagePrefix.ERROR)
        exit(-1)

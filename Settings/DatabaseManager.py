import os
import mysql.connector

from Settings import SettingsManager as sm, MessageTools as mt, FileManager as fm
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
            host=sm.configuration.get("DB_HOST"),
            port=sm.configuration.get("DB_PORT"),
            user=sm.configuration.get("DB_USER"),
            passwd=sm.configuration.get("DB_PASS")
        )

        db_cursor = db.cursor()
        db.autocommit = True
        database_connected = True
    except Exception as exc:
        print(exc)
        exit(-1)


def get_cores_from_database():
    db_cursor.execute("USE ct_scans;")

    db_cursor.execute("SELECT * FROM asphalt_cores;")
    cores = db_cursor.fetchall()

    db_cursor.execute("USE ***REMOVED***_Phase1;")

    return cores


def populate_ct_scan_database():
    db_cursor.execute("CREATE DATABASE IF NOT EXISTS ct_scans;")
    db_cursor.execute("USE ct_scans;")
    db_cursor.execute("CREATE TABLE IF NOT EXISTS asphalt_cores"
                      "(ID varchar(10) NOT NULL,"
                      "ScanDirectory VARCHAR(256) NOT NULL,"
                      "ModelFileLocation VARCHAR(256) NULL,"
                      "AirVoidContent DOUBLE NULL,"
                      "MasticContent DOUBLE NULL,"
                      "Tortuosity DOUBLE NULL,"
                      "EulerNumber DOUBLE NULL,"
                      "AverageVoidDiameter DOUBLE NULL,"
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

    for ct_id in ct_ids:
        directory = unprocessed_ct_directory + ct_id
        sql = "INSERT INTO asphalt_cores(ID, ScanDirectory) " \
              "VALUES (%s, %s) ON DUPLICATE KEY UPDATE ScanDirectory=%s;"
        values = (ct_id, directory, directory)

        db_cursor.execute(sql, values)


def initialise_machine_learning_database():
    db_cursor.execute("CREATE DATABASE IF NOT EXISTS ***REMOVED***_Phase1;")
    db_cursor.execute("USE ***REMOVED***_Phase1;")

    db_cursor.execute("CREATE TABLE IF NOT EXISTS model_architectures"
                      "(ID INT AUTO_INCREMENT NOT NULL,"
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
                      "Timestamp TIMESTAMP NOT NULL,"
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
                      "ExperimentID INT NOT NULL,"
                      "FinalDiscriminatorLoss DOUBLE NOT NULL,"
                      "FinalDiscriminatorAccuracy DOUBLE NOT NULL,"
                      "FinalGeneratorLoss DOUBLE NOT NULL,"
                      "FinalGeneratorMSE DOUBLE NOT NULL,"
                      "PRIMARY KEY(ID),"
                      "FOREIGN KEY (ExperimentID) REFERENCES experiments(ID));")

    db_cursor.execute("CREATE TABLE IF NOT EXISTS training "
                      "(ID INT AUTO_INCREMENT,"
                      "ExperimentID INT NOT NULL,"
                      "Fold INT NOT NULL,"
                      "Epoch INT NOT NULL,"
                      "TrainingSet INT NOT NULL,"
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

    print_notice("Initialising database... ", mt.MessagePrefix.INFORMATION,  end='')
    try:
        initialise_machine_learning_database()
        populate_ct_scan_database()
    except Exception as exc:
        print_notice(str(exc), mt.MessagePrefix.ERROR)
        exit(-1)

    print("done!")

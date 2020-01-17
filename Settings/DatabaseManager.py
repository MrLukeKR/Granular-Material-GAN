import mysql.connector

from Settings import SettingsManager as sm

db = None
db_cursor = None
database_connected = False


def reinitialise_database():
    global db, db_cursor, database_connected

    if not database_connected:
        print("Database is not connected!")
        exit(-1)

    print("Deleting database... ", end='')
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


def initialise_database():
    global db, db_cursor, database_connected

    if not database_connected:
        print("Database is not connected!")
        exit(-1)

    print("Initialising database... ", end='')
    try:
        db_cursor.execute("CREATE DATABASE IF NOT EXISTS ***REMOVED***_Phase1;")
        db_cursor.execute("USE ***REMOVED***_Phase1;")

        db_cursor.execute("CREATE TABLE IF NOT EXISTS model_architectures"
                          "(ID INT AUTO_INCREMENT NOT NULL,"
                          "NetworkType TEXT NOT NULL,"
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
                          "UNIQUE (ID, NetworkType, GeneratorStrides, GeneratorKernelSize, GeneratorNumberOfLevels,"
                          "GeneratorFilters, GeneratorNormalisationMomentum, GeneratorActivationAlpha, "
                          "DiscriminatorStrides, DiscriminatorKernelSize, DiscriminatorNumberOfLevels, "
                          "DiscriminatorFilters, DiscriminatorNormalisationMomentum, DiscriminatorActivationAlpha));")

        db_cursor.execute("CREATE TABLE IF NOT EXISTS model_instances"
                          "(ID INT AUTO_INCREMENT NOT NULL,"
                          "ArchitectureID INT NOT NULL,"
                          "FilePath TEXT NOT NULL,"
                          "PRIMARY KEY (ID),"
                          "FOREIGN KEY (ArchitectureID) REFERENCES model_architectures(ID)"
                          ");")

        db_cursor.execute("CREATE TABLE IF NOT EXISTS experiments "
                          "(ID INT AUTO_INCREMENT,"
                          "Timestamp TIMESTAMP NOT NULL,"
                          "Epochs INT NULL,"
                          "Folds INT NULL,"
                          "BatchSize INT NULL,"
                          "CPUCores INT NOT NULL,"
                          "CPUSpeed TEXT NOT NULL,"
                          "GPUVRAMSize TEXT NULL,"
                          "GPUCUDACores INT NULL,"
                          "RAMSize TEXT NOT NULL,"
                          "PRIMARY KEY(ID));")

        db_cursor.execute("CREATE TABLE IF NOT EXISTS model_experiments"
                          "(ID INT AUTO_INCREMENT,"
                          "ExperimentID INT NOT NULL,"
                          "ModelInstanceID INT NOT NULL,"
                          "PRIMARY KEY(ID),"
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
    except Exception as exc:
        print(exc)
        exit(-1)

    print("done!")
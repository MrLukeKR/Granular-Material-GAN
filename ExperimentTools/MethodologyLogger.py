import datetime
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

        db_cursor.execute("CREATE TABLE IF NOT EXISTS experiments "
                          "(ID INT AUTO_INCREMENT,"
                          "Timestamp TIMESTAMP NOT NULL,"
                          "PRIMARY KEY(ID));")

        db_cursor.execute("CREATE TABLE IF NOT EXISTS experiment_settings"
                          "(ID INT AUTO_INCREMENT NOT NULL,"
                          "ExperimentID INT NOT NULL,"
                          "NetworkType TEXT NOT NULL,"
                          "Folds INT NOT NULL,"
                          "Epochs INT NOT NULL,"
                          "BatchSize INT NOT NULL,"
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
                          "FOREIGN KEY (ExperimentID) REFERENCES experiments(ID));")

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

class Logger:
    _initialised = False
    experiment_id = None
    current_fold = None
    current_set = None
    log_file = None

    def __init__(self, log_directory, log_file_name=None):
        try:
            if not Logger._initialised:
                if database_connected:
                    db_cursor.execute("INSERT INTO experiments (Timestamp) VALUES (CURRENT_TIMESTAMP);")

                    db_cursor.execute("SELECT ID FROM experiments ORDER BY Timestamp DESC LIMIT 1;")
                    Logger.experiment_id = db_cursor.fetchall()[0][0]

                if not Logger.experiment_id:
                    Logger.experiment_id = Logger.get_timestamp()

                if not log_file_name:
                    log_file_name = "experiment_" + str(Logger.experiment_id)

                filepath = log_directory + log_file_name + '.log'

                Logger.log_file = open(filepath, 'w')
                Logger.print("Starting experiment logger "
                             "(Experiment " + str(Logger.experiment_id) + " @ " + str(Logger.get_timestamp()) + ")")

                Logger.initialised = True
            else:
                raise AttributeError("Logger is a singleton and has already been initialised")
        except Exception as exc:
            print(exc)

    @staticmethod
    def get_timestamp():
        dt = datetime.datetime.now()
        d, t = str(dt).split(" ")

        d = str(d).replace('-', '')
        t = str(t).split(".")[0].replace(':', '')

        return d + '-' + t

    @staticmethod
    def save_figure(figure, name=None):
        if name is None:
            name = "figure_" + Logger.get_timestamp()

        figure.save()

        pass

    @staticmethod
    def print(message="", end='\r\n', flush=False):
        if not Logger.log_file:
            raise AssertionError("Logger has not been initialised")

        print(message, end=end, flush=flush)
        Logger.log_file.write("[" + str(datetime.datetime.now()) + "] " + message + end)

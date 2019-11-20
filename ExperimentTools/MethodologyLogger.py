import datetime
import mysql.connector
from Settings import SettingsManager as sm

db = None
db_cursor = None
database_connected = False


def initialise_database():
    global db, db_cursor, database_connected

    try:
        db = mysql.connector.connect(
            host="localhost",
            user=sm.configuration.get("DB_USER"),
            passwd=sm.configuration.get("DB_PASS")
        )

        db_cursor = db.cursor()
        db.autocommit = True

        db_cursor.execute("USE ***REMOVED***_Phase1;")

        database_connected = True
    except Exception as exc:
        print(exc)


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

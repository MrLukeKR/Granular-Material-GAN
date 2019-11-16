import datetime
import mysql.connector
from Settings import SettingsManager as sm

from Settings import FileManager as fm

db = None
db_cursor = None


def initialise_database():
    global db, db_cursor

    db = mysql.connector.connect(
        host="localhost",
        user=sm.configuration.get("DB_USER"),
        passwd=sm.configuration.get("DB_PASS")
    )

    db_cursor = db.cursor()


class Logger:
    initialised = False

    experiment_id = None
    log_file = None
    log_directory = None

    def __init__(self, log_directory, log_file_name=None):
        if not Logger.initialised:
            Logger.initialised = True
        else:
            raise AttributeError("Logger is a singleton and has already been initialised")

        if not log_file_name:
            log_file_name = "experiment_" + Logger.get_timestamp()

        filepath = log_directory + log_file_name + '.log'

        Logger.log_file = open(filepath, 'w')
        Logger.print("Starting experiment logger (" + str(Logger.get_timestamp()) + ")")

    @staticmethod
    def get_timestamp():
        dt = datetime.datetime.now()
        d, t = str(dt).split(" ")

        d = str(d).replace('-', '')
        t = str(t).split(".")[0].replace(':', '')

        return d + '-' + t

    @staticmethod
    def save_figure(self, figure, name=None):
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

import datetime
import cpuinfo

from psutil import virtual_memory
from Settings import DatabaseManager as dm


def get_system_info():
    print("Gathering system information... ", end='')

    cpu_info = cpuinfo.get_cpu_info()
    # TODO: Fix gpu information here
    system_info = (cpu_info["count"], cpu_info["hz_actual"], "Unknown", -1, virtual_memory().total)

    print("done!")

    return system_info


class Logger:
    _initialised = False
    experiment_id = None
    current_fold = None
    current_set = None
    log_file = None

    def __init__(self, log_directory, log_file_name=None):
        try:
            if not Logger._initialised:
                if dm.database_connected:
                    sql = "INSERT INTO experiments (Timestamp, CPUCores, CPUSpeed, GPUVRAMSize, "\
                          "GPUCUDACores, RAMSize) "\
                          "VALUES (CURRENT_TIMESTAMP, '%s', '%s', '%s', '%s', '%s');" % get_system_info()

                    dm.db_cursor.execute(sql)

                    dm.db_cursor.execute("SELECT ID FROM experiments ORDER BY Timestamp DESC LIMIT 1;")
                    Logger.experiment_id = dm.db_cursor.fetchall()[0][0]

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
    def log_model_instance_to_database(architecture_id, gen_filepath, disc_filepath):
        if dm.database_connected:
            sql = "INSERT INTO model_instances (ArchitectureID, GeneratorFilePath, DiscriminatorFilePath) VALUES (%s, %s, %s);"
            val = (architecture_id, gen_filepath, disc_filepath)
            dm.db_cursor.execute(sql, val)

            dm.db_cursor.execute("SELECT ID FROM model_instances "
                                 "WHERE GeneratorFilePath = '%s' AND DiscriminatorFilePath = '%s'" % (gen_filepath, disc_filepath))

            return dm.db_cursor.fetchone()
        else:
            raise ConnectionError

    @staticmethod
    def log_model_experiment_to_database(experiment_id, instance_id):
        if dm.database_connected:
            sql = "INSERT INTO model_experiments (ExperimentID, ModelInstanceID) VALUES (%s, %s);"
            val = (experiment_id, instance_id)

            dm.db_cursor.execute(sql, val)
        else:
            raise ConnectionError

    @staticmethod
    def log_model_to_database(gen_settings, disc_settings):
        if dm.database_connected:
            sql = "INSERT INTO model_architectures (NetworkType," \
                  "GeneratorStrides, GeneratorKernelSize, GeneratorNumberOfLevels, GeneratorFilters, " \
                  "GeneratorNormalisationMomentum, GeneratorActivationAlpha, " \
                  "DiscriminatorStrides, DiscriminatorKernelSize, DiscriminatorNumberOfLevels, DiscriminatorFilters, " \
                  "DiscriminatorNormalisationMomentum, DiscriminatorActivationAlpha) " \
                  "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"

            val = ("DCGAN (Deep Convolutional Generative Adversarial Network)",
                   gen_settings["strides"], gen_settings["kernel_size"], gen_settings["levels"],
                   gen_settings["filters"],
                   gen_settings["normalisation_momentum"], gen_settings["activation_alpha"],
                   disc_settings["strides"], disc_settings["kernel_size"], disc_settings["levels"],
                   disc_settings["filters"],
                   disc_settings["normalisation_momentum"], disc_settings["activation_alpha"],)

            dm.db_cursor.execute(sql, val)

            dm.db_cursor.execute("SELECT ID FROM model_architectures ORDER BY ID DESC LIMIT 1;")

            result = dm.db_cursor.fetchone()[0]

            return result
        else:
            raise ConnectionError

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

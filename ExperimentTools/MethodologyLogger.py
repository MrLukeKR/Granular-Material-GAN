import datetime
import cpuinfo

from psutil import virtual_memory
from Settings import DatabaseManager as dm, MessageTools as mt
from Settings.MessageTools import print_notice


def get_system_info():
    print_notice("Gathering system information... ", end='')

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

    def __init__(self, log_directory, folds, epochs, batch_size, log_file_name=None):
        try:
            if not Logger._initialised:
                sql = "INSERT INTO experiments (Timestamp, CPUCores, CPUSpeed, GPUVRAMSize, "\
                      "GPUCUDACores, RAMSize, Folds, Epochs, BatchSize) "\
                      "VALUES (CURRENT_TIMESTAMP, '%s', '%s', '%s', '%s', '%s', " % get_system_info()
                sql += "'%s', '%s', '%s');" % (folds, epochs, batch_size)

                db_cursor = dm.get_cursor()

                db_cursor.execute(sql)

                db_cursor.execute("SELECT ID FROM experiments ORDER BY Timestamp DESC LIMIT 1;")
                Logger.experiment_id = db_cursor.fetchall()[0][0]

                if not Logger.experiment_id:
                    Logger.experiment_id = Logger.get_timestamp()

                if not log_file_name:
                    log_file_name = "experiment_" + str(Logger.experiment_id)

                filepath = log_directory + log_file_name + '.log'

                Logger.log_file = open(filepath, 'w')
                print_notice("Starting experiment logger "
                             "(Experiment " + str(Logger.experiment_id) + " @ " + str(Logger.get_timestamp()) + ")")

                Logger.initialised = True
            else:
                raise AttributeError("Logger is a singleton and has already been initialised")
        except Exception as exc:
            print(exc)

    @staticmethod
    def log_model_instance_to_database(architecture_id, gen_filepath, disc_filepath):
        print_notice("Logging model instance to database... ", mt.MessagePrefix.DEBUG, end='')

        sql = "INSERT INTO model_instances (ArchitectureID, GeneratorFilePath, DiscriminatorFilePath) VALUES (%s, %s, %s);"
        val = (architecture_id, gen_filepath, disc_filepath)
        db_cursor = dm.get_cursor()

        db_cursor.execute(sql, val)

        db_cursor.execute("SELECT ID FROM model_instances "
                          "WHERE GeneratorFilePath = '%s' AND DiscriminatorFilePath = '%s'" % (gen_filepath, disc_filepath))

        print('done')

        return db_cursor.fetchone()

    @staticmethod
    def log_model_experiment_to_database(experiment_id, instance_id):
        sql = "INSERT INTO model_experiments (ExperimentID, ModelInstanceID) VALUES (%s, %s);"
        val = (experiment_id, instance_id)

        db_cursor = dm.get_cursor()

        db_cursor.execute(sql, val)

    @staticmethod
    def log_experiment_results_to_database(experiment_id, d_loss, g_loss):
        sql = "INSERT INTO results (ExperimentID, FinalDiscriminatorLoss, FinalDiscriminatorAccuracy," \
              "FinalGeneratorLoss, FinalGeneratorMSE)" \
              "VALUES (%s, %s, %s, %s, %s)"

        val = (experiment_id, str(d_loss[-1][0]), str(d_loss[-1][1]), str(g_loss[-1][0]), str(g_loss[-1][1]))
        db_cursor = dm.get_cursor()

        db_cursor.execute(sql, val)

    @staticmethod
    def log_batch_training_to_database(epoch, batch, g_loss, d_loss, fold=None):
        sql = "INSERT INTO training (ExperimentID, Fold, Epoch, Batch, " \
              "DiscriminatorLoss, DiscriminatorAccuracy, GeneratorLoss, GeneratorMSE)" \
              "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"

        val = (str(Logger.experiment_id), str(fold if fold else 0), str(epoch), str(batch),
               str(d_loss[0]), str(d_loss[1]), str(g_loss[0]), str(g_loss[1]))

        db_cursor = dm.get_cursor()

        db_cursor.execute(sql, val)

    @staticmethod
    def log_model_to_database(gen_settings, disc_settings):
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

        db_cursor = dm.get_cursor()
        db_cursor.execute(sql, val)

        db_cursor.execute("SELECT ID FROM model_architectures ORDER BY ID DESC LIMIT 1;")

        result = db_cursor.fetchone()[0]

        return result

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

        figure.save(name)

        pass

    @staticmethod
    def print(message="", end='\r\n', flush=False):
        if not Logger.log_file:
            raise AssertionError("Logger has not been initialised")

        print(message, end=end, flush=flush)
        Logger.log_file.write("[" + str(datetime.datetime.now()) + "] " + message + end)

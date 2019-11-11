import datetime


class Logger:
    instance = None

    class __Logger:
        def __init__(self, log_directory, log_file_name):
            print("Starting experiment logger (" + str(Logger.get_timestamp()) + ")")
            Logger.instance.log_file = open(log_directory + '/' + log_file_name + '.log')

    experiment_id = None
    log_file = None
    log_directory = None

    def __init__(self, log_directory, log_file_name):
        if not Logger.instance:
            Logger.instance = Logger.__Logger(log_directory, log_file_name)
        else:
            raise AttributeError("Logger is a singleton and has already been initialised")

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
            name = Logger.get_timestamp()

        figure.save()

        pass

    @staticmethod
    def print(message, end='\r\n'):
        if not Logger.instance or not Logger.log_file:
            raise AssertionError("Logger has not been initialised")

        print(message, end=end)
        Logger.log_file.write("[" + str(datetime.datetime.now()) + "] " + message + end)

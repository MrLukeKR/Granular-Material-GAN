import datetime


class Logger:
    experiment_id = None
    log_file = None

    def __init__(self, log_file_directory, log_file_name):
        print("Starting experiment logger (" + str(Logger.get_timestamp()) + ")")
        pass

    @staticmethod
    def get_timestamp():
        dt = datetime.datetime.now()
        d, t = str(dt).split(" ")

        d = str(d).replace('-', '')
        t = str(t).split(".")[0].replace(':', '')

        return d + '_' + t

    def save_figure(self, figure, name=None):
        if name is None:
            name = Logger.get_timestamp()

        figure.save

        pass



import numpy as np
import os
import tensorflow as tf

import Settings.DatabaseManager as dm
import Settings.MessageTools as mt
from Settings.MessageTools import print_notice

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

settings_file = []
auth = dict()

rescale_factor = 0
image_channels = 1
segmentation_amount = 3
segmentation_mode = "2D"
enable_normalisation = True
display_available = "DISPLAY" in os.environ

# Reduction factor of downscaling an image (imres * resc) (DO NOT EDIT!)
rescale_amount = 2 ** rescale_factor

# Resolution of image (images will be resized to square [imres * imres]
image_resolution = np.uint(1024 / rescale_amount)

USE_BW = True


def get_setting(setting_id):
    sql = "SELECT Value FROM ***REMOVED***_Phase1.settings WHERE Name='" + setting_id + "';"
    dm.db_cursor.execute(sql)

    setting = dm.db_cursor.fetchone()

    if setting is None:
        print_notice("Setting '%s' does not exist!" % setting_id, mt.MessagePrefix.ERROR)

        exit(-1)
    else:
        return setting[0]


def load_auth():
    for line in open("Settings/ConfigFiles/auth.conf", "r"):
        line = line.strip()

        if not len(line) == 0 and not line.startswith('#'):
            sanitised = line.replace('"', '')
            setting = sanitised.split('=')

            auth.__setitem__(setting[0], setting[1])

    return auth

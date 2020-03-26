import Settings.FileManager as fm
import Settings.SettingsManager as sm
import numpy as np
import matplotlib.pyplot as plt


def save_image(image, root_directory, save_location, filename, digits=None, ind=None, use_current_directory=True):
    if digits is not None and ind is not None:
        preamble_digits = digits - len(str(ind))
        filename += "_" + ('0' * preamble_digits) + str(ind)

    if isinstance(root_directory, fm.SpecialFolder):
        directory = fm.compile_directory(root_directory)
    else:
        directory = root_directory

    if len(save_location) > 0:
        directory += save_location

    if use_current_directory:
        directory += fm.current_directory

    fm.create_if_not_exists(directory)

    file_loc = directory + filename + '.png' # + sm.configuration.get("IO_IMAGE_FILETYPE")

    if not fm.file_exists(file_loc):
        if len(image.shape) != 2:
            image = np.squeeze(image, 2)
        if sm.USE_BW:
            plt.imsave(file_loc, image, cmap='gray')
        else:
            plt.imsave(file_loc, image, cmap='jet')


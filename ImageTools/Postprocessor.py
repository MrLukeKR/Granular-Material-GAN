from scipy import ndimage
import ImageTools.ImageManager as im


from tqdm import tqdm
from sklearn.preprocessing import binarize
from Settings import SettingsManager as sm
from skimage import exposure, restoration


def clean_segments(images, pool):
    fixed_images = list()

    total = len(images)
    curr = 0

    print("\tMorphologically Cleaning Segments... ", end='\r')
    for ind, res in enumerate(pool.map(clean_segment, images)):
        fixed_images.insert(ind, res)
        curr += 1
        print("\tMorphologically Cleaning Segments... " + str(curr / total * 100) + "%", end='\r', flush=True)

        if sm.configuration.get("ENABLE_IMAGE_SAVING") == "True":
            im.save_image(res, str(ind), "Pre-processing/De-Noised/")

    print("\tDe-noising Images... done!")
    return fixed_images


def clean_segment(image):
    # return restoration.denoise_nl_means(image)
    open_image = ndimage.binary_opening(image)
    close_image = ndimage.binary_closing(open_image)

    return close_image


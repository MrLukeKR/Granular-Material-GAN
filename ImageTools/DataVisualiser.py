import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np

from matplotlib import style

style.use('fast')


def generate_animation(images):
    ims = []

    for img in images:
        ims.append([plt.imshow(np.reshape(img, newshape=(1024, 1024)))])
    fig = plt.figure()

    return anim.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)


def save_animation(animation, save_location, frames_per_second):
    animation.save(save_location, fps=frames_per_second, extra_args=['-vcodec', 'libx264'])



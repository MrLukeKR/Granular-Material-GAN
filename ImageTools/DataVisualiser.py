import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np

from matplotlib import style

style.use('fast')

current_axis = None


def generate_animation(images):
    ims = []

    for img in images:
        ims.append([plt.imshow(np.reshape(img, newshape=(1024, 1024)))])
    fig = plt.figure()

    return anim.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)


def save_animation(animation, save_location, frames_per_second):
    animation.save(save_location, fps=frames_per_second, extra_args=['-vcodec', 'libx264'])


def live_graph(title, x_label, y_label):
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    return fig, ax


def set_current_axis(ax):
    global current_axis

    current_axis = ax


def update_live_graph(i, xs, ys):
    global current_axis

    current_axis.clear()
    current_axis.plot(xs, ys)

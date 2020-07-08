from matplotlib import animation

from ImageTools import ImageManager as im

import numpy as np

from Settings.MessageTools import print_notice


def plot_training_data(generator_losses, generator_errors, discriminator_losses, discriminator_accuracies, epochs=None,
                       experiment_id=None, x=None, gen_error_ax=None, dis_error_ax=None, acc_ax=None):
    if gen_error_ax is None or dis_error_ax is None or acc_ax is None:
        fig = im.plt.figure()

        gen_error_ax = fig.add_subplot(3, 1, 1)
        dis_error_ax = fig.add_subplot(3, 1, 2)
        acc_ax = fig.add_subplot(3, 1, 3)
    else:
        if gen_error_ax is not None:
            gen_error_ax.clear()
        if dis_error_ax is not None:
            dis_error_ax.clear()
        if acc_ax is not None:
            acc_ax.clear()

    if x is None:
        x = range(len(generator_losses))

    if experiment_id is not None:
        im.plt.gcf().suptitle("Experiment " + str(experiment_id))

    gen_error_ax.plot(x, generator_losses, '-g', label="Generator Loss")
    gen_error_ax.plot(x, generator_errors, '-b', label="Generator MSE")
    dis_error_ax.plot(x, discriminator_losses, '-r', label="Discriminator Loss")
    acc_ax.plot(x, discriminator_accuracies, '-m', label="Discriminator Accuracy")

    if epochs is not None:
        interval = len(x) // epochs
        gen_error_ylim = gen_error_ax.get_ylim()
        dis_error_ylim = dis_error_ax.get_ylim()
        acc_ylim = acc_ax.get_ylim()

        for epoch in range(epochs):
            loc = interval * epoch

            gen_error_ax.plot((loc, loc), gen_error_ylim, "--", color="k", linewidth=0.1)
            dis_error_ax.plot((loc, loc), dis_error_ylim, "--", color="k", linewidth=0.1)
            acc_ax.plot((loc, loc), acc_ylim, "--", color="k", linewidth=0.1)

    for axis in [gen_error_ax, acc_ax, dis_error_ax]:
        axis.set_xlabel("Epochs")

    for axis in [gen_error_ax, dis_error_ax]:
        axis.set_ylabel("Error")

    acc_ax.set_ylabel("Accuracy")
    gen_error_ax.legend(loc="upper right")
    dis_error_ax.legend(loc="upper right")
    acc_ax.legend(loc="upper right")

    return gen_error_ax, dis_error_ax, acc_ax


def save_training_graphs(d_loss, g_loss, directory, experiment_id, fold, epochs=None, animate=False):
    fig = im.plt.figure()

    gen_error_ax = fig.add_subplot(3, 1, 1)
    dis_error_ax = fig.add_subplot(3, 1, 2)
    acc_ax = fig.add_subplot(3, 1, 3)

    x = range(len(g_loss[0]))

    filepath = directory + '/' + experiment_id + '_Fold-' + str(fold)

    if not animate:
        plot_training_data([x[0] for x in g_loss], [x[1] for x in g_loss],
                           [x[0] for x in d_loss], [x[1] for x in d_loss],
                           epochs=epochs,
                           gen_error_ax=gen_error_ax,
                           dis_error_ax=dis_error_ax, acc_ax=acc_ax)

        im.plt.gcf().savefig(filepath + '.pdf')
    else:
        gen_error_line, = gen_error_ax.plot(x, g_loss[0], '-g', label="Generator Loss", linewidth=1)
        gen_mse_line, = gen_error_ax.plot(x, g_loss[1], '-b', label="Generator MSE", linewidth=1)
        dis_error_line, = dis_error_ax.plot(x, d_loss[0], '-r', label="Discriminator Loss", linewidth=1)
        accuracy_line, = acc_ax.plot(x, d_loss[1], '-m', label="Discriminator Accuracy", linewidth=1)

        im.plt.gcf().suptitle("Experiment " + str(experiment_id))

        if epochs is not None:
            interval = len(x) // epochs
            gen_error_ylim = gen_error_ax.get_ylim()
            dis_error_ylim = dis_error_ax.get_ylim()
            acc_ylim = acc_ax.get_ylim()

            for epoch in range(epochs):
                loc = interval * epoch

                gen_error_ax.plot((loc, loc), gen_error_ylim, '--', color="k", linewidth=0.1)
                dis_error_ax.plot((loc, loc), dis_error_ylim, '--', color="k", linewidth=0.1)
                acc_ax.plot((loc, loc), acc_ylim, '--', color="k", linewidth=0.1)

        for axis in [gen_error_ax, acc_ax, dis_error_ax]:
            axis.set_xlabel("Epochs")

        for axis in [gen_error_ax, dis_error_ax]:
            axis.set_ylabel("Error")

        acc_ax.set_ylabel("Accuracy")
        gen_error_ax.legend(loc="upper right")
        dis_error_ax.legend(loc="upper right")
        acc_ax.legend(loc="upper right")

        lines = list([gen_error_line, gen_mse_line, dis_error_line, accuracy_line])

        y_data = [g_loss[0], g_loss[1], d_loss[0], d_loss[1]]

        def init():
            for line in lines:
                line.set_data(x, [np.nan] * len(x))
            return lines,

        def animate(i):
            for lnum, line in enumerate(lines):
                data = y_data[lnum][:i] + ([np.nan] * (len(x) - i))
                line.set_ydata(data)
            return lines,

        print_notice("Animating training graph... ", end='')
        ani = animation.FuncAnimation(im.plt.gcf(), animate, init_func=init, frames=len(x), interval=1)
        print("done!")

        print_notice("Saving animation... ", end='')
        ani.save(filepath + '.mp4', fps=24, dpi=300)
        print("done!")

    im.plt.close(im.plt.gcf())

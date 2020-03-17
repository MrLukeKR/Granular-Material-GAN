def plot_training_data(gen_error_ax, dis_error_ax, acc_ax,
                       x, generator_losses, generator_errors, discriminator_losses, discriminator_accuracies):
    gen_error_ax.clear()
    dis_error_ax.clear()
    acc_ax.clear()

    gen_error_ax.plot(x, generator_losses, '-g', label="Generator Loss")
    gen_error_ax.plot(x, generator_errors, '-b', label="Generator MSE")

    dis_error_ax.plot(x, discriminator_losses, '-r', label="Discriminator Loss")

    acc_ax.plot(x, discriminator_accuracies, '-m', label="Discriminator Accuracy")

    for axis in [gen_error_ax, acc_ax, dis_error_ax]:
        axis.set_xlabel("Epochs")

    for axis in [gen_error_ax, dis_error_ax]:
        axis.set_ylabel("Error")

    acc_ax.set_ylabel("Accuracy")
    gen_error_ax.legend(loc="upper right")
    dis_error_ax.legend(loc="upper right")
    acc_ax.legend(loc="upper right")

    return gen_error_ax, dis_error_ax, acc_ax

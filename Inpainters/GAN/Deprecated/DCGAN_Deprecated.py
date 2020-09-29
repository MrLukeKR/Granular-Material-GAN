class Network(AbstractGAN.Network):
    @classmethod
    def train_network(cls, epochs, batch_size, features, labels, core_animation_data=None):
        print_notice("Preparing feature/label matrices...", mt.MessagePrefix.INFORMATION)
        if isinstance(features, list):
            features = np.asarray(features)
        if not len(features.shape) == 5:
            if len(features.shape) > 5:
                features = np.squeeze(features)
            elif len(features.shape) == 4:
                features = np.expand_dims(features, 4)

        if isinstance(labels, list):
            labels = np.asarray(labels)

        if not len(labels.shape) == 5:
            labels = np.expand_dims(np.array(labels), 4)
        print_notice("Matrices are now ready for machine learning input", mt.MessagePrefix.SUCCESS)

        Logger.print("Training network with: " + str(epochs) + " EPOCHS, " + str(batch_size) + " BATCH SIZE")

        x = []
        discriminator_losses = []
        discriminator_accuracies = []
        generator_losses = []
        generator_MSEs = []

        fig = plt.figure()

        gen_error_ax = fig.add_subplot(3, 1, 1)
        dis_error_ax = fig.add_subplot(3, 1, 2)
        acc_ax = fig.add_subplot(3, 1, 3)

        if sm.display_available:
            plt.show(block=False)

        def animate(_):
            dv.plot_training_data(generator_losses, generator_MSEs, discriminator_losses, discriminator_accuracies, x=x,
                                  gen_error_ax=gen_error_ax, dis_error_ax=dis_error_ax, acc_ax=acc_ax)

        # One sided label smoothing
        valid = np.full((batch_size, 1), 0.9)
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, len(features), batch_size)

            # This is the binder generated for a given aggregate arrangement
            if mlm.get_available_gpus() == 2:
                with tf.device('gpu:1'):
                    gen_missing = cls.generator.predict(features[idx] * 2.0 - 1.0)
            else:
                gen_missing = cls.generator.predict(features[idx] * 2.0 - 1.0)

            if mlm.get_available_gpus() == 2:
                with tf.device('gpu:0'):
                    # This trains the discriminator on real samples
                    d_loss_real = cls.discriminator.train_on_batch(labels[idx] * 2.0 - 1.0, valid)
                    # This trains the discriminator on fake samples
                    d_loss_fake = cls.discriminator.train_on_batch(gen_missing * 2.0 - 1.0, fake)
            else:
                d_loss_real = cls.discriminator.train_on_batch(labels[idx] * 2.0 - 1.0, valid)
                d_loss_fake = cls.discriminator.train_on_batch(gen_missing, fake)
            d_loss = 0.5 * tf.add(d_loss_real, d_loss_fake)

            if mlm.get_available_gpus() == 2:
                with tf.device('gpu:1'):
                    g_loss = cls.adversarial.train_on_batch(features[idx] * 2.0 - 1.0, [labels[idx] * 2.0 - 1.0, valid])
            else:
                g_loss = cls.adversarial.train_on_batch(features[idx] * 2.0 - 1.0, [labels[idx] * 2.0 - 1.0, valid])

            Logger.print("%d [DIS loss: %f, acc: %.2f%%] [GEN loss: %f, mse: %f]"
                         % (epoch, d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1]))

            discriminator_losses.append(d_loss[0])
            discriminator_accuracies.append(d_loss[1])
            generator_losses.append(g_loss[0])
            generator_MSEs.append(g_loss[1])

            if sm.display_available:
                x.append(len(x) + 1)
                animate(epoch)
                plt.draw()
                plt.pause(0.1)

            sql = "INSERT INTO training (ExperimentID, Fold, Epoch, TrainingSet, DiscriminatorLoss, " \
                  "DiscriminatorAccuracy, GeneratorLoss, GeneratorMSE) " \
                  "VALUES (%s, %s, %s, %s, %s, %s, %s, %s);"

            val = (Logger.experiment_id, Logger.current_fold + 1, epoch + 1, Logger.current_set + 1,
                   float(d_loss[0]), float(d_loss[1]), float(g_loss[0]), float(g_loss[1]))

            db_cursor = dm.get_cursor()

            db_cursor.execute(sql, val)

            if core_animation_data is not None and len(core_animation_data) == 3:
                generated_core = gan_to_core(cls.adversarial, core_animation_data[0], core_animation_data[1], batch_size)

                try:
                    p = Process(target=cls.animate_gan, args=(
                    core_animation_data, generated_core, 0,))  # TODO: Make current batch no enterable here
                    p.start()
                    p.join()
                except MemoryError:
                    print_notice("Ran out of memory when creating mesh!", mt.MessagePrefix.ERROR)
                    h = hpy()
                    print(h.heap())

            # im.save_voxel_image_collection(gen_missing, fm.SpecialFolder.VOXEL_DATA, "figures/postGAN/generated")
            # im.save_voxel_image_collection(labels, fm.SpecialFolder.VOXEL_DATA, "figures/postGAN/expected")

        return (discriminator_losses, discriminator_accuracies), (generator_losses, generator_MSEs)
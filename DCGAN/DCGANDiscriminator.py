class DCGANDiscriminator:
    parallelProcessing = False

    def __init__(self, parallel):
        self.parallelProcessing = parallel
        print("\tInitialising Deep Convolutional Generative Adversarial Network (Discriminator)")

class Network:
    @classmethod
    def create_network(cls, data):
        raise NotImplementedError("create_network not implemented in abstract base class")

    @classmethod
    def train_network(cls):
        raise NotImplementedError("train_network not implemented in abstract base class")

    @classmethod
    def test_network(cls):
        raise NotImplementedError("test_network not implemented in abstract base class")

    @property
    def generator(self):
        return self.generator

    @property
    def discriminator(self):
        return self.discriminator

    @generator.setter
    def generator(self, value):
        self.generator = value

    @discriminator.setter
    def discriminator(self, value):
        self.discriminator = value

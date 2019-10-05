class Network:
    def __init__(self):
        self._generator = None
        self._discriminator = None

    @classmethod
    def create_network(cls, data):
        raise NotImplementedError("create_network not implemented in abstract base class")

    @classmethod
    def train_network(cls, training_set):
        raise NotImplementedError("train_network not implemented in abstract base class")

    @classmethod
    def test_network(cls, testing_set):
        raise NotImplementedError("test_network not implemented in abstract base class")

    @property
    def generator(self):
        return self._generator

    @property
    def discriminator(self):
        return self._discriminator

    @generator.setter
    def generator(self, value):
        self._generator = value

    @discriminator.setter
    def discriminator(self, value):
        self._discriminator = value


class GeneratorNetwork:
    model = None

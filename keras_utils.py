from keras import activations
from keras.layers.core import MaskedLayer

class Sigmoid(MaskedLayer):
    def __init__(self, alpha, **kwargs):
        super(Sigmoid, self).__init__(**kwargs)
        self.alpha = alpha

    def get_output(self, train=False):
        X = self.get_input(train)
        return activations.sigmoid(self.alpha*X)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'alpha': self.alpha}
        base_config = super(Sigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
#import ipdb; ipdb.set_trace()

class Biais(Layer):
    """Custom keras layer adding a scalar bias to each location in the input"""

    def __init__(self, **kwargs):
        super(Biais, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = K.zeros(input_shape[1:])
        self.trainable_weights = [self.kernel]

    def call(self, x, mask=None):
        return x + self.kernel

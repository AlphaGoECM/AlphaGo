from keras.models import load_model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D as Conv2D
from keras.layers.core import Activation, Flatten

from Tools import Biais
from keras.models import load_model

class CNN :


    """model creation"""
    """     Mastering GoG quotes
    The SL policy network p(w)(ajs) alternates between convolutional layers with weights w and rectifier non-linearities.
    A final softmax layer outputs a probability distribution over all legal moves a.
    We trained a 13 layer policy network
    The input to the policy network is a 19  19  48 image stack
    consisting of 48 feature planes. The first hidden layer zero pads the input into a 23  23 image,
    then convolves k filters of kernel size 55 with stride 1 with the input image and applies a rectifier
    nonlinearity. Each of the subsequent hidden layers 2 to 12 zero pads the respective previous hidden
    layer into a 2121 image, then convolves k filters of kernel size 33 with stride 1, again followed
    by a rectifier nonlinearity. The final layer convolves 1 filter of kernel size 1  1 with stride 1, with
    a different bias for each position, and applies a softmax function. The match version of AlphaGo
    used k = 192 filters; Figure 2,b and Extended Data Table 3 additionally show the results of training
    with k = 128; 256; 384 filters."""


    def __init__(self, layers=None, name=None):
        self.layers = []  # Stack of layers.
        self.model = None  # Internal Model instance.
        self.inputs = []  # List of input tensors
        self.outputs = []  # List of length 1: the output tensor (unique).
        self._trainable = True
        self._initial_weights = None

        # Model attributes.
        self._inbound_nodes = []
        self._outbound_nodes = []
        self.built = False


        self.name = name

        # Add to the model any layers passed to the constructor.
        if layers:
            for layer in layers:
                self.add(layer)


    @staticmethod
    def create_CNN(size, layers, features_nb) :


        CNN = Sequential()
        CNN.add (Conv2D(
        filters = 128,
        kernel_size = (5,5),
        input_shape = (size,size,features_nb),
        padding='same',
        kernel_initializer='uniform',
        activation = 'relu'))
        for i in range(layers):
            CNN.add (Conv2D(filters = 128,kernel_size = (3,3),
            activation = 'relu',padding='same',
            kernel_initializer='uniform'))

        CNN.add (Conv2D(
        filters = 1,
        kernel_size = (1,1),
        padding='same',
        kernel_initializer='uniform'))

        CNN.add(Flatten())
        CNN.add(Biais())
        CNN.add(Activation('softmax'))
        
        return CNN

    def load(self,filename):
        CNN=load_model(filename, custom_objects={'Biais': Biais})
        self.model=CNN



    def pred(self,tensor):
        return self.model.predict(tensor)


from keras.models import load_model

from keras.models import Sequential
from keras.layers import Convolution2D as Conv2D
from keras.layers.core import Activation, Flatten

from SupervisedLearning.Biaislayer import Biais



class CNN :
    def save_SLCNN (self, name) :
        self.save(name)
        del self

    def load_SLCNN (file_name):
        model = network_class(46, init_network=False)
        model = load_model(file_name)
        return model

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

    def create_CNN(i) :
        CNN = Sequential()
        CNN.add (Conv2D(
        filters = 128,
        kernel_size = (5,5),
        input_shape = (46,19,19),
        padding='same',
        kernel_initializer='uniform',
        activation = 'relu'))
        for i in range(i):
            CNN.add (Conv2D(filters = 128,kernel_size = (3,3),
            activation = 'relu',padding='same',
            kernel_initializer='uniform'))
            CNN.add (Conv2D(
            filters = 1,
            kernel_size = (1,1),
            padding='same',
            kernel_initializer='uniform'))

            CNN.add(Flatten())

            CNN.add(Activation('softmax'))
            CNN.add(Biais())
            CNN.summary()
            return CNN

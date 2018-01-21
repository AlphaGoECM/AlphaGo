from keras.models import Sequential
from keras.layers import Convolutional2D
from keras.layers.core import Activation, Flatten


from keras.models import load_model

import ipdb; ipdb.set_trace() # debugger

import os
import h5py as h5
import sys
import argparse

class CNN:

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

    def save_SLCNN (self, name) :
        self.save(name)
        del self


    def load_SLCNN (file_name):
        model = network_class(46, init_network=False)
        model = load_model(file_name)
        return model

        """model creation"""

    def create_CNN() :
        CNN = Sequential()

        CNN.add (Convolution2D(
        filters = 128,
        kernel_size = (5,5),
        input_shape = (46,19,19),
        activation = 'relu'))

        for i in range(11):
            CNN.add (Convolution2D(filters = 128,kernel_size = (3,3),activation = 'relu'))
            print (i)

        CNN.add (Convolution2D(
        filters = 1,
        kernel_size = (1,1)))
        CNN.add(Flatten())

        CNN.add(Activation('softmax'))

        return CNN


        """training"""

    def run_training(cmd_line_args=None):

        import argparse
        parser = argparse.ArgumentParser(description='Perform supervised training on a policy network.')

        parser.add_argument("model", help="Path to a hdf5 model file", defaut =None)
        parser.add_argument("train_data", help="A hdf5 file of training data")
        parser.add_argument("out_directory", help="directory where metadata and weights will be saved")

        parser.add_argument("--minibatch", "-B", help="Size of training data minibatches. Default: 16", type=int, default=16)
        parser.add_argument("--epochs", "-E", help="Total number of iterations on the data. Default: 10", type=int, default=10)
        parser.add_argument("--epoch-length", "-l", help="Number of training examples considered 'one epoch'. Default: # training data", type=int, default=None)
        parser.add_argument("--learning-rate", "-r", help="Learning rate - how quickly the model learns at first. Default: .03", type=float, default=.03)
        parser.add_argument("--decay", "-d", help="The rate at which learning decreases. Default: .0001", type=float, default=.0001)
        parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")

        if cmd_line_args is None:
            args = parser.parse_args()
        else:
            args = parser.parse_args(cmd_line_args)


        if args.verbose:
            if os.path.exists(args.out_directory):
                    print("directory %s exists. any previous data will be overwritten" %args.out_directory)
            else:
                    print("starting fresh output directory %s" % args.out_directory)

        # load
        if (args.model == None) :
            model = network_class(46, init_network=False)
            model = create_CNN
            if args.verbose:
                print ("model created")
        else :
            model = load_model(args.model)
            model_features = features.DEFAULT_FEATURES
            if args.verbose:
                print ("model imported")

        dataset = h5.File(args.train_data)


        # ensure output directory is available
        if not os.path.exists(args.out_directory):
            os.makedirs(args.out_directory)



        # training indices are the first consecutive set of shuffled indices, val
        # next, then test gets the remainder
        train_indices = shuffle_indices[0:n_train_data]
        val_indices = shuffle_indices[n_train_data:n_train_data + n_val_data]
        # test_indices = shuffle_indices[n_train_data + n_val_data:]

        # create dataset generators
        train_data_generator = shuffled_hdf5_batch_generator(
            dataset["states"],
            dataset["actions"],
            train_indices,
            args.minibatch)
        val_data_generator = shuffled_hdf5_batch_generator(
            dataset["states"],
            dataset["actions"],
            val_indices,
            args.minibatch)

        sgd = SGD(lr=args.learning_rate, decay=args.decay)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

        samples_per_epoch = args.epoch_length or n_train_data

        if args.verbose:
            print ('model compiled')
            print("STARTING TRAINING")

        model.fit_generator(
            generator=train_data_generator,
            samples_per_epoch=samples_per_epoch,
            nb_epoch=args.epochs,
            callbacks=[checkpointer, meta_writer],
            validation_data=val_data_generator,
            nb_val_samples=n_val_data)


    if __name__ == '__main__':
        run_training()

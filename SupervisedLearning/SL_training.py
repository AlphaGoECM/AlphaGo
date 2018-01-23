from keras.optimizers import SGD

from SupervisedLearning.CNN_SL import CNN


#import ipdb; ipdb.set_trace()

import numpy as np
import os
import h5py as h5
import sys
import argparse


def __init__(self, path):
    self.file = path
    self.metadata = {
        "epochs": [],
        "best_epoch": 0
    }

def one_hot_action(action, size=19):

    categorical = np.zeros((size, size))
    categorical[action] = 1
    return categorical


def shuffled_hdf5_batch_generator(state_dataset, action_dataset, indices, batch_size, transforms=[]):

    state_batch_shape = (batch_size,) + state_dataset.shape[1:]
    game_size = state_batch_shape[-1]
    Xbatch = np.zeros(state_batch_shape)
    Ybatch = np.zeros((batch_size, game_size * game_size))
    batch_idx = 0
    while True:
        for data_idx in indices:
            # choose a random transformation of the data (rotations/reflections of the board)
            transform = np.random.choice(transforms)
            # get state from dataset and transform it.
            # loop comprehension is used so that the transformation acts on the
            # 3rd and 4th dimensions
            state = np.array([transform(plane) for plane in state_dataset[data_idx]])
            # must be cast to a tuple so that it is interpreted as (x,y) not [(x,:), (y,:)]
            action_xy = tuple(action_dataset[data_idx])
            action = transform(one_hot_action(action_xy, game_size))
            Xbatch[batch_idx] = state
            Ybatch[batch_idx] = action.flatten()
            batch_idx += 1
            if batch_idx == batch_size:
                batch_idx = 0
                yield (Xbatch, Ybatch)

        """training"""


BOARD_TRANSFORMATIONS = {
    "noop": lambda feature: feature,
    "rot90": lambda feature: np.rot90(feature, 1),
    "rot180": lambda feature: np.rot90(feature, 2),
    "rot270": lambda feature: np.rot90(feature, 3),
    "fliplr": lambda feature: np.fliplr(feature),
    "flipud": lambda feature: np.flipud(feature),
    "diag1": lambda feature: np.transpose(feature),
    "diag2": lambda feature: np.fliplr(np.rot90(feature, 1))
}


def run_training(cmd_line_args=None):

        import argparse

        parser = argparse.ArgumentParser(description='Perform supervised training on a policy network.')

        #parser.add_argument("model", help="Path to a hdf5 model file", defaut =model)
        parser.add_argument("--train_data","-t", help="A hdf5 file of training data")
        parser.add_argument("--out_directory","-o", help="directory where metadata and weights will be saved")

        parser.add_argument("--layersnb", "-L", help="Total number of intern Conv2D layers, Default: 11 (+2 others)", type=int, default=11)

        parser.add_argument("--minibatch", "-B", help="Size of training data minibatches. Default: 16", type=int, default=16)
        parser.add_argument("--epochs", "-E", help="Total number of iterations on the data. Default: 10", type=int, default=10)
        parser.add_argument("--epoch-length", "-l", help="Number of training examples considered 'one epoch'. Default: # training data", type=int, default=None)
        parser.add_argument("--learning-rate", "-r", help="Learning rate - how quickly the model learns at first. Default: .03", type=float, default=.03)
        parser.add_argument("--decay", "-d", help="The rate at which learning decreases. Default: .0001", type=float, default=.0001)
        parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")


        parser.add_argument("--symmetries", help="Comma-separated list of transforms, subset of noop,rot90,rot180,rot270,fliplr,flipud,diag1,diag2", default='noop,rot90,rot180,rot270,fliplr,flipud,diag1,diag2')  # noqa: E501

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
        #if (args.model == None) :

        model = CNN.create_CNN(args.layersnb )
        if args.verbose:
            print ("model created")
#        else :
#            model = load_model(args.model)
#            model_features = features.DEFAULT_FEATURES
#            if args.verbose:
#                print ("model imported")

        dataset = h5.File(args.train_data)


        # ensure output directory is available
        if not os.path.exists(args.out_directory):
            os.makedirs(args.out_directory)

        n_total_data = len(dataset["states"])
        n_train_data = int(0.9 * n_total_data)
    # Need to make sure training data is divisible by minibatch size or get
    # warning mentioning accuracy from keras
        n_train_data = n_train_data - (n_train_data % args.minibatch)
        n_val_data = n_total_data - n_train_data
# n_test_data = n_total_data - (n_train_data + n_val_data)
        shuffle_indices = np.random.permutation(n_total_data)
        # training indices are the first consecutive set of shuffled indices, val
        # next, then test gets the remainder
        train_indices = shuffle_indices[0:n_train_data]
        val_indices = shuffle_indices[n_train_data:n_train_data + n_val_data]
        # test_indices = shuffle_indices[n_train_data + n_val_data:]
        symmetries = [BOARD_TRANSFORMATIONS[name] for name in args.symmetries.strip().split(",")]

        # create dataset generators
        train_data_generator = shuffled_hdf5_batch_generator(
            dataset["states"],
            dataset["actions"],
            train_indices,
            args.minibatch,
            symmetries)
        val_data_generator = shuffled_hdf5_batch_generator(
            dataset["states"],
            dataset["actions"],
            val_indices,
            args.minibatch,
            symmetries)

        sgd = SGD(lr=args.learning_rate, decay=args.decay)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

        samples_per_epoch = args.epoch_length or n_train_data

        if args.verbose:
            print ('model compiled')
            print("STARTING TRAINING")

        model.fit_generator(generator=train_data_generator,samples_per_epoch=samples_per_epoch,nb_epoch=args.epochs,validation_data=val_data_generator,nb_val_samples=n_val_data)

        print("finished")

if __name__ == '__main__':

            run_training()

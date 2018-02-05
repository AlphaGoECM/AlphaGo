from keras.optimizers import SGD

from CNN_policy import CNN

#export PATH=/users/usrlocal/artieres/Anaconda/bin/
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


def prepare_data(state_dataset, action_dataset, indices, batch_size):

    state_batch_shape = (batch_size,) + state_dataset.shape[1:]
    game_size = state_batch_shape[-1]
    Xbatch = np.zeros(state_batch_shape)
    Ybatch = np.zeros((batch_size, game_size * game_size))
    batch_idx = 0
    for data_idx in  range(batch_size):
        state = np.array([plane for plane in state_dataset[data_idx]])
        action_xy = tuple(action_dataset[data_idx])
        action = one_hot_action(action_xy, game_size)
        Xbatch[batch_idx] = state
        Ybatch[batch_idx] = action.flatten()
        batch_idx += 1

        return (Xbatch, Ybatch)


def shuffled_hdf5_batch_generator(state_dataset, action_dataset, indices, batch_size, transforms=[]):

    state_batch_shape = (batch_size,) + state_dataset.shape[1:]
    game_size = state_batch_shape[-1]
    Xbatch = np.zeros(state_batch_shape)
    Ybatch = np.zeros((batch_size, game_size * game_size))
    batch_idx = 0
    while True:
        for data_idx in indices:
            transform = np.random.choice(transforms)
            state = np.array([transform(plane) for plane in state_dataset[data_idx]])
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

        parser.add_argument("--train_data","-t", help="A hdf5 file of training data")
        parser.add_argument("--out_directory","-o", help="directory where metadata and weights will be saved")
        parser.add_argument("--model", "-mo", help="load a hdf5 model file", default=None)
        parser.add_argument("--layers_nb", "-L", help="Total number of intern Conv2D layers, Default: 11 (+2 others)", type=int, default=11)
        parser.add_argument("--board_size", "-s", help="Size of the go board", type=int, default=19)

        parser.add_argument("--batch", "-B", help="Size of training data batches. Default: 16", type=int, default=16)
        parser.add_argument("--epochs", "-E", help="Total number of iterations on the data. Default: 4", type=int, default=4)
        parser.add_argument("--epoch-length", "-l", help="Number of training examples considered 'one epoch'. Default: # training data", type=int, default=None)
        parser.add_argument("--learning-rate", "-r", help="Learning rate - how quickly the model learns at first. Default: .03", type=float, default=.03)
        parser.add_argument("--decay", "-d", help="The rate at which learning decreases. Default: .0001", type=float, default=.0001)

        parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")
        parser.add_argument("--generator", "-gen", help="Turn on generator data mode", default=False, action="store_true")


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

        dataset = h5.File(args.train_data)
        if args.verbose:
            print ("DATA LOADED")

        dataset_features = dataset['features'][()]
        dataset_features = dataset_features.split(",") # a terme, cela permettra de verifier quelle feature est utilisee pour le modele et si cela match avec les donnees
        features_nb = dataset['features_nb'][()]

        if args.verbose :
            print ("%s FEATURES LOADED" %features_nb)

        # load
        if args.model:
            if not os.path.exists(os.path.join(args.out_directory, args.model)):
                raise ValueError("Cannot resume without existing output directory")

            if not os.path.exists(args.out_directory):
                if args.verbose:
                    print("creating output directory {}".format(args.out_directory))
                os.makedirs(args.out_directory)

            if os.path.exists(os.path.join(args.out_directory, args.model)):
                model = load_model(args.model)
                model_features = features.DEFAULT_FEATURES
                if args.verbose:
                    print ("model loaded")

        else:
            model = CNN.create_CNN(args.board_size,args.layers_nb,features_nb)
            if args.verbose:
                print ("MODEL CREATED")
                model.summary()

        # ensure output directory is available
        if not os.path.exists(args.out_directory):
            os.makedirs(args.out_directory)

        n_total_data = len(dataset["states"])
        n_train_data = int(0.9 * n_total_data)

        sgd = SGD(lr=args.learning_rate, decay=args.decay)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
        verbose = 0
        if args.verbose:
            verbose = 2
            print ('MODEL COMPILED')

        if (args.generator): # A REVOIR
            if args.verbose:
                print ('fit generator mode actived')

            n_train_data = n_train_data - (n_train_data % args.batch) # Need to have data divisible by batch size
            n_val_data = n_total_data - n_train_data

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
                args.batch,
                symmetries)
            val_data_generator = shuffled_hdf5_batch_generator(
                dataset["states"],
                dataset["actions"],
                val_indices,
                args.batch,
                symmetries)

            samples_per_epoch = args.epoch_length or n_train_data

            if args.verbose:
                print("STARTING TRAINING - SL gen")

            model.fit_generator(generator=train_data_generator,
            samples_per_epoch=samples_per_epoch,
            nb_epoch=args.epochs,
            validation_data=val_data_generator,
            nb_val_samples=n_val_data)

            model.save("weights_gen.00000.hdf5")

        else :
            symmetries = [BOARD_TRANSFORMATIONS[name] for name in args.symmetries.strip().split(",")]
            shuffle_indices = np.random.permutation(n_total_data)
            n_train_data = n_train_data - (n_train_data % args.batch)
            train_indices = shuffle_indices[0:n_train_data]

            x,y = prepare_data(dataset["states"], dataset["actions"], train_indices,len(dataset["states"]))

            if args.verbose :
                print("STARTING TRAINING - SL")
                print "Entry dataset shape ",x.shape
                print "Exit dataset shape ",y.shape



            model.fit(x=x, y=y,
            batch_size=args.batch,
            epochs=args.epochs,
            verbose=verbose)

            model.save("weights.00001.hdf5")


        if args.verbose:
            print("TRAINING FINISHED")


if __name__ == '__main__':

            run_training()

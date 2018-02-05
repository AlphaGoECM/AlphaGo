from keras.optimizers import SGD
from keras.models import load_model
from Biaislayer import Biais

#export PATH=/users/usrlocal/artieres/Anaconda/bin/
#import ipdb; ipdb.set_trace()

import numpy as np
import os
import h5py as h5
import sys
import argparse

def one_hot_action(action, size=19):

    categorical = np.zeros((size, size))
    categorical[action] = 1
    return categorical



def prepare_data(state_dataset, action_dataset, indices):

    batch_size =  len(state_dataset)
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

def run_eval(cmd_line_args=None):

    import argparse

    parser = argparse.ArgumentParser(description='Perform supervised training on a policy network.')
    parser.add_argument("--eval_data","-e", help="A hdf5 file of evaluating data")
    parser.add_argument("--model", "-mo", help="load a hdf5 model file", default=None)
    parser.add_argument("--verbose", "-v", help="turn on verbose mode", default=False, action="store_true")

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    if os.path.exists(args.model):
        model = load_model(args.model,custom_objects={"Biais": Biais()})
        if args.verbose:
            print ("MODEL LOADED")
    else :
        raise ValueError("Cannot resume without existing model")

    dataset = h5.File(args.eval_data)
    if args.verbose:
        print ("DATA LOADED")

    n_total_data = len(dataset["states"])
    shuffle_indices = np.random.permutation(n_total_data)
    train_indices = shuffle_indices[0:n_total_data]
    x,y = prepare_data(dataset["states"], dataset["actions"], train_indices)
    score, accuracy = model.evaluate(x,y)
    print (score,accuracy)


if __name__ == '__main__':

        run_eval()

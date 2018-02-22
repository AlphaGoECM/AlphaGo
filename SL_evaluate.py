from keras.optimizers import SGD
from keras.models import load_model
from Tools import Biais, Tools

#export PATH=/users/usrlocal/artieres/Anaconda/bin/
#import ipdb; ipdb.set_trace()

import numpy as np
import os
import h5py as h5
import sys
import argparse


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
    x,y = Tools.prepare_data(dataset["states"], dataset["actions"], train_indices)
    score, accuracy = model.evaluate(x,y)
    print (score,accuracy)


if __name__ == '__main__':

        run_eval()

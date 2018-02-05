from keras.optimizers import SGD

from CNN_policy import CNN

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



    parser.add_argument("--out_directory","-o", help="directory where metadata and weights will be saved")
    parser.add_argument("--model", "-mo", help="load a hdf5 model file", default=None)


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
            print ("MODEL LOADED")


    score, accuracy = model.evaluate
    print (score,accuracy)


if __name__ == '__main__':

        run_eval()

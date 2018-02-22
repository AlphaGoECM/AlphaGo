#Ne pas utiliser toute la mémoire du gpu par défaut (alloc dynamique)
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from CNN_policy import CNN
from Tools import  Tools
import datetime


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

        parser.add_argument("--train_data","-t", help="A hdf5 file of training data MANDATORY")
        parser.add_argument("--proportion", "-p", help=" %\of data used in the data file. Default: 100%", type=int, default=100)
        parser.add_argument("--out_directory","-o", help="directory where metadata and weights will be saved MANDATORY")

        parser.add_argument("--model", "-mo", help="load a hdf5 model file. You have to give the file path. Do not use if you want the model to be created", default=None)
        parser.add_argument("--layers_nb", "-L", help="total number of intern Conv2D layers, Default: 11 (+2 others)", type=int, default=11)
        parser.add_argument("--board_size", "-s", help="size of the go board Default : 19", type=int, default=19)

        parser.add_argument("--batch", "-B", help="size of training data batches. Default: 16", type=int, default=16)
        parser.add_argument("--epochs", "-E", help="total number of iterations on the data. Default: 4", type=int, default=4)
        parser.add_argument("--epoch-length", "-l", help="number of training examples considered 'one epoch'. Default: #training data", type=int, default=None)
        parser.add_argument("--learning-rate", "-r", help="learning rate - how quickly the model learns at first. Default: .03", type=float, default=.03)
        parser.add_argument("--decay", "-d", help="the rate at which learning decreases. Default: .0001", type=float, default=.0001)

        parser.add_argument("--verbose", "-v", help="turn on verbose mode", default=False, action="store_true")



        symmetries = 'noop,rot90,rot180,rot270,fliplr,flipud,diag1,diag2'

        date = datetime.datetime.now()

        if cmd_line_args is None:
            args = parser.parse_args()
        else:
            args = parser.parse_args(cmd_line_args)

        verbose = 0
        verbose_save = 0
        if args.verbose:
            verbose_save =1
            verbose = 2

        # ensure output directory is available

        out_dir = args.out_directory +'/model_'+ str(date)

        if os.path.exists(out_dir):
            if args.verbose:
                print("directory %s exists. any previous data will be overwritten" %out_dir)
        else:
            if args.verbose:
                print("creating output file {}".format(out_dir))
            os.makedirs(out_dir)

        # load dataset
        dataset = h5.File(args.train_data)
        if args.verbose:
            print ("DATA LOADED")

        # update features for model
        dataset_features = dataset['features'][()]
        dataset_features = dataset_features.split(",") # a terme, cela permettra de verifier quelle feature est utilisee pour le modele et si cela match avec les donnees
        features_nb = dataset['features_nb'][()]
        if args.verbose :
            print ("%s FEATURES LOADED" %features_nb)
            print (dataset_features)


        # load
        if args.model:
            if not os.path.exists(args.model):
                raise ValueError("Cannot resume without existing model")
            else :
                model = load_model(args.model,custom_objects={"Biais": Biais()})
                if args.verbose:
                    print ("MODEL LOADED")

        else:
            model = CNN.create_CNN(args.board_size,args.layers_nb,features_nb)
            if args.verbose:
                print ("MODEL CREATED")
                model.summary()


        n_total_data = len(dataset["states"])*args.proportion/100
        n_train_data = int(0.9 * n_total_data)

        if args.verbose:
            print ("Using %s states for the training in %s "%( n_train_data, len(dataset["states"])) )

        n_train_data = n_train_data - (n_train_data % args.batch) # Need to have data divisible by batch size
        n_val_data = n_total_data - n_train_data
        shuffle_indices = np.random.permutation(n_total_data)

        train_indices = shuffle_indices[0:n_train_data]
        val_indices = shuffle_indices[n_train_data:n_train_data + n_val_data]

        #callbacks

        filepath = ("%s/model_temp.{epoch:02d}.hdf5" %(out_dir,))
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=verbose_save, save_best_only=True, mode='max')
        callbacks = [checkpoint]

        # compilation
        sgd = SGD(lr=args.learning_rate, decay=args.decay)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
        if args.verbose:
            print ('MODEL COMPILED')



        if args.verbose:
            print (str(date))


        #if (not(args.nogenerator)):


        symmetries = [BOARD_TRANSFORMATIONS[name] for name in symmetries.strip().split(",")]

            #dataset generators
        train_data_generator = Tools.batch_generator( dataset["states"],
        dataset["actions"], train_indices, args.batch, symmetries)
        val_data_generator = Tools.batch_generator( dataset["states"],
        dataset["actions"], val_indices, args.batch, symmetries)

        samples_per_epoch = args.epoch_length or n_train_data

        if args.verbose:
            print("STARTING TRAINING - SL IN GENERATOR MODE")

        model.fit_generator(generator=train_data_generator,
            samples_per_epoch=samples_per_epoch,
            nb_epoch=args.epochs,
            validation_data=val_data_generator,
            nb_val_samples=n_val_data,callbacks=callbacks)

        date = datetime.datetime.now()
        filepath = ("%s/model_%s_%s_%sh%s.hdf5" %(out_dir,date.day,date.month,date.hour, date.minute))
        tfilepath = ("%s/model_%s_%s_%sh%s.txt" %(out_dir,date.day,date.month,date.hour, date.minute))
        '''
        PREVIOUS VERSION WITHOUT DATA GENERATOR
        else :
            x,y = prepare_data(dataset["states"], dataset["actions"],n_train_data)

            if args.verbose :
                print("STARTING TRAINING - SL")
                print "Entry dataset shape ",x.shape
                print "Exit dataset shape ",y.shape


            model.fit(x=x, y=y, # history = model.(fit) => history.history gives the history :)
            batch_size=args.batch,
            epochs=args.epochs,
            verbose=verbose)
            date = datetime.datetime.now()
            name = ("%s/model_%s_%s_%sh%s.hdf5" %(args.out_directory,date.day,date.month,date.hour,date.minute))
        '''


        if args.verbose:
            print("TRAINING FINISHED")
            print (str(date))

        model.save(filepath)
        Tools.text_file(tfilepath,model, n_total_data, args.epochs, date)

        if args.verbose:
            print("SAVE DONE IN %s" % out_dir)

if __name__ == '__main__':

            run_training()

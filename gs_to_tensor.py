#!/usr/bin/env python

import numpy as np
from features import Preprocess
from sgf_to_gs import sgf_iter_states
import go
import os
import warnings
import sgf
import h5py as h5
import sys
import argparse

class SizeMismatchError(Exception):
    pass

class GameConverter:

    def __init__(self, features):
        self.feature_processor = Preprocess(features)
        self.n_features = self.feature_processor.output_dim

    def convert_game(self, file_name, bd_size):
        """Read the given SGF file into an iterable of (input, output) pairs
        for neural network training.
        Each input is a GameState converted into one-hot neural net features.
        Each output is an action as an (x,y) pair (passes are skipped)
        If this game's size does not match bd_size, a SizeMismatchError is raised
        """
        with open(file_name, 'r') as file_object:
            state_action_iterator = sgf_iter_states(file_object.read(), include_end=False)

        for (state, move, player) in state_action_iterator:
            if state.size != bd_size:
                raise SizeMismatchError()
            if move != go.PASS_MOVE:
                nn_input = self.feature_processor.state_to_tensor(state)
                yield (nn_input, move)


    def sgfs_to_hdf5(self, sgf_files, hdf5_file, bd_size=19, ignore_errors=True):
        """Convert all files in the iterable sgf_files into an hdf5 group to be stored in hdf5_file
        Arguments:
        - sgf_files : an iterable of relative or absolute paths to SGF files
        - hdf5_file : the name of the HDF5 where features will be saved
        - bd_size : side length of board of games that are loaded
        - ignore_errors : if True, issues a Warning when there is an unknown
            exception rather than halting. Note that sgf.ParseException and
            go.IllegalMove exceptions are always skipped
        The resulting file has the following properties:
            states  : dataset with shape (n_data, n_features, board width, board height)
            actions : dataset with shape (n_data, 2) (actions are stored as x,y tuples of
                      where the move was played)
            file_offsets : group mapping from filenames to tuples of (index, length)
        For example, to find what positions in the dataset come from 'test.sgf':
            index, length = file_offsets['test.sgf']
            test_states = states[index:index+length]
            test_actions = actions[index:index+length]
        """

        # Make a hidden temporary file in case of a crash.
        # If success, this is renamed to hdf5_file
        tmp_file = os.path.join(os.path.dirname(hdf5_file), ".tmp." + os.path.basename(hdf5_file))
        h5f = h5.File(tmp_file, 'w')

        try:
            states = h5f.require_dataset(
                'states',
                dtype=np.uint8,
                shape=(1, self.n_features, bd_size, bd_size),
                maxshape=(None, self.n_features, bd_size, bd_size),  # 'None' == arbitrary size
                exact=False,  # allow non-uint8 datasets to be loaded, coerced to uint8
                chunks=(64, self.n_features, bd_size, bd_size),  # approximately 1MB chunks
                compression="lzf")
            actions = h5f.require_dataset(
                'actions',
                dtype=np.uint8,
                shape=(1, 2),
                maxshape=(None, 2),
                exact=False,
                chunks=(1024, 2),
                compression="lzf")

            # Store comma-separated list of feature planes in the scalar field 'features'. The
            # String can be retrieved using h5py's scalar indexing: h5f['features'][()]
            h5f['features'] = np.string_(','.join(self.feature_processor.feature_list))

            next_idx = 0
            for file_name in sgf_files:
                # count number of state/action pairs yielded by this game
                n_pairs = 0
                file_start_idx = next_idx
                for state, move in self.convert_game(file_name, bd_size):
                    if next_idx >= len(states):
                        states.resize((next_idx + 1, self.n_features, bd_size, bd_size))
                        actions.resize((next_idx + 1, 2))
                    states[next_idx] = state
                    actions[next_idx] = move
                    n_pairs += 1
                    next_idx += 1

        except go.IllegalMove:
            warnings.warn("Illegal Move encountered in %s\n"
                                  "dropping the remainder of the game" % file_name)
        except sgf.ParseException:
            warnings.warn("Could not parse %s\n\tdropping game" % file_name)

        except SizeMismatchError:
            warnings.warn("Skipping %s; wrong board size" % file_name)

        except Exception as e:
                    # catch everything else
            if ignore_errors:
                warnings.warn("Unkown exception with file %s\n\t%s" % (file_name, e),
                                stacklevel=2)
            else:
                raise e

        except Exception as e:
            print("sgfs_to_hdf5 failed")
            os.remove(tmp_file)
            raise e

        h5f.close()
        os.rename(tmp_file, hdf5_file)


def run_game_converter(cmd_line_args=None):
    """Run conversions. command-line args may be passed in as a list
    """
    parser = argparse.ArgumentParser(
    description='Prepare SGF Go game files for training the neural network model.',
    epilog="Available features are: stone_color_feature, ones, turns_since_move, liberties,\
    capture_size, atari_size, liberties_after, sensibleness, and zeros.\
    Ladder features are not currently implemented")
    parser.add_argument("--features", "-f", help="Comma-separated list of features to compute and store or 'all'", default='all')  # noqa: E501
    parser.add_argument("--outfile", "-o", help="Destination to write data (hdf5 file)", required=True)  # noqa: E501
    parser.add_argument("--recurse", "-R", help="Set to recurse through directories searching for SGF files", default=False, action="store_true")  # noqa: E501
    parser.add_argument("--directory", "-d", help="Directory containing SGF files to process. if not present, expects files from stdin", default=None)  # noqa: E501
    parser.add_argument("--size", "-s", help="Size of the game board. SGFs not matching this are discarded with a warning", type=int, default=19)  # noqa: E501
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    if args.features.lower() == 'all':
        feature_list = [
            "stone_color_feature",
            "ones",
            "turns_since_move",
            "liberties",
            "capture_size",
            "atari_size",
            "sensibleness",
            "zeros"]
    else:
        feature_list = args.features.split(",")

    converter = GameConverter(feature_list)

    def _is_sgf(fname):
        return fname.strip()[-4:] == ".sgf"

    def _walk_all_sgfs(root):
        """a helper function/generator to get all SGF files in subdirectories of root
        """
        for (dirpath, dirname, files) in os.walk(root):
            for filename in files:
                if _is_sgf(filename):
                    # yield the full (relative) path to the file
                    yield os.path.join(dirpath, filename)

    def _list_sgfs(path):
        """helper function to get all SGF files in a directory (does not recurse)
        """
        files = os.listdir(path)
        return (os.path.join(path, f) for f in files if _is_sgf(f))
    # get an iterator of SGF files according to command line args
    if args.directory:
        if args.recurse:
            files = _walk_all_sgfs(args.directory)
        else:
            files = _list_sgfs(args.directory)
    else:
        files = (f.strip() for f in sys.stdin if _is_sgf(f))

    converter.sgfs_to_hdf5(files, args.outfile, bd_size=args.size)

if __name__ == '__main__':
    run_game_converter()

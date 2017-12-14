#!/usr/bin/env python

import numpy as np
import go

maximum = 8

"""
Features used by AlphaGo, in approximate order of importance.
Feature                 # Notes
Stone colour            3 Player stones; oppo. stones; empty
Ones                    1 Constant plane of 1s
    (Because of convolution w/ zero-padding, this is the only way the NN
     can know where the edge of the board is !!!)
Turns since last move   8 How many turns since a move played
Liberties               8 Number of liberties
Capture size            8 How many opponent stones would be captured
Self-atari size         8 How many own stones would be captured
Liberties after move    8 Number of liberties after this move played
Ladder capture          1 Whether a move is a successful ladder cap
Ladder escape           1 Whether a move is a successful ladder escape
Sensibleness            1 Whether a move is legal + doesn't fill own eye
Zeros                   1 Constant plane of 0s
"""

def stone_color_feature(state):
    planes = np.zeros((3, state.size, state.size))
    for x in range(state.size):
        for y in range(state.size):
            # First layer is current player, second is opponent, third is empty
            planes[0, x, y] = state.board[x][y] == state.current_player
            planes[1, x, y] = state.board[x][y] == -state.current_player
            planes[2, x, y] = state.board[x][y] == go.EMPTY
    return planes

def ones(state):
    return np.ones((1, state.size, state.size))

# for all the following 8-layers features, the [maximum-1] layer contains any
# stone which has the property >=8

def turns_since_move(state, maximum=8):
    planes = np.zeros((maximum, state.size, state.size))
    for i in range(maximum):
        for x in range(state.size):
            for y in range(state.size):
                planes[i, x, y] = state.stone_ages[x][y] == i
                if state.stone_ages[x][y] >= maximum:
                    planes[maximum-1, x, y] = 1
    return planes

def get_liberties(state, maximum=8):
    planes = np.zeros((maximum, state.size, state.size))
    for i in range(maximum):
        for x in range(state.size):
            for y in range(state.size):
                planes[i, x, y] = state.liberty_counts[x,y] == i
                if state.liberty_counts[x,y] >= maximum:
                    planes[maximum-1, x, y] = 1
    return planes

def get_capture_size(state, maximum=8):
    """A feature encoding the number of opponent stones that would be captured by
    playing at each location, up to 'maximum'
    """
    planes = np.zeros((maximum, state.size, state.size))
    for (x, y) in state.get_legal_moves():
        # multiple disconnected groups may be captured, hence we loop over
        # groups and count sizes if captured.
        n_captured = 0
        for neighbor_group in state.get_groups_around((x,y)):
            # if the neighboring group is opponent stones and they have
            # one liberty, it is (x,y) so we are capturing them
            (gx, gy) = next(iter(neighbor_group))
            if (state.liberty_counts[gx][gy] == 1) and (state.board[gx][gy] != state.current_player):
                n_captured += len(state.group_sets[gx][gy])
        planes[min(n_captured, maximum - 1), x, y] = 1
    return planes

def get_atari_size(state, maximum=8):
    """Atari is when a play leads to a situation where the number of liberties
    of a stone/a group of stones is only one. This feature encods the number of
    stones is the atari group, up to 'maximum'
    Logic is the following :
    We play a move ; then we see what happens and actualize situation ;
    then we check if the group of stones our last move is rattached on has
    1 liberty.
    """
    planes = np.zeros((maximum, state.size, state.size))
    for (x, y) in state.get_legal_moves():
        liberty_set_after = set(state.liberty_sets[x][y])
        group_set_after = set()
        group_set_after.add((x,y))
        captured_stones = set()
        for neighbor_group in state.get_groups_around((x,y)):
            # if the neighboring group is of the same color as the current player
            # then playing here will connect this stone to that group
            (gx, gy) = next(iter(neighbor_group))
            if state.board[gx, gy] == state.current_player:
                liberty_set_after |= state.liberty_sets[gx][gy]
                group_set_after |= state.group_sets[gx][gy]
            # if instead neighboring group is opponent *and about to be captured*
            # then we might gain new liberties
            elif state.liberty_counts[gx][gy] == 1:
                captured_stones |= state.group_sets[gx][gy]
        # add captured stones to liberties if they are neighboring the 'group_set_after'
        # i.e. if they will become liberties once capture is resolved
        if len(captured_stones) > 0:
            for (gx, gy) in group_set_after:
                # intersection of group's neighbors and captured stones will become liberties
                liberty_set_after |= set(state._neighbors((gx, gy))) & captured_stones
        if (x, y) in liberty_set_after:
            liberty_set_after.remove((x, y))

        # check if this move resulted in atari
        if len(liberty_set_after) == 1:
            group_size = len(group_set_after)
            # as always, 0th plane used for size=1, so group_size-1 is the index
            planes[min(group_size - 1, maximum - 1), x, y] = 1
    return planes


def get_liberties_after(state, maximum=8):
    """A feature encoding what the number of liberties *would be* of the group connected to
    the stone *if* played at a location
    Logic is exactly the same as previously : we check what consequences would be,
    we actualize the board and then we count liberties, up to maximum-1.
    """
    planes = np.zeros((maximum, state.size, state.size))
    for (x, y) in state.get_legal_moves():
        liberty_set_after = set(state.liberty_sets[x][y])
        group_set_after = set()
        group_set_after.add((x, y))
        captured_stones = set()
        for neighbor_group in state.get_groups_around((x, y)):
            # if the neighboring group is of the same color as the current player
            # then playing here will connect this stone to that group and
            # therefore add in all that group's liberties
            (gx, gy) = next(iter(neighbor_group))
            if state.board[gx, gy] == state.current_player:
                liberty_set_after |= state.liberty_sets[gx][gy]
                group_set_after |= state.group_sets[gx][gy]
            # if instead neighboring group is opponent *and about to be captured*
            # then we might gain new liberties
            elif state.liberty_counts[gx][gy] == 1:
                captured_stones |= state.group_sets[gx][gy]
        # add captured stones to liberties if they are neighboring the 'group_set_after'
        # i.e. if they will become liberties once capture is resolved
        if len(captured_stones) > 0:
            for (gx, gy) in group_set_after:
                # intersection of group's neighbors and captured stones will become liberties
                liberty_set_after |= set(state._neighbors((gx, gy))) & captured_stones
        # (x,y) itself may have made its way back in, but shouldn't count
        # since it's clearly not a liberty after playing there
        if (x, y) in liberty_set_after:
            liberty_set_after.remove((x, y))
        planes[min(maximum - 1, len(lib_set_after) - 1), x, y] = 1
    return planes

def sensibleness(state):
    plane = np.zeros((1, state.size, state.size))
    for x in range(state.size):
        for y in range(state.size):
            if (x, y) in state.get_legal_moves(include_eyes=False):
                plane[0, x, y] += 1
    return plane

def zeros(state):
    return np.zeros((1, state.size, state.size))

# For now, doesn't include ladders
FEATURES = {
    "stone_color_feature": {
        "size": 3,
        "function": stone_color_feature
    },
    "ones": {
        "size": 1,
        "function": ones
    },
    "turns_since_move": {
        "size": 8,
        "function": turns_since_move
    },
    "liberties": {
        "size": 8,
        "function": get_liberties
    },
    "capture_size": {
        "size": 8,
        "function": get_capture_size
    },
    "atari_size": {
        "size": 8,
        "function": get_atari_size
    },
    "liberties_after": {
        "size": 8,
        "function": get_liberties_after
    },
    "sensibleness": {
        "size": 1,
        "function": sensibleness
    },
    "zeros": {
        "size": 1,
        "function": zeros
    },
}

DEFAULT_FEATURES = ["stone_color_feature", "ones", "turns_since_move", "liberties", "capture_size",
                    "atari_size", "liberties_after", "sensibleness", "zeros"]

class Preprocess(object):
    """Now we have all the features, we have to construct a final tensor
    in order to feed the NN later. All the features will be one-hot encoded"""

    def __init__(self, feature_list=DEFAULT_FEATURES):
        """Creates a preprocessor object that will concatenate together the
        given list of features"""
        self.output_dim = 0
        self.feature_list = feature_list
        # processors are lists of functions
        # output_dim is the dimension of the layers (about ~40)
        self.processors = [None] * len(feature_list)
        for i in range(len(feature_list)):
            feat = feature_list[i].lower()
            if feat in FEATURES:
                self.processors[i] = FEATURES[feat]["function"]
                self.output_dim += FEATURES[feat]["size"]
            else:
                raise ValueError("uknown feature: %s" % feat)

    def state_to_tensor(self, state):
        """Convert a GameState to a Theano-compatible tensor
        """
        feat_tensors = [app(state) for app in self.processors]
        # concatenate along feature dimension then add in a singleton 'batch' dimension
        f, s = self.output_dim, state.size
        return np.concatenate(feat_tensors).reshape((1, f, s, s))

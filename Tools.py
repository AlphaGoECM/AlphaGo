from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
#import ipdb; ipdb.set_trace()

class Biais(Layer):
    """Custom keras layer adding a scalar bias to each location in the input"""

    def __init__(self, **kwargs):
        super(Biais, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = K.zeros(input_shape[1:])
        self.trainable_weights = [self.kernel]

    def call(self, x, mask=None):
        return x + self.kernel


class Tools :

    @staticmethod
    def one_hot_action(action, size=19):

        categorical = np.zeros((size, size))
        categorical[action] = 1
        return categorical

    @staticmethod
    def prepare_data(state_dataset, action_dataset):

        batch_size =  len(state_dataset)
        state_batch_shape = (batch_size,) + state_dataset.shape[1:]
        game_size = state_batch_shape[-1]
        Xbatch = np.zeros(state_batch_shape)
        Ybatch = np.zeros((batch_size, game_size * game_size))
        batch_idx = 0
        for data_idx in  range(batch_size):
            state = np.array([plane for plane in state_dataset[data_idx]])
            action_xy = tuple(action_dataset[data_idx])
            action = Tools.one_hot_action(action_xy, game_size)

            Xbatch[batch_idx] = state
            Ybatch[batch_idx] = action.flatten()
            batch_idx += 1

            return (Xbatch, Ybatch)

    @staticmethod
    def batch_generator(state_dataset, action_dataset, indices, batch_size, transforms=[]):

        if state_dataset.shape[1] == 38:
            state_dataset=np.swapaxes(state_dataset,1,3)
            state_dataset=np.swapaxes(state_dataset,1,2)

        state_batch_shape = (batch_size,) + state_dataset.shape[1:]
        game_size = state_dataset.shape[1]
        Xbatch = np.zeros(state_batch_shape)
        Ybatch = np.zeros((batch_size, game_size * game_size))
        batch_idx = 0
        while True:
            for data_idx in indices:
                transform = np.random.choice(transforms)
                state = np.array([plane for plane in state_dataset[data_idx]])
                action_xy = tuple(action_dataset[data_idx])
                action = Tools.one_hot_action(action_xy, game_size)
                Xbatch[batch_idx] = state
                Ybatch[batch_idx] = action.flatten()
                batch_idx += 1
                if batch_idx == batch_size:
                    batch_idx = 0
                    yield (Xbatch, Ybatch)

    @staticmethod
    def text_file (filepath,model, nb_data, epochs, date):

        f = open(filepath,'w')
        f.write(str(date)+'\n')
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write('sur '+str(nb_data)+' coups et ' + str(epochs) +' epochs' )
        f.close()

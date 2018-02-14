import CNN_policy
import numpy as np
import go

def flatten_idx(position, size):
    (x, y) = position
    return x * size + y

class Player_rd(object): #joue au hasard

    def __init__(self):
        return
    
    def get_move(self, state):
        # list with sensible moves
        sensible_moves = [move for move in state.get_legal_moves(include_eyes=False)]
        
      
        
        #parcourt les coups possible et ressort celui qu'aurai jouer le policy network    
        if len(sensible_moves) > 0:
            a=np.random.randint(0,len(sensible_moves),1) #on prend un coup au hasard
            return(sensible_moves[a[0]])
        

        # No 'sensible' moves available, so do pass move
        return go.PASS_MOVE

class Player_pl(object): # joue coup le plus probable donnée par le policy

    def __init__(self, policy_function,convertor):
        self.policy = policy_function
        self.convertor=convertor

       
    def eval_state(self, state, moves=None): # evalue le coup joué par le policy network dans une liste de coup donnée

        tensor = self.convertor.state_to_tensor(state)
        # run the tensor through the network
        network_output = self.policy.forward(tensor)  # A AJOUTER
        moves = moves or state.get_legal_moves()


        if len(moves) == 0:
            return []
        move_indices = [flatten_idx(m, state.size) for m in moves] #A VERIFIER
        # get network activations at legal move locations
        distribution = network_output[0][move_indices]
        distribution = distribution / distribution.sum()
        return zip(moves, distribution)
    
       # return self._select_moves_and_normalize(network_output[0], moves, state.size)
        
    def get_move(self, state):
        # list with sensible moves
        sensible_moves = [move for move in state.get_legal_moves(include_eyes=False)]
             
        
        #parcourt les coups possible et ressort celui qu'aurai jouer le policy network    
        if len(sensible_moves) > 0:

            move_probs = self.eval_state(state, sensible_moves)
            max_prob = max(move_probs, key=itemgetter(1))
            return max_prob[0]
       
        # No 'sensible' moves available, so do pass move
        return go.PASS_MOVE
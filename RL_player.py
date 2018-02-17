import CNN_policy
import numpy as np
import go

def conv_idx(position, size):   # convertit indice matrice [x][y] en indice liste[x*size+y]
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

       
    def eval_state(self, state, moves=None): # renvoie la probabilité de jouer chacun des coups dans moves en fonction de state par le policy

        if len(moves) == 0:  # pas de coup possible
            return []
        tensor = self.convertor.state_to_tensor(state) # convertit l'état du jeu en entré pour le CNN

        network_output = self.policy.predict(tensor)  # A FAIRE
  
        move_indices = [conv_idx(m, state.size) for m in moves] 
        
        # A faire : 
        
        # get network activations at legal move locations
        distribution = network_output[0][move_indices] #A VERIFIER
        distribution = distribution / distribution.sum()
        return zip(moves, distribution)
    
       # return self._select_moves_and_normalize(network_output[0], moves, state.size)
    
    
    
    def get_move(self, state):
        # list with sensible moves
        sensible_moves = [move for move in state.get_legal_moves(include_eyes=False)]
             
        
        #parcourt les coups possible et ressort celui joué le policy network    
        if len(sensible_moves) > 0:

            move_probs = self.eval_state(state, sensible_moves)
            max_prob = max(move_probs, key=itemgetter(1))
            return max_prob[0]
       
        # No 'sensible' moves available, so do pass move
        return go.PASS_MOVE
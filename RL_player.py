import CNN_policy
import numpy as np
import go


def conv_mat(position, size):   # convertit indice matrice [x][y] en indice liste[x*size+y]
    (x, y) = position
    return x * size + y

def conv_lis(position, size): # le contraire
    y=position%size
    x=(position-y)/size
    #print(x)
    #print(y)
    return (x.astype(int),y.astype(int))

class Player_rd(object): #joue au hasard

    def __init__(self):
        return
    
    def get_move(self, state):
        # list with sensible moves
        sensible_moves = [move for move in state.get_legal_moves(include_eyes=False)]        

        if len(sensible_moves) > 0:
            a=np.random.randint(0,len(sensible_moves),1) #on prend un coup au hasard
            return(sensible_moves[a[0]])
        

        return go.PASS_MOVE

class Player_pl(object): # joue coup le plus probable donnee par le policy

    def __init__(self, policy_function,convertor):
        self.policy = policy_function
        self.convertor=convertor

       
    def eval_state(self, state, moves=None): # renvoie la probabilite de jouer chacun des coups dans moves en fonction de state par le policy

        if len(moves) == 0:  # pas de coup possible
            return []
        tensor = self.convertor.state_to_tensor(state) # convertit l etat du jeu en entre pour le CNN
        #tensor=np.swapaxes(tensor,1,3)
        #tensor=np.swapaxes(tensor,2,3)

        network_output = self.policy.pred(tensor)  # 
        move_indices = [conv_mat(m, state.size) for m in moves] 
        
        # A faire : 
        # get network activations at legal move locations
        distribution = network_output[0][move_indices] 
        return distribution
    
    
    
    
    def get_move(self, state):
        # list with sensible moves
        sensible_moves = [move for move in state.get_legal_moves(include_eyes=False)]
             
        
        #parcourt les coups possible et ressort celui joue le policy network    
        if len(sensible_moves) > 0:

            move_probs = self.eval_state(state, sensible_moves)
            max_prob = np.argmax(move_probs)
            (x,y)=conv_lis(max_prob,state.size)
            return (x,y)
       
        return go.PASS_MOVE
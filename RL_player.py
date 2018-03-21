import CNN_policy
import numpy as np
import go
import features as ft



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

    def __init__(self,convertor):
        # juste pour pouvoir executer play_game
        self.convertor=convertor
        return
    
    def get_move(self, state):
        if len(state.history) > 100 and state.history[-3] == go.PASS_MOVE:
        	return go.PASS_MOVE

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

        network_output = self.policy.pred(tensor)  
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
            
	    coup=(-1,-1) # init
	    while state.is_legal(coup)!=True: # si le coup donnee par le SL est illegal on prend le prochain
            	max_prob = np.argmax(move_probs) # prend le coup le plus probable
            	coup=conv_lis(max_prob,state.size)               
                move_probs= np.delete(move_probs, max_prob)
        
        
		if len(move_probs)==0:
        		sensible_moves = [move for move in state.get_legal_moves(include_eyes=False)]        

        		if len(sensible_moves) > 0:
            			a=np.random.randint(0,len(sensible_moves),1) #on prend un coup au hasard
            			return(sensible_moves[a[0]])
			return go.PASS_MOVE

            return coup       
        return go.PASS_MOVE

    
import visualisation as vis
import sys    
    
class Player_human(object): #joueur humain

    def __init__(self,convertor):
        # juste pour pouvoir executer play_game
        self.convertor=convertor
        return
    
    def get_move(self, state):
        
        # list with sensible moves
        coup=(-1,-1)
        while state.is_legal(coup)!=True:
            coup = eval(raw_input("Entrez les coordonnees:"))
            if coup ==0:
                return go.PASS_MOVE 
            if state.is_legal(coup)!=True:
                print "coup illegal"
        return coup
        


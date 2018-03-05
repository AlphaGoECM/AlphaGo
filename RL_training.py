#import keras
import go 
import CNN_policy
import RL_player as pl
import go
import time
import features as ft
#import visualisation as vis
from keras.optimizers import SGD
import numpy as np
from Tools import Tools




def play_game(player,opponent,nb_partie,preprocessor,size=9,verbose=False):
    
    #init
    etat = [go.GameState(size) for _ in range(nb_partie)] #liste des parties actuelle
    coups=[[] for _ in range(nb_partie)]  #liste des coups joues
    parties=[[] for _ in range(nb_partie)] # liste des etats du jeu
    id_gagne=[] #liste des indices des parties gagnees
    id_aband=[]#liste des parties non finies
    ratio=0

    # deroulement
    start=time.time()
    s1=0 #tps premiere partie
    
    # on joue le premier coup de chaque partie
    for i in range(nb_partie):
        coup=opponent.get_move(etat[i]) 
        coups[i].append(Tools.one_hot_action(coup,19))
        etat[i].do_move(coup)
        parties[i].append(conv.state_to_tensor(etat[i]))

    #on joue tout les coups
    actuel=player
    ancien=opponent
    fin=0 # pour arreter la boucle
    tour=1 #pour verbose
    while(fin<nb_partie): 
        for i in range(nb_partie):

                if(etat[i].is_end_of_game==False): # verifie que la partie n'est pas fini
                    coup = actuel.get_move(etat[i]) # on recupere le coup joue 
    
                    if etat[i].is_legal(coup)!=True:
                        #print("on entrre")
                        etat[i].is_end_of_game=True # met fin a la partie -> qui a gagne?
                        id_aband.append(i)
                        fin+=1
                        
                    else:   
                        coups[i].append(Tools.one_hot_action(coup,19)) # on le sauvegarde
                        etat[i].do_move(coup) #on le joue
                        parties[i].append(conv.state_to_tensor(etat[i])) #on sauvegarde l'etat du jeu

                    
                    if(etat[i].is_end_of_game==True ): 
                        fin+=1  # pour arreter la boucle
                        if(etat[i].get_winner()==-1): # -1 pour blanc
                            id_gagne.append(i)
                            ratio+=1
                    
                    if (i==1 & verbose==True):
                        tour+=1
                        print
                        print("Coup numero %i" %tour)
                        vis.vis_gs(etat[i])
                                                    
                    if (fin==1&s1==0&verbose==True):
                        print("1ere partie finie en %f secondes" % (time.time()-start))
                        s1=1
        		#vis_gs(parties[1])
        
    
        #on change de joueur
        temp=actuel
        actuel=ancien
        ancien=actuel	

    if(len(id_aband)!=nb_partie):
        ratio /=float(nb_partie-len(id_aband))
    else:
	ratio/=float(nb_partie)


    print("%d parties executees en %f secondes"%(nb_partie,time.time()-start))
    print("ratio de victoire: %f" % ratio)
   
    if(len(id_aband)!=0):
        print("Nombre de partie abandonnee: %d" % len(id_aband))

    return (coups,parties,id_gagne)
    
    



def R_learning(coups,parties,id_gagnes,policy,player):

    for i in range(len(parties)):
	
        if i in id_gagnes:
        	optimizer.lr = np.absolute(optimizer.lr)
	else :
		optimizer.lr = np.absolute(optimizer.lr)*-1
        player.policy.model.train_on_batch(np.concatenate(parties[i], axis=0),np.concatenate(coups[i],axis=0))
        #player.policy.model.train_on_batch(parties[i],coups[i])
        #player.policy.model.train_on_batch(np.concatenate(parties[i], axis=0),coups[i])

            
            
#initialisation
FEATURES = ["stone_color_feature", "ones", "turns_since_move", "liberties", "capture_size",
                    "atari_size",  "sensibleness", "zeros"]
conv=ft.Preprocess(FEATURES)
filename="model_26_2_19h53.hdf5"
#filename="model_gen5_02_.hdf5"
#filename="model_gen_8_2_5h54.hdf5"
#filename="model_gen_10_2_18h53.hdf5"
#filename="model_gen_6_2_15h.hdf5"
policy_pl=CNN_policy.CNN()
policy_pl.load (filename) 

player_rd = pl.Player_rd()
opponent_rd =pl.Player_rd()
opponent=pl.Player_pl(policy_pl,conv)
learning_rate=0.001
optimizer = SGD(lr=learning_rate)

policy_pl.model.compile(loss='categorical_crossentropy',optimizer=optimizer)
player=pl.Player_pl(policy_pl,conv) 
            
print("joueur random contre random")
play_game(player_rd,opponent_rd,1,conv,19,False)
print("joueur SL contre random")
(coups,parties,id_gagne)=play_game(player,opponent_rd,3,conv,19,False)
R_learning(coups,parties,id_gagne,policy_pl,player)

print("joueur SL contre lui meme")
(coups,parties,id_gagne)=play_game(player,opponent,10,conv,19,False)
print("joueur SL contre random apres apprentissage")
play_game(player,opponent_rd,5,conv,19,False)
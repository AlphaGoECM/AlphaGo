#import keras
import RL_player as pl

import CNN_policy
import RL_player as pl
import go
import time
import features as ft
import visualisation as vis
from keras.optimizers import SGD
import numpy as np
from Tools import Tools
import datetime

def play_game(player,opponent,size=19,verbose=True):
    
    #init
    etat = go.GameState(size) #parties actuelles
    coups=[[]]  #liste des coups joues
    parties=[[]] # liste des etats du jeu
    ratio=0
    conv=player.convertor
    # deroulement
    start=time.time()
    
    coup=opponent.get_move(etat)                 
    etat.do_move(coup)

    #on joue tout les coups
    actuel=player
    ancien=opponent
    fin=-1 # pour arreter la boucle
    i=0
    tour=1 #pour verbose
    while(etat.is_end_of_game==False):
        
        coup = actuel.get_move(etat) # on recupere le coup joue 
        etat.do_move(coup) #on le joue
        if actuel == player: 
            coups[i].append(Tools.one_hot_action(coup,19).flatten()) # on le sauvegarde
            parties[i].append(conv.state_to_tensor(etat)) #on sauvegarde l'etat du jeu

                    
        if(etat.is_end_of_game==True ): 
            fin+=1  # pour arreter la boucle
            if(etat.get_winner()==-1): # -1 pour blanc
                ratio+=1
                    
        # affiche les coups de la partie 1
        if (verbose==True):
            tour+=1
            print
            print("Coup numero %i" %tour)
            vis.vis_gs(etat) 
     
    
        #on change de joueur
        temp=actuel
        actuel=ancien
        ancien=temp
    if(ratio==1):
        print("Felicitation !!!")
    return
       
#initialisation


FEATURES = ["stone_color_feature", "ones", "turns_since_move", "liberties", "capture_size",
                    "atari_size",  "sensibleness", "zeros"]

FEATURES_with_lib_after = ["stone_color_feature", "ones", "turns_since_move", "liberties", "capture_size","liberties_after",
                    "atari_size",  "sensibleness", "zeros"]


conv=ft.Preprocess(FEATURES)
conv_with_lib_after=ft.Preprocess(FEATURES_with_lib_after)

learning_rate=0.001
optimizer = SGD(lr=learning_rate)
    


print("creation des joueur")

#joueur humain

humain=pl.Player_human(conv)


# joueur adverse

# SL cree par Mathias
f_m="model_26_2_19h53.hdf5"  # a modifier pour jouer contre un autre modele ( sans toutes les featuures )
policy_m=CNN_policy.CNN()

# SL cree par Evan
f_e="model_temp.25.hdf5" # a modifier pour jouer contre un autre modele ( avec toutes les featuures )



opponent_rd =pl.Player_rd(conv)

policy_m2=CNN_policy.CNN()
policy_m2.load(f_m)
policy_m2.model.compile(loss='categorical_crossentropy',optimizer=optimizer)
opponent_m=pl.Player_pl(policy_m2,conv)

policy_e2=CNN_policy.CNN()
policy_e2.load(f_e)
policy_e2.model.compile(loss='categorical_crossentropy',optimizer=optimizer)
opponent_e=pl.Player_pl(policy_e2,conv_with_lib_after)

print ("creation terminee")
print


print ("Debut de la partie ")
print ("Regle : \n  -entrez les coordonnees sous ce format : x,y \n  -entrez 0 pour passer votre tour")

#play_game(opponent_m,humain)  # pour jouer contre le SL de Mathias
play_game(opponent_e,humain)  # pour jouer contre le SL d'Evan



          


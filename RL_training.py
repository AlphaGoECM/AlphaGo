#import keras
import go 
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




def play_game(player,opponent,nb_partie,size=9,verbose=False):
    
    #init
    etat = [go.GameState(size) for _ in range(nb_partie)] #liste des parties actuelle
    coups=[[] for _ in range(nb_partie)]  #liste des coups joues
    parties=[[] for _ in range(nb_partie)] # liste des etats du jeu
    id_gagne=[] #liste des indices des parties gagnees
    ratio=0
    conv=player.convertor
    # deroulement
    start=time.time()
    
    # on joue le premier coup de chaque partie
    for i in range(nb_partie):
        coup=opponent.get_move(etat[i]) # celui qui commence est l'opposant il a les pierres noirs
        etat[i].do_move(coup)

    #on joue tout les coups
    actuel=player
    ancien=opponent
    fin=0 # pour arreter la boucle
    
    tour=1 #pour verbose
    while(fin<nb_partie): 
        for i in range(nb_partie):
                if(etat[i].is_end_of_game==False): # verifie que la partie n'est pas fini
                    coup = actuel.get_move(etat[i]) # on recupere le coup joue 
                    etat[i].do_move(coup) #on le joue

                    if actuel == player: 
                        coups[i].append(Tools.one_hot_action(coup,19).flatten()) # on le sauvegarde
                        parties[i].append(conv.state_to_tensor(etat[i])) #on sauvegarde l'etat du jeu

                    
                    if(etat[i].is_end_of_game==True ): 
                        fin+=1  # pour arreter la boucle
                        if(etat[i].get_winner()==-1): # -1 pour blanc
                            id_gagne.append(i)
                            ratio+=1
                    
                    # affiche les coups de la partie 1
                    if (i==1 & verbose==True):
                        tour+=1
                        print
                        print("Coup numero %i" %tour)
                        vis.vis_gs(etat[i]) 
     
    
        #on change de joueur
        temp=actuel
        actuel=ancien
        ancien=temp
             
    ratio/=float(nb_partie)
    print("%d parties executees en %f secondes"%(nb_partie,time.time()-start))
    print("ratio de victoire: %f" % ratio)   
    return (coups,parties,id_gagne,ratio)
    
    



def R_learning(coups,parties,id_gagnes,player,name,ratio,nb_partie,epoch):
    #print ('-'*15, 'Apprentissage', '-'*15)
    nb_coup_total=0
    for i in range(len(parties)):
        #print ('-'*15, 'Parties %d' %i, '-'*15)
        coups[i]=np.array(coups[i])
        parties[i]=np.array(parties[i])
        parties[i]=np.concatenate(parties[i], axis=0)
        
        nb_coup_total+=len(coups[i])
        if i in id_gagnes:
        	optimizer.lr = np.absolute(optimizer.lr)
        else :
            optimizer.lr = np.absolute(optimizer.lr)*-1
        #player.policy.model.train_on_batch(np.concatenate(parties[i], axis=0),np.concatenate(coups[i],axis=0))
        loss=player.policy.model.train_on_batch(parties[i],coups[i])
        #print("loss =",loss)
        #player.policy.model.train_on_batch(np.concatenate(parties[i], axis=0),coups[i])
    date = datetime.datetime.now()   
    filepath = ("%s/%s_R=%2f_N=%d_H=%s_%s_%sh%s.hdf5" %("RL",name,ratio,nb_partie,date.day,date.month,date.hour, date.minute))
    tfilepath = ("%s/%s_R=%2f_N=%d_H=%s_%s_%sh%s.txt" %("RL",name,ratio,nb_partie,date.day,date.month,date.hour, date.minute))
    player.policy.model.save(filepath)
    print ( '%d coups sur  %d parties appris' %(nb_coup_total,len(parties)))
    Tools.text_file(tfilepath,player.policy.model.model, nb_coup_total,epoch, date)

    return filepath 

def play_learn(player,opponent,nb_partie,epoch,policy,name,size=19,verbose=False):
	date = datetime.datetime.now()   
	preprocessor=player.convertor

	print("apprentissage debute a _%s_%s_%sh%s" %(date.day,date.month,date.hour, date.minute))          

	i=1
	print ('-'*15, 'Epoch %d' %i, '-'*15)

	(coups,parties,id_gagne,ratio1)=play_game(player,opponent,nb_partie,19,verbose)
	new_model=R_learning(coups,parties,id_gagne,player,name,ratio1,nb_partie,i)
	policy_pl=CNN_policy.CNN()
	policy_pl.load (new_model)
	policy_pl.model.compile(loss='categorical_crossentropy',optimizer=optimizer)
	del player
	player=pl.Player_pl(policy_pl,preprocessor) 
	
	while i<epoch+1	:
		i+=1
		print ('-'*15, 'Epoch %d' %i, '-'*15)
		(coups,parties,id_gagne,ratio)=play_game(player,opponent,nb_partie,19,False)
		new_model=R_learning(coups,parties,id_gagne,player,name,ratio,nb_partie,i)	
		del policy_pl
		policy_pl=CNN_policy.CNN()
		policy_pl.load (new_model)
		policy_pl.model.compile(loss='categorical_crossentropy',optimizer=optimizer)
		del player
		player=pl.Player_pl(policy_pl,preprocessor)   
	date = datetime.datetime.now()   

	print("apprentissage termine a _%s_%s_%sh%s" %(date.day,date.month,date.hour, date.minute))
	print("Ratio de victoire initial : %f, celui du dernier train : %f" %(ratio1,ratio))
	return new_model  # return le nom du dernier modele entraine
            
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

# SL cree par Mathias
f_m="model/SL/model_26_2_19h53.hdf5"
policy_m=CNN_policy.CNN()
policy_m.load(f_m)
policy_m.model.compile(loss='categorical_crossentropy',optimizer=optimizer)
player_m=pl.Player_pl(policy_m,conv)

# SL cree par Evan
f_e="model/SL/model_temp.25.hdf5"
policy_e=CNN_policy.CNN()
policy_e.load(f_e)
policy_e.model.compile(loss='categorical_crossentropy',optimizer=optimizer)
player_e=pl.Player_pl(policy_e,conv_with_lib_after)

# joueur adverse
player_rd = pl.Player_rd(conv)
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


print ("Test de performance joueur aleatoire contre lui meme ")
play_game(player_rd,opponent_rd,2,19,False)

print

#parametre d'apprentissage

nb_partie=100
epoch=10
verbose=False
size=19

# entrainement SL Mathias
policy=policy_m

print(" SL cree par Mathias")
print
print("entrainement contre joueur random") 
name="M_random"

#play_learn(player_m,opponent_rd,nb_partie,epoch,policy,name,size,verbose)

# entrainement contre lui meme
print("entrainement contre lui-meme")
name="M_M"

#play_learn(player_m,opponent_m,nb_partie,epoch,policy,name,size,verbose)

#entrainement contre le SL d'Evan
print("entrainement contre le SL d'Evan")
name="M_E"

#play_learn(player_m,opponent_e,nb_partie,epoch,policy,name,size,verbose)

print

# entrainement SL Evan
policy=policy_e

print(" SL cree par Evan")
print
print("entrainement contre joueur random") 
name="E_random"

#play_learn(player_e,opponent_rd,nb_partie,epoch,policy_e,name,size,verbose)

# entrainement contre lui meme
print("entrainement contre lui-meme")
name="E_E"
#play_learn(player_e,opponent_e,nb_partie,epoch,policy,name,size,verbose)

#entrainement contre le SL de Mathias
print("entrainement contre le SL de Mathias")
name="E_M"

play_learn(player_e,opponent_m,nb_partie,epoch,policy,name,size,verbose)

print

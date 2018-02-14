import sgf_to_gs
import features as ft

#visualisation d'un gs
def vis_gs(gs):
    x=0
    taille=len(gs.board)
    while x < taille:
        y=0
        string=""
        while y < taille:
            case=gs.board[y,x] #  il faut inverser x et y ici
            if  case==-1:
                string+="W"
            if  case==0:
                string+="."
            if  case==1: 
                string+="B"
            string+=" "
            y+=1
        print (string) 
        x+=1
    if gs.current_player==1:
        print ("current_player=black")
    if gs.current_player==-1:
        print ("current_player=white")

#visualisation d'un layer
def vis_layer(layer,num_layer):
    x=0
    taille=len(layer[num_layer,:,:])
    while x < taille:
        string =""
        y=0
        while y < taille:
            if layer[num_layer,y,x]==1:  # il faut inverser x et y ici
                string = string + " 1"    
            else:
                string = string + " ."
            y+=1
        print (string)
        x+=1
        
# visualisation d'un layer et du gs associé
def vis_gs_layer(gs,layer,num_layer):
    x=0
    taille=len(gs.board)
    while x < taille:
        string =""
        y=0
        string=""
        #affichage du gs
        while y < taille:
            case=gs.board[y,x] #  il faut inverser x et y ici
            if  case==-1:
                string+="W"
            if  case==0:
                string+="."
            if  case==1: 
                string+="B"
            string+=" "
            y+=1
        #affichage du layer
        string+="          "
        y=0
        while y < taille:
            if layer[num_layer,y,x]==1:  # il faut inverser x et y ici
                string = string + " 1"    
            else:
                string = string + " ."
            y+=1
        print (string) 
        x+=1
# -*- coding: utf-8 -*-
"""
Calcul sur reseau - Exemple 2 - Modèle d'Ising
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#----------------------Fonctions-----------------------------

def conditions_initiales():
    global N,nIter
    N=32                                #taille du reseau
    nIter=100                          #nombre d'iterations temporelles

    TEMPERATURE=2.35
    BOLTZMANN=1
    J=1
    H=0

    init=[BOLTZMANN,TEMPERATURE,J,H]
    return init

def initialize_table():
    spin=np.zeros([N+2,N+2])            #le reseau des spins

    for i in range(1,N+1):
        for j in range(1,N+1):
            spin[i,j]=-1+2*(np.random.randint(0, 1 + 1))  #initialisation aleatoire
    return spin
    
def evolve_table(init,spin):
    #balayage des noeuds blancs du reseau
    for i in range(1,N+1):
        ij=(i%2)+1                  #on alterne le noeud de depart
        for j in range(ij,N+1,2):   #chaque 2 noeuds
            nb_neighbors=count_neighbors(spin,[i,j])
            ener=-init[2]*spin[i,j]*nb_neighbors-init[3]*spin[i,j]
            enerprime=-init[2]*(-spin[i,j])*nb_neighbors-init[3]*spin[i,j]
            if metropolis(enerprime-ener,init):
                spin[i,j]=-spin[i,j]
    #fin du balayage des noeuds blancs
    
    #balayage des noeuds noirs
    for i in range(1,N+1):
        ij=((i+1)%2)+1              #on alterne le noeud de depart
        for j in range(ij,N+1,2):   #chaque 2 noeuds
            nb_neighbors=count_neighbors(spin,[i,j])
            ener=-init[2]*spin[i,j]*nb_neighbors-init[3]*spin[i,j]
            enerprime=-init[2]*(-spin[i,j])*nb_neighbors-init[3]*spin[i,j]
            if metropolis(enerprime-ener,init):
                spin[i,j]=-spin[i,j]
    #fin du balayage des noeuds noirs
    return spin

def metropolis(deltaE,init):
    prob=min(1,np.exp(-deltaE/(init[0]*init[1])))
    if np.random.random()<=prob:
        return True
    else:
        return False
    
def count_neighbors(oldgrid,pos):
    nb_neighbors=0              #n
    for i in [-1,0,1]:
        for j in [-1,0,1]:            #loop over all neighbors of the cell
            if abs(i)+abs(j)!=1:      #excluding self and diagonals            
                continue
            elif (pos[0] + i) < 0 or (pos[1] + j) < 0:
                continue              #avoiding negative index when on a side
            try:
                nb_neighbors += oldgrid[pos[0]+i][pos[1]+j]
            except IndexError:        #out of bounds (cell on side)
                continue
    return nb_neighbors

#----------------------------Main----------------------------

init=conditions_initiales()
spin=initialize_table()

fig = plt.figure()

process=[]
for iter in range(0,nIter):         #boucle temporelle
    spin=evolve_table(init,spin)
    im = plt.imshow(spin,animated=True,cmap='Greys')
    process.append([im])
    if iter%100==0:
        print(iter)

spin_ani = animation.ArtistAnimation(fig, process,interval=50)
                                 
spin_ani.save('spin_ani1.mp4')

# -*- coding: utf-8 -*-
"""
exemple 1 : calcul sur réseau

Tectonique des plaques

(Modèle de Olami-Christensen-Feder 2D de failles tectoniques)
"""

import numpy as np
import matplotlib.pyplot as plt

#---------------------Conditions initiales-------------------
N=128                           #taille du reseau
NN=4                            #nombre de voisins immediats
fc=1.                           #seuil d'instabilite
deltaf=1.e-4                    #amplitude du forcage
alpha=0.15                      #parametres de conservation
nIter=10000                    #nombre d'iterations temporelles
#----------------------Main----------------------------------

dx=np.array([-1,0,1,0])         #stencil du voisinage
dy=np.array([0,-1,0,1])
force=np.zeros([N+2,N+2])       #tableau des forces
toppling=np.zeros(nIter,dtype='int')    #sequence temporelle avalanches
for i in range(1,N+1):
    for j in range(1,N+1):
        force[i,j]=fc*(np.random.uniform()) #force initiale aleatoire

graph=0
fig310=[]
valeurpt1=[]
valeurpt2=[]
valeurpt3=[]

for iter in range(0,nIter):             #iteration temporelle
    move=np.zeros([N+2,N+2])            #remise a zero du tableau
    #balayage du reseau pour identifier les noeuds instables
    for i in range(1,N+1):
        for j in range(1,N+1):
            if force[i,j]>=fc:           #le noeud i,j est instable
                move[i,j]-=force[i,j]           #eqn 3.13, a zero
                move[i+dx[:],j+dy[:]]+=alpha*force[i,j] #eqn 3.14, aux voisins
                toppling[iter]+=1           #cumul des noeuds instables
    #fin du balayage du reseau
                
    if toppling[iter]>0:                #il y a eu des instabilites
        force+=move                     #mise a jour du reseau
    else:                               #pas d'instabilite
        force[:,:]+=deltaf              #eqn 3.11 : forcage du reseau
    
#--------------------Commandes graphiques--------------------
        
    valeurpt1.append(force[64,64])
    valeurpt2.append(force[32,32])
    valeurpt3.append(force[64,76])
    
    if iter in [0,250,500,999]:
        fig310.append(force)
        
    
    

#---------------------Affichage------------------------------
    totalf=force.sum()                  #force totale sur le reseau
    print('%d, force %f, toppl %d'%(iter,totalf,toppling[iter]))
#fin de l'iteration temporelle
#END
    
#fig 3.8
plt.figure(0,figsize=(9,6))
plt.plot(np.linspace(0,nIter-1,nIter),toppling)
plt.xlabel("Iteration")
plt.ylabel("E")
plt.axis([0,nIter,0,max(toppling)+2])
#plt.savefig('verif308.png')

#fig 3.9
plt.figure(1,figsize=(9,6))
plt.plot(np.linspace(0,nIter-1,nIter),valeurpt1,'r')
plt.plot(np.linspace(0,nIter-1,nIter),valeurpt2,'g')
plt.plot(np.linspace(0,nIter-1,nIter),valeurpt2,'b')
plt.xlabel('Iteration')
plt.axis([0,nIter,-0.05,fc+0.05])
#plt.savefig('verif309.png')

#fig3.10

f, axarr = plt.subplots(2, 2,figsize=(12,9))
axarr[0, 0].imshow(fig310[0],cmap='plasma')
axarr[0,0].axis('off')
axarr[0, 1].imshow(fig310[1],cmap='plasma')
axarr[0,1].axis('off')
axarr[1, 0].imshow(fig310[2],cmap='plasma')
axarr[1,0].axis('off')
axarr[1, 1].imshow(fig310[3],cmap='plasma')
axarr[1,1].axis('off')
plt.tight_layout()
#plt.savefig('verif310.png')

plt.show()
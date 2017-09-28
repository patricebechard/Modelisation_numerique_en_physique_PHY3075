# -*- coding: utf-8 -*-
"""
Projet 3 - Gravure magnétique
Patrice Béchard p1088418
mars 2017
"""
#------------------Modules-----------------------------------
import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
import time
import scipy.constants as cst
from matplotlib import animation

plt.style.use('patrice')
print("Execution Start Time :",time.asctime())
start=time.time()            #time starts at beginning of execution
#---------------------Initial Conditions---------------------  
"""
Grid will be size NxN (All are global variables)
config represents the initial configuration
    random : every cell is initialized randomly
    uniform-: every cell is initialized as '-1'
    uniform+: every cell is initialized as '+1'

evolmethod represents the way the grid is updated
    copy : all the cells are updated from an old version of the grid
    chess: cells are updates following a black/white cells oscillation
stencil represents the number of neighbors of each cell
    1 : von Neumann stencil (4 neighbors) 
    2 : Moore stencil (8 neighbors)
    3 : Triangular stencil (6 neighbors)
"""
N=64                              #dimensions of grid
nIter=200                         #number of iterations
config='uniform-'
evolmethod='chess'
stencil=1

temperatures=[0.1]  #temperateures to loop over
source=[+0.2]              #and values of H_0   
deltatemp=[2.4]         
#---------------------Classes--------------------------------
class InputError(Exception):
    """Bad input for a given problem"""
    pass

class Parameters:
    """Default parameters for the simulation"""
    BOLTZMANN=1
    T0=0.1              #temperature of system
    DELTAT=2.4          #temperature pulse amplitude
    J=1
    R=10                #radius of temperature pulse area
    P=500               #period for oscillating H
    H0=-0.2             #outside magnetization term
    SIGMA=10
    X0,Y0=32,32         #where temperature pulse is centered
    time0=100           #time for temperature pulse
    
    GRAVURE=True
    VARH=False
    ANIMATION=True

class Grid(Parameters):
    """Grid containing each element of the network"""
    def __init__(self,dim,distribution):
        """Initialize dimensions and grid values depending on config"""
        if len(dim)!=2:
            raise InputError('Grid must be 2-D')
        self._dimensions=dim
        self._grid=[]
        if distribution=='uniform-':
            for i in range(self._dimensions[0]):#each node of the grid is a cell
                self._grid.append([-1 for j in range(self._dimensions[1])])  
        elif distribution=='uniform+':
            for i in range(self._dimensions[0]):#each node of the grid is a cell
                self._grid.append([+1 for j in range(self._dimensions[1])])
        elif distribution=='random':
            for i in range(self._dimensions[0]):#each node of the grid is a cell
                self._grid.append([-1 if np.random.random()<0.5 else 1 for j in range(self._dimensions[1])])
        elif distribution=='zeros':
            for i in range(self._dimensions[0]):
                self._grid.append([0 for j in range(self._dimensions[1])])
        else:
            raise InputError("Bad initial distribution. Please choose between 'uniform (+/-)' and 'random'")
            
    def __getitem__(self,pos):
        """To access item via indexes"""
        if len(pos)!=2:
            raise IndexError("Grid is 2-D, enter index as [x,y]")
        if not (0<=pos[0]<self._dimensions[0] and 0<=pos[1]<self._dimensions[1]):
            raise IndexError("Out of bounds")
        return self._grid[pos[0]][pos[1]]

    def evolution(self,cIter):
        """We make the grid evolve for one time step"""
        if evolmethod=='copy':
            self._copy_evol(cIter)
        elif evolmethod=='chess':
            self._chess_evol(cIter)
        else:
            raise InputError("Bad evolution method, please choose another one")
        ener=self.ener_spin()               #compute mean energy
        magnet=self.magnet_spin()           #compute mean magnetization
        return ener,magnet
    
    def _copy_evol(self,cIter):
        """Evolving the system by copying old grid"""
        self._old=copy.deepcopy(self._grid)   #we are updating the grid, we deepcopy info in old grid
        self.H=self.define_H()
        for i in range(self._dimensions[0]):  #loop over all cells
            for j in range(self._dimensions[1]):
                self._count_neighbors([i,j]) 
                ener=-self.J*self._grid[i][j]*self._nb_neighbors\
                                            -self.H*self._grid[i][j]
                enerprime=-self.J*(-self._grid[i][j])*self._nb_neighbors\
                                            -self.H*(-self._grid[i][j])
                if self._metropolis(enerprime-ener,[i,j],cIter):
                    self._grid[i][j]= -self._grid[i][j]
                    
    def _chess_evol(self,cIter):
        """Evolving the system with the chess method (see notes)"""
        self._old=copy.deepcopy(self._grid)   #we are updating the grid, we deepcopy info in old grid
        self.H=self.define_H()
        for i in range(N):          #white cells
            ij=(i%2)
            for j in range(ij,N,2):
                self._count_neighbors([i,j]) 
                ener=-self.J*self._grid[i][j]*self._nb_neighbors\
                                            -self.H*self._grid[i][j]
                enerprime=-self.J*(-self._grid[i][j])*self._nb_neighbors\
                                            -self.H*(-self._grid[i][j])
                if self._metropolis(enerprime-ener,[i,j],cIter):
                    self._grid[i][j]= -self._grid[i][j]     

        self._old=copy.deepcopy(self._grid)   #we are updating the grid, we deepcopy info in old grid
        for i in range(N):          #black cells
            ij=((i+1)%2)
            for j in range(ij,N,2):
                self._count_neighbors([i,j]) 
                ener=-self.J*self._grid[i][j]*self._nb_neighbors\
                                            -self.H*self._grid[i][j]
                enerprime=-self.J*(-self._grid[i][j])*self._nb_neighbors\
                                            -self.H*(-self._grid[i][j])
                if self._metropolis(enerprime-ener,[i,j],cIter):
                    self._grid[i][j]= -self._grid[i][j]                        
    
    def _count_neighbors(self,pos):
        """Counts the neighbors of a cell"""
        self._nb_neighbors=0              
        if stencil==1:
            self._von_neumann(pos)
        elif stencil==2:
            self._moore(pos)
        elif stencil==3:
            self._triangular(pos)
        else:
            raise InputError("Stencil doesn't exist, please choose another one")
    
    def _von_neumann(self,pos):
        """Von Neumann Stencil"""
        self.J=1
        for i in [-1,0,1]:
            for j in [-1,0,1]:            #loop over all neighbors of the cell
                if abs(i)+abs(j)!=1:      #excluding self and diagonals            
                    continue
                elif (pos[0] + i) < 0 or (pos[1] + j) < 0:
                    continue              #avoiding negative index when on a side
                try:
                    self._nb_neighbors += self._old[pos[0]+i][pos[1]+j]
                except IndexError:        #out of bounds (cell on side)
                    continue

    def _moore(self,pos):
        """Moore Stencil"""
        self.J=1/3
        for i in [-1,0,1]:
            for j in [-1,0,1]:            #loop over all neighbors of the cell
                if abs(i)+abs(j)==0:      #excluding self           
                    continue
                elif (pos[0] + i) < 0 or (pos[1] + j) < 0:
                    continue              #avoiding negative index when on a side
                elif abs(i)+abs(j)==1:
                    try:
                        self._nb_neighbors += 2*self._old[pos[0]+i][pos[1]+j]
                    except IndexError:        #out of bounds (cell on side)
                        continue
                elif abs(i)+abs(j)==2:
                    try:
                        self._nb_neighbors += self._old[pos[0]+i][pos[1]+j]
                    except IndexError:        #out of bounds (cell on side)
                        continue             
                
    def _triangular(self,pos):
        """Triangular Stencil"""
        self.J=2/3
        for point in [[0,1],[0,-1],[1,0],[-1,0],[-1,-1],[1,1]]:        
            if (pos[0] + point[0]) < 0 or (pos[1] + point[1]) < 0:
                continue              #avoiding negative index when on a side
            try:
                self._nb_neighbors += self._old[pos[0]+point[0]][pos[1]+point[1]]
            except IndexError:        #out of bounds (cell on side)
                continue
                
    def _metropolis(self,deltaE,pos,cIter):
        """Probabilistic test"""
        prob=min(1,np.exp(-deltaE/(self.BOLTZMANN*self.temp_profile(pos,cIter))))
        if np.random.random()<=prob:
            return True
        return False
    
    def define_H(self,param=False):
        """Outside magnetization term"""
        if self.VARH==True:
            return self.H0*np.sin(2*cst.pi*cIter/self.P)
        else:
            return self.H0
         
    def ener_spin(self,param=False):
        """mean energy"""
        sum=0
        for i in range(N):
            for j in range(N):
                self._nb_neighbors=0
                if param:
                    self._old=self._grid
                self._count_neighbors([i,j],)
                self.H=self.define_H()
                sum+=-self.J*self._grid[i][j]*self._nb_neighbors-self.H*self._grid[i][j]
        return (1/(N*N))*sum

    def magnet_spin(self,param=False):
        """mean magnetization"""
        sum=0
        for i in range(N):
            for j in range(N):
                sum+=self._grid[i][j]
        return (1/(N*N))*sum
    
    def space_profile(self,pos):
        """statial profile for the computation of the temperature pulse"""
        radius=(pos[0]-self.X0)**2+(pos[1]-self.Y0)**2
        return 1 if radius<=(self.R*self.R) else 0
    
    def temp_profile(self,pos,cIter):
        """Temperature profile for magnetic engraving"""
        if self.GRAVURE:
            return self.T0+self.DELTAT*self.space_profile(pos)*np.exp(-(cIter-self.time0)**2/self.SIGMA**2)
        else:
            return self.T0
    
    def show_grid(self):
        """We show the playing grid cell by cell"""
        data=np.zeros([self._dimensions[0],self._dimensions[1]])
        for i in range(self._dimensions[0]):
            for j in range(self._dimensions[1]):
                data[i,j]=self._grid[i][j]
        plt.figure(figsize=(9,6))
        plt.imshow(data,cmap='Greys')
        plt.show()
                  
#-------------------------Functions--------------------------
def profile_H(grille):
    """Only used to plot sin wave"""
    return 0.75*np.sin(2*cst.pi*np.linspace(0,nIter,nIter+1)/grille.P)

def generate_evolution(grille,i,length):
    """
    Main part of the code
    Computing of energy, magnetization and animations is done here
    """
    cIter=0
    enermoy=[grille.ener_spin(True)]
    magnetmoy=[grille.magnet_spin(True)]
    profil=[grille.define_H(True)]
    Tprofil=[grille.temp_profile([grille.X0,grille.Y0],cIter)]
    if grille.ANIMATION:
        fig0=plt.figure(0)
        process=[]
        im=plt.imshow(grille._grid,animated=True,cmap='Greys')  #initial config
        process.append([im])
    for cIter in range(1,nIter+1):
        ener,magnet=grille.evolution(cIter)
        Tprofil+=[grille.temp_profile([grille.X0,grille.Y0],cIter)]
        enermoy.append(ener)
        magnetmoy.append(magnet)
        profil.append(grille.H)
        if grille.ANIMATION:
            im=plt.imshow(grille._grid,animated=True,cmap='Greys')            
            process.append([im])
        if (cIter)%(nIter//10)==0:
            currenttime=time.time()-start
            print("%d  Elapsed time : %d h %d m %d s"\
                %(cIter,currenttime//3600,currenttime//60,currenttime%60))  
    plt.figure(1)
    if grille.H0!=0:
        magnetization.append(magnetmoy)
    if grille.ANIMATION:
        spin_ani = animation.ArtistAnimation(fig0, process,interval=50)
        spin_ani.save('ising%d_%d.mp4'%(int(grille.T0*100),nIter))           
    return
#--------------------------MAIN------------------------------
grille=Grid([N,N],config)           #initialize grid

global magnetization
magnetization=[]           
tempinit=True
for i in range(len(temperatures)):
    grille.T0=temperatures[i]                   #setting T0 for this simulation
    grille.H0=source[i]                         #and source term
    grille.DELTAT=deltatemp[i]                  #and delta T
    generate_evolution(grille,i,len(temperatures))    #evolution of the grid over nIter iterations
    grille=Grid([N,N],config)       #reset to initial config for next simulation


totaltime=time.time()-start
print("Total time : %d h %d m %d s"%(totaltime//3600,(totaltime//60)%60,totaltime%60))

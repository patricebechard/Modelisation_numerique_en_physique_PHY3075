# -*- coding: utf-8 -*-
"""
Runge-Kutta Method (EDO)

auteur : Patrice Béchard
date : 10 janvier 2017
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

############# FONCTIONS

def main():
    nMax=1000                           #max number of steps
    tFin=10.                            #max time
    eps=1.e-5                           #tolerance for RK avec pas adaptif
    
    t=np.zeros(nMax)                    #array to store time
    u=np.zeros([nMax,2])                #array to store positions
    u[0,:]=np.array([3.,3.])            #initial position
    
    nn=0                                #iterator for number of steps
    h=0.1                               #initial step
    
    while t[nn]<tFin and nn<nMax:
        u1=rungeKutta(t[nn],u[nn,:],h)  #full step
        u2a=rungeKutta(t[nn],u[nn,:],h/2.)  #first half-step for adaptive step method
        u2=rungeKutta(t[nn],u2a[:],h/2.)    #second half-step
        delta=max(abs(u2[0]-u1[0]),abs(u2[1]-u1[1]))   #eqn 1.42
        if delta>eps:
            h/=1.5
        else:
            nn+=1
            t[nn]=t[nn-1]+h
            u[nn,:]=u2[:]
            if delta<=eps/2.:
                h*=1.5
#        print("{0},t {1}, X {2}, Y {3}".format(nn,t[nn],u[nn,0],u[nn,1]))
    plt.plot(u[:,0],u[:,1],'.')
    
def slope(t,u):
    A,B=1.,3.
    gX=A-(B+1.0)*u[0]+u[0]**2*u[1]      #eq 1.39 RHS
    gY=B*u[0]-u[0]**2*u[1]              #eq 1.40 rhs
    eval=np.array([gX,gY])              #vecteurs rhs pentes
    return eval
    
def rungeKutta(t0,uu,h):
    g1=slope(t0,uu)                   #slope1 ()
    g2=slope(t0+h/2,uu+(h/2)*g1)      #slope2
    g3=slope(t0+h/2,uu+(h/2)*g2)      #slope3
    g4=slope(t0+h,uu+h*g3)            #slope4
    
    uNext=uu+(h/6)*(g1+2*g2+2*g3+g4)
    return uNext

############### MAIN

main()
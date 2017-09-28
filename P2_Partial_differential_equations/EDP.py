# -*- coding: utf-8 -*-
"""
Title : Plasma confinement in a Tokamak

Author : Patrice Bechard

Date : February 20th, 2017
"""
#------------------------Import Modules----------------------
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy as sp
import time
import random
import copy

plt.style.use('patrice')
start=time.time()            #time starts at beginning of execution

#--------------------------Functions-------------------------
def diffusionFTCS(u,dt,dx,D):
    """FTCS equation 2.13"""
    results=np.zeros(len(u))
    for j in range(1,len(u)-1):
        results[j]=u[j]+(D*dt/(dx*dx))*(u[j+1]-2*u[j]+u[j-1])
    results[0]=0
    results[-1]=0
    return results
    
def TuringDiffusion(X,Y,dt,dx,D):
    """FTCS 2 variables, eqn 2.24"""
    resultsX,resultsY=np.zeros(len(X)),np.zeros(len(X)) #both are same size
    mu=D
    nu=D/2
    resultsX[0]=X[0]+(dt/(dx*dx))*mu*(X[-2]-2*X[0]+X[1])+ \
                (dt/32)*(-7*X[0]*X[0]-50*X[0]*Y[0]+57)
    resultsY[0]=Y[0]+(dt/(dx*dx))*nu*(Y[-2]-2*Y[0]+Y[1])+ \
                (dt/32)*(7*X[0]*X[0]+50*X[0]*Y[0]-2*Y[0]-55)

    for j in range(1,len(X)-1):
        resultsX[j]=X[j]+(dt/(dx*dx))*mu*(X[j-1]-2*X[j]+X[j+1])+ \
                    (dt/32)*(-7*X[j]*X[j]-50*X[j]*Y[j]+57)
        resultsY[j]=Y[j]+(dt/(dx*dx))*nu*(Y[j-1]-2*Y[j]+Y[j+1])+ \
                    (dt/32)*(7*X[j]*X[j]+50*X[j]*Y[j]-2*Y[j]-55)
    resultsX[-1]=resultsX[0]
    resultsY[-1]=resultsY[0]
    return resultsX, resultsY
    
def advectionFTCS(u,dt,dx,v):
    """FTCS advection, eqn 2.30"""
    results=np.zeros(len(u))
    for j in range(1,len(u)-1):
        results[j]=u[j]-(v*dt/(2*dx))*(u[j+1]-u[j-1])
    results[0]=1
    results[-1]=0
    return results 
    
def advectionLAX(u,dt,dx,v):
    """FTCS advection, eqn 2.30"""
    results=np.zeros(len(u))
    for j in range(1,len(u)-1):
        results[j]=0.5*(u[j+1]+u[j-1])-(v*dt/(2*dx))*(u[j+1]-u[j-1])
    results[0]=1
    results[-1]=0
    return results 

def advectionLW(u,dt,dx,v):
    halfstep=np.zeros(len(u))
    results=np.zeros(len(u))
    for j in range(len(u)-1):
        halfstep[j]=0.5*(u[j+1]+u[j])-(v*dt/(2*dx))*(u[j+1]-u[j])   #lax for half step
    for j in range(1,len(u)-1):
        results[j]=u[j]-(dt/dx)*v*(halfstep[j]-halfstep[j-1])
    results[0]=1
    results[-1]=0
    return results
    
def tridiag(a,b,c,r,u,n):
    """Tridiagonal solver from Press et. al. (1992), section 2.4
    
    a[n], b[n], c[n] are the three diagonals of the matrix
    r[n]is the RHS vector
    u[n] is the solution the the n step at input, solution to n+1 at output
    """
    gam=np.zeros(n)                 #vecteur de travail
    bet=b[0]
    if bet == 0 : raise Exception("beta=0 1")
    u[0]=r[0]/bet
    for j in range(1,n-1):          #decomposition LU
        gam[j]=c[j-1]/bet
        bet=b[j]-a[j]*gam[j]
        if bet == 0 : raise Exception("beta=0 2")
        u[j]=(r[j]-a[j]*u[j-1])/bet
    for j in range(n-2,0,-1):       #backsubstitution
        u[j]-=gam[j+1]*u[j+1]
    return 0
    
def crank_nicolson(r,u,a,b,c):
    r[0]=u[0]-0.5*(b[0]*u[0]+c[0]*u[1])
    for i in range(1,len(r)-1):
        r[i]=u[i]-0.5*(a[i]*u[i-1]+b[i]*u[i]+c[i]*u[i+1])
    r[-1]=u[-1]-0.5*(a[-1]*u[-2]+b[-1]*u[-1])
    r=2*r
    return r

def func(x,center):
    """eqn 2.14"""
    return 0.5*(1-sp.special.erf((x-center)/0.1))
    
def define_abc(D,dt,dx):
    a=np.ones(len(x))*(-(D*dt)/(dx*dx))
    a[0]=0
    b=np.ones(len(x))*(1+2*D*dt/(dx*dx))
    c=np.ones(len(x))*(-(D*dt)/(dx*dx))  
    c[-1]=0
    return a,b,c

def define_r(r,u,a,b,c):
    #r=u
    r=crank_nicolson(r,u,a,b,c)
    r[0],r[-1]=1,0                      #limit conditions
    r[1],r[-2]=r[1]-a[1]*r[0],r[-2]+c[-2]*r[-1]
    a[1],b[0],c[0]=0,1,0                #changing the matrix
    a[-1],b[-1],c[-2]=0,0,0
    return r,a,b,c

#----------------------------Main----------------------------
"""
#FIGURE 2.1      Diffusion FTCS

nmax=2000
D=1
dx=0.05
dt=0.001
x=np.linspace(0,5,101)
u=[func(x,2.5)]

plt.plot(x,u[0][:],'k:',lw=0.3)

for i in range(1,nmax+1):
    u=np.append(u,[np.zeros(len(x))],axis=0)
    u[i][:]=diffusionFTCS(u[i-1][:],dt,dx,D)
    if i==2000:
        plt.plot(x,u[i][:],'r')
    elif i%200==0:
        plt.plot(x,u[i][:],'k',lw=0.3)
    
plt.show()
"""
"""
#Figure 2.2     diffusion FTCS

D=0.01
nmax=2000
x=np.linspace(0,1,101)
dx=0.01
dt=0.001
u=np.array([random.random()-0.5 for i in range(len(x))])
u[0]=0
u[-1]=0
u=[u]
plt.plot(x,u[0][:])
plotdispersion=1

for i in range(1,nmax+1):
    temps=i*dt
    u=np.append(u,[np.zeros(len(x))],axis=0)
    u[i][:]=diffusionFTCS(u[i-1][:],dt,dx,D)
    if temps in [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2]:
        plt.plot(x,u[i][:]+plotdispersion*0.5)
        plotdispersion+=1
        
plt.show()
"""  
"""
# TURING, figures 2.3, 2.4, Diffusion FTCS 2 variables

x=np.linspace(0,1,101)
dx=x[2]-x[1]                #they are all the same
dt=0.001
D=0.00075
a=0.15
nmax=10000
Xeq=1
Yeq=1

X=np.array([Xeq+a*(2*random.random()-1) for i in range(len(x))])
Y=np.array([Yeq+a*(2*random.random()-1) for i in range(len(x))])
X=[X]
Y=[Y]

plt.plot(x,X[0][:])
plotdispersion=1

for i in range(1,nmax+1):
    temps=i*dt
    X=np.append(X,[np.zeros(len(x))],axis=0)
    Y=np.append(Y,[np.zeros(len(x))],axis=0)
    X[i][:],Y[i][:]=TuringDiffusion(X[i-1][:],Y[i-1][:],dt,dx,D)
    if temps in [0.01,0.03,0.1,0.3,1.,3.,10.,30.,200.]:
        plt.plot(x,X[i][:]+plotdispersion*0.5)
        plotdispersion+=1
    if temps%1==0:
        print(temps)
        
plt.show()
"""
"""
#Figures 2.7, 2.8, 2.9 ADVECTION FTCS, LAX & LAX-WENDROFF
nmax=1000
x=np.linspace(0,5,501)
dx=x[2]-x[1]

dt=0.01
v=0.4
u=[func(x,0.5)]

plt.plot(x,u[0][:],':k',lw=0.3)

for i in range(1,nmax+1):
    u=np.append(u,[np.zeros(len(x))],axis=0)
    #u[i][:]=advectionFTCS(u[i-1][:],dt,dx,v)       #ftcs
    #u[i][:]=advectionLAX(u[i-1][:],dt,dx,v)         #lax
    u[i][:]=advectionLW(u[i-1][:],dt,dx,v)          #lax-wendroff
    if i==1000:
        plt.plot(x,u[i][:],'r',lw=0.5)
    elif i%100==0:
        plt.plot(x,u[i][:],'k',lw=0.3)

result=func(x,4.5)
plt.plot(x,result,'k:',lw=0.3)
plt.axis([0,5,-0.5,1.5])
plt.show()
"""

#Methodes implicites

nmax=2000
x=np.linspace(0,5,101)
dx=0.05
dt=0.002
D=1

a,b,c,r=[np.zeros(len(x)) for i in range(4)]    #initiate empty arrays

u=func(x,2.5)

plt.plot(x,u,'k:',lw=0.3)

"""
pseudocode :
u[j] selon cond initiales

for n in range(1,nmax+1):
    recalculer D (chi dans tokamak)
    recalculer a,b,c
    calculer le vecteur r
    enoncer les conditions limites (voir 2.61 pour cas lineaire)
    tridiag
"""

for n in range(1,nmax+1):
    D=1
    a,b,c=define_abc(D,dt,dx)
    r,a,b,c=define_r(r,u,a,b,c)
    tridiag(a,b,c,r,u,len(x))
    if n==nmax:
        plt.plot(x,u,'r')
    elif n%(nmax/50)==0:
        plt.plot(x,u,'k',lw=0.3)
    


plt.show()

print("TOTAL TIME : ",time.time()-start)
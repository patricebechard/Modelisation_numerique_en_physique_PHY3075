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
from itertools import cycle

plt.style.use('patrice')
print("Execution Start Time :",time.asctime())
start=time.time()            #time starts at beginning of execution

#--------------------------Functions-------------------------

def cond_initiales():
    """Setting initial conditions"""
    alpha=1
    beta=4
    chi0=1
    chi1=2
    init=[alpha,beta,chi0,chi1]     #to compute chi
    
    global dx,dt    
    dx=0.02                         #gap between spatial points
    dt=5e-4                         #temporal step
    
    xrange=[-10,10]
    pos=np.linspace(xrange[0],xrange[1],int((xrange[1]-xrange[0])/dx))
    
    global tmax,nx
    tmax=2.5                         #maximum time
    nx=len(pos)                     #number of points in x
    
    temp=np.ones(nx)
    
    H0=14.4
    sigma=2
    values_H=[H0,sigma]
    
    return pos,temp,init,values_H

def confinement(pos,temp,init,values_H):
    
    a,b,c,r=[np.zeros(nx) for i in range(4)]    #initiate empty arrays    
    temps=0
    
    timenum3=np.array([])                       #for the #3 exploration
    posnum3=np.array([])
    
    while temps<tmax:
        temps+=dt               #step forward in time
        temps=round(temps,6)    #having 'squared' time (no truncation bias)
        D=compute_D(temp,init)
        a,b,c=compute_abc(D)
        r=compute_r(r,temp,a,b,c,pos,values_H)
        r,a,b,c=cond_limites(r,a,b,c)
        tridiag(a,b,c,r,temp)
        
        if temps in [0.05,0.1,0.25,0.5,1.,2.5,5.,10.,25.,50.,100.]:
            
            plt.figure(0,figsize=(9,6))     #we plot figures like 2.16 and 2.17
            plt.plot(pos,temp,'k',lw=0.75)
            
            plt.figure(1,figsize=(9,6))
            
            dtdx=np.zeros(nx)
            for i in range(-1,nx-1):
                dtdx[i]=(temp[i-1]-temp[i+1])/(2*dx)    #derivative
            chi=compute_chi(temp,init)                  #chi
            
            plt.plot(pos,dtdx,'r',lw=0.75)
            plt.plot(pos,chi,'g--',lw=0.75)
            
            #num 3
            if temps>0.1:
                i=nx//2
                while chi[i]>4:i+=1
                timenum3=np.append(timenum3,[temps])
                posnum3=np.append(posnum3,[pos[i]])
            
            #num5
            if temps==tmax:                  #at equilibrium
                #plt.figure(4,figsize=(9,6))
                #plt.plot(-dtdx,-chi*dtdx,'.')
                f=open('datanum4_%s_chi%s.txt'%(str(values_H[0]),str(init[3])),'w')
                f.write('%s%s%s'%(-dtdx,'\n',-chi*dtdx))
                f.close()
            
                
                        
            
    plt.figure(0)
    plt.plot(pos,compute_H(pos,values_H),'k--',lw=0.5)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$T(x,t)$')
    plt.axis([-10,10,0,max(temp)+5])
    plt.savefig('fig_216_%d_chi%d.png'%(values_H[0],init[3]))
    
    plt.figure(1)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$dT(x,t)/dx$ , $\chi(x,t)$')
    plt.axis([0,1.5,-0.25,10.25])
    plt.savefig('fig_217%d_chi%d.png'%(values_H[0],init[3]))
    
    plt.figure(2,figsize=(9,6))
    sigtime=np.ones(len(timenum3))*dt
    sigpos=np.ones(len(timenum3))*dx
    f=open('num3data.txt','w')
    f.write('%s%s%s%s'%(str(timenum3),'\n',str(sigtime),'\n'))
    f.write('%s%s%s%s'%(str(posnum3),'\n',str(sigpos),'\n'))
    f.close()
    plt.loglog(timenum3,posnum3,'ko')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    
    plt.show()
    
    return

def cond_limites(r,a,b,c):
    
    r[0],r[-1]=1,1                      #limit conditions
    r[1],r[-2]=r[1]-a[1]*r[0],r[-2]+c[-2]*r[-1]
    a[1],b[0],c[0]=0,1,0                #changing the matrix
    a[-1],b[-1],c[-2]=0,1,0
    
    return r,a,b,c

def compute_D(temp,init):
    chi=compute_chi(temp,init)
    D0,D1,D2=[np.zeros(nx) for i in range(3)]
    for j in range(-1,nx-1):            #using negative index to avoid IndexError
        D0[j]=(chi[j-1]+chi[j])
        D1[j]=-(chi[j-1]+2*chi[j]+chi[j+1])
        D2[j]=(chi[j]+chi[j+1])    
    return [D0,D1,D2]

def compute_chi(temp,init):
    chi=np.zeros(nx)
    for j in range(-1,nx-1):            #using negative index to avoid IndexError
        dtdx=(temp[j-1]-temp[j+1])/(2*dx)
        chi[j]=init[2]+init[3]/(1+init[0]*(dtdx**init[1]))
    return chi
    
def compute_abc(D):
    coeff=dt/(4*dx*dx)
    a=-coeff*D[0]                       #computing the matrix coefficients
    b=1-coeff*D[1]
    c=-coeff*D[2]
    return a,b,c

def compute_H(pos,values_H):
    H=np.zeros(nx)                      #computing the source term
    for i in range(nx):                 #gaussian profile
        H[i]=values_H[0]*np.exp(-pos[i]*pos[i]/(values_H[1]*values_H[1]))
#    for i in range(nx):                #step fct profile
#        if abs(pos[i])<values_H[1]:
#            H[i]=values_H[0]
    return H

def compute_r(r,temp,a,b,c,pos,values_H):
    r=crank_nicolson(r,temp,a,b,c)      #computingthe RHS vector
    H=compute_H(pos,values_H)   
    r+=(dt*H)
    return r
        
def crank_nicolson(r,u,a,b,c):          #matrix product
    r[0]=(2-b[0])*u[0]-c[0]*u[1]
    for i in range(1,nx-1):
        r[i]=-a[i]*u[i-1]+(2-b[i])*u[i]-c[i]*u[i+1]
    r[-1]=-a[-1]*u[-2]+(2-b[-1])*u[-1]
    return r

def fig2_15(init):              
    """Figure 2.15 from notes"""
    dtdx=np.linspace(-6,0,601)
    lines = ['k','k--','k-.']
    linecycler=cycle(lines)
    
#    plt.figure(figsize=(9,6))
    
    for chi1 in [8,2,0.5]:
        chi=init[2]+chi1/(1+init[0]*dtdx**init[1])
        Q=-chi*dtdx
        plt.plot(-dtdx,Q,next(linecycler),label=r'$\chi_1 = $%.1f'%chi1)
    
    plt.plot(-dtdx,10*(-dtdx),'r--')
    plt.plot(-dtdx,1*(-dtdx),'r--')
    plt.xlabel(r'$-dT/dx$')
    plt.ylabel(r'$Q$')
    plt.axis([0,6,0,6])
    plt.legend(shadow=True,fancybox=True)
    plt.text(0.65, 5.6, 'Mode L',color='r')
    plt.text(4, 3.6, 'Mode H',color='r')
    plt.text(1.5, .25, r'$\alpha = 1, \beta = 4, \chi_0 = 1$')
    plt.savefig('verif_fig_215.png')
    plt.show()
    return

def integrale_hamiltonien(pos,values_H):
    """for #1 (not really interesting)"""
    
    echantillon=np.linspace(2,22,11)
    for i in echantillon:
        values_H[0]=i
        H=compute_H(pos,values_H)
        integrale=trapeze(pos,H)
        plt.plot(i,integrale,'k.')
        #plt.plot(pos,H)
    plt.show()
    
    return

def trapeze(x,y):
    """numerical integration"""
    total=0
    for i in range(len(x)-1):
        dx=x[i+1]-x[i]
        dy=0.5*(y[i+1]+y[i])
        total+=(dx*dy)
    return total
        
    
def tridiag(a,b,c,r,u):
    """Tridiagonal solver from Press et. al. (1992), section 2.4
    
    a[n], b[n], c[n] are the three diagonals of the matrix
    r[n]is the RHS vector
    u[n] is the solution the the n step at input, solution to n+1 at output
    """
    gam=np.zeros(nx)                 #vecteur de travail
    bet=b[0]
    if bet == 0 : raise Exception("beta=0 1")
    u[0]=r[0]/bet
    for j in range(1,nx-1):          #decomposition LU
        gam[j]=c[j-1]/bet
        bet=b[j]-a[j]*gam[j]
        if bet == 0 : raise Exception("beta=0 2")
        u[j]=(r[j]-a[j]*u[j-1])/bet
    for j in range(nx-2,0,-1):       #backsubstitution
        u[j]-=gam[j+1]*u[j+1]
    return 0

#----------------------------Main----------------------------

pos,temp,init,values_H=cond_initiales()

#integrale_hamiltonien(pos,values_H)

#fig2_15(init)

i=0

confinement(pos,temp,init,values_H)

totaltime=time.time()-start
print("Total time : %d h %d m %d s"%(totaltime//3600,totaltime//60,totaltime%60))
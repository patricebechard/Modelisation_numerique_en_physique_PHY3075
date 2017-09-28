# -*- coding: utf-8 -*-
"""
Title : Hudgkin-Huxley Model resolution with Runge-Kutta Method (ODE)

Author : Patrice Bechard

Date : February 5th, 2017

Program that computes the Hudgkin-Huxley model for the transmission
of electric signal in the brain locally.

We use the 4th degree Runge-Kutta method with adaptive step to solve 
ordinary differential equations.
    
Electric potential is calculated in mV, time in ms, capacitance in μF cm^-2,
Electric current in μA cm^-2.
"""
############# MODULES

import numpy as np                          #for arrays and mathematical functions
import matplotlib.pyplot as plt             #for plots
import sys                                  #for debugging purposes
import time

plt.style.use('patrice')                    #custom graphics package

############# FONCTIONS

def find_equilibrium():
    """Function that finds the values of the N,H,M functions 
    and characteristic time for a given V.
    
    Using equations 1.65 and 1.66 to calculate xN, tauM, etc.
    """
    V=np.linspace(-100,100,10001)   #list of values for potential from -100 to 100       
    for i in range(len(V)):
    
        xN=alphaN(V[i])/(alphaN(V[i])+betaN(V[i]))
        xM=alphaM(V[i])/(alphaM(V[i])+betaM(V[i]))
        xH=alphaH(V[i])/(alphaH(V[i])+betaH(V[i]))
        tauN=1/(alphaN(V[i])+betaN(V[i]))
        tauM=1/(alphaM(V[i])+betaM(V[i]))
        tauH=1/(alphaH(V[i])+betaH(V[i]))
        
        if i==0:                            #set the first part of array
            x=np.array([[xN,xM,xH]])
            tau=np.array([[tauN,tauM,tauH]])
        else:                               #dynamically add values to array
            x=np.concatenate((x,[[xN,xM,xH]]),axis=0)
            tau=np.concatenate((tau,[[tauN,tauM,tauH]]),axis=0)
            
    return V,x,tau
    
def adaptive_step(nMax,tFin):
    """Function that performs an evolution of the H-H model in time
    for one situation using an adaptive step.
    
    Calls the Runke-Kutta function to solve the ODEs
    """
    #initiate default step and iterator
    eps=1.e-5                   #tolerance for RK adaptive step
    h=0.1                       #initial step
    nn=0                        #iterator for number of steps

    #initiate arrays to store time and position with equilibrium values
    t=np.array([0])                  
    u=np.array([[0.,x[index,0],x[index,1],x[index,2]]])
    dim=len(u[0])               #number of parameters to analyse
    
    while t[-1]<tFin and nn<nMax-1:
        u1=rungeKutta(t[-1],u[-1],h)        #full step    
        u2a=rungeKutta(t[-1],u[-1],h/2.)    #first half-step for adaptive step method
        u2=rungeKutta(t[-1],u2a,h/2.)       #second half-step
        delta=max(abs(u2[0]-u1[0]),abs(u2[1]-u1[1]),abs(u2[2]-u1[2]),abs(u2[3]-u1[3]))#eqn 1.42
        if delta>eps:
            h/=1.5
        else: 
            nn+=1
            t=np.append(t,t[-1]+h)
            u=np.append(u,u2)               #add new values to array
            u=u.reshape((nn+1,dim))         #and reshape it as a nn+1 by "dim" matrix
            if delta<=eps/2.:
                h*=1.5
    
    localmax=[]                             #empty array where we store max
    period=[]                               #same for when the max happen
    for i in range(1,len(t)-2):             #in case of IndexError
        if (u[i-1][0]<u[i][0]>=u[i+1][0]):  #else value is not a max
                localmax.append(u[i,0])   
                period.append(t[i])
    localmax=localmax[1:]                   #first max is not good
    period=period[1:]
    meanmax=np.mean(localmax)               #value of mean max
    freq=[(1000/(period[i+1]-period[i])) for i in range(len(period)-1)] #and frequency
    meanfreq=np.mean(freq)                  #mean value of frequency
    
    return [meanmax,meanfreq]
  
def slope(t0,uu):
    """Function that contains right side of the differential equations to solve """
    
    #constants for the problem (see table 1.2)
    VK,VNA,VL = -12., 115., 10.6  #resting potentials (constants)
    GK,GNA,GL = 36., 120., 0.3    #conductance coefficients (constants)
    CM=1                          #conductance membranaire (constant)
    
    global ione, itwo, second     #variables from main() can be used here
    
    # Current applied to the membrane
    if second<0:                  #current is constant fot the whole time
        courantA=ione           
    elif second==0:               #current applied for 1ms and never again
        courantA=ione if t0<1 else 0  
    else:                         #two pulses happening at t=0 and t=second
        if t0<=1:               
            courantA=ione
        elif second<=t0<=second+1:
            courantA=itwo
        else:
            courantA=0
            
    ###slopes (eq. 1.68)
    gV=(1./CM)*(courantA-GK*(uu[1]**4)*(uu[0]-VK)-GNA*(uu[2]**3)*uu[3]*(uu[0]-VNA)-GL*(uu[0]-VL))
    gn=alphaN(uu[0])*(1-uu[1])-betaN(uu[0])*uu[1]
    gm=0                        #comment following line for 6)
    gm=alphaM(uu[0])*(1-uu[2])-betaM(uu[0])*uu[2]
    gh=alphaH(uu[0])*(1-uu[3])-betaH(uu[0])*uu[3]

    return np.array([gV,gn,gm,gh])
    
def rungeKutta(t0,uu,h):
    """Function that performs the Runge-Kutta method for solving ODEs"""
        
    g1=slope(t0,uu)                  #slope1
    g2=slope(t0+h/2,uu+(h/2)*g1)     #slope2
    g3=slope(t0+h/2,uu+(h/2)*g2)     #slope3
    g4=slope(t0+h,uu+h*g3)           #slope4
    return uu+(h/6)*(g1+2*g2+2*g3+g4)
    
# Alpha and Beta functions (1.62, 1.63, 1.64)

def alphaH(V): return 0.07*np.exp(-V/20.)  
def alphaM(V): return (2.5-0.1*V)/(np.exp(2.5-0.1*V)-1.)
def alphaN(V): return (0.1-0.01*V)/(np.exp(1.-0.1*V)-1.)
def betaH(V):  return 1./(np.exp(3.-0.1*V)+1.)
def betaM(V):  return 4.*np.exp(-V/18.)
def betaN(V):  return 0.125*np.exp(-V/80.)      

############### MAIN

t_init=time.time()          #we start the time when the program begins

V,x,tau=find_equilibrium()          #we find the equilibrium values
index=int(np.where(V==0.)[0])       #index where is V=0 for
"""    
#Two graphs with same x axis
f, axarr = plt.subplots(2, sharex=True,figsize=(5,8))

#first graph is for x_eq
axarr[0].plot(V, x[:,0],':',label='n')  
axarr[0].plot(V, x[:,1],'--',label='m')
axarr[0].plot(V, x[:,2],'-.',label='h')
axarr[0].set_ylabel(r"$x_{eq}$ (V)")
axarr[0].axis([-100,100,-0.1,1.1])
axarr[0].legend(loc=5,fancybox=True,shadow=True)
 
#second graph is for tau
axarr[1].plot(V,tau[:,0],':',label='n')
axarr[1].plot(V,tau[:,1],'--',label='m')
axarr[1].plot(V,tau[:,2],'-.',label='h')
axarr[1].set_ylabel(r"$\tau$ (V) [ms]")
axarr[1].set_xlabel("V [mV]")
axarr[1].axis([-100,100,0,10])
axarr[1].legend(fancybox=True,shadow=True)

f.subplots_adjust(hspace=0.05)          #reduce space in between
f.savefig("verif_equilibre.png")        #save figure
"""
    
# Initialize loop contraints
    
nMax=10000                           #max number of steps
tFin=100.                            #max time
courantA=np.array([6.0,10])
itwo=10                             #intensity of second pulse (#3)
#time of second pulse (<1 if current is constant, 0 if only one pulse)
secondpulse=np.array([-1])          
frequency=[]                        #empty array for mean frequency
maximum=[]                          #and mean max for one simulation

for ione in courantA:               #comment if we test on second pulses
    second=secondpulse[0]
#for second in secondpulse:         #comment if we test on multiple currents
#    ione=courantA[0]
    
    results=adaptive_step(nMax,tFin)
    
    maximum.append(results[0])
    frequency.append(results[1])    
    
"""
rangeI=np.linspace(10,20,11)          #values found for min time vs current
rangeI=np.append(rangeI,np.array([21,22,24,26,28,30]))
data=np.array([14.524,14.06,13.66,13.304,12.981,12.689,12.420,12.173,\
11.944,11.731,11.531,11.345,11.169,10.848,10.560,10.303,10.069])
plt.loglog(rangeI,data,'*')

logx = np.log(rangeI)                   #for the fit
logy = np.log(data)
coeffs = np.polyfit(logx,logy,deg=2)
poly = np.poly1d(coeffs)
print(poly)

x=np.linspace(0,10000,10001)
yfit = lambda rangeI: np.exp(poly(np.log(rangeI)))
plt.loglog(rangeI,yfit(rangeI))
plt.show()
plt.plot(rangeI,yfit(rangeI),':',label=r'$\exp{(-0.001844x^2-0.3225x+3.428)}$')
plt.plot(rangeI,data,'o',label='Données')

plt.xlabel(r"Courant appliqué [$\mu A/cm^2$]")
plt.ylabel(r"$\Delta t_{seuil}$ [ms]")
plt.legend(fancybox=True,shadow=True)
plt.show()
"""

"""
#graph with different currents plotted
plt.axis([0,tFin,-20,120])
#plt.axis([0,tFin,-25,150])
plt.fill_between([0,1],-20,120,facecolor="0.75",alpha=0.5)
plt.plot([0,tFin],[0,0],':',c="0.66",linewidth=0.5)
plt.legend(fancybox=True,shadow=True)
plt.xlabel("V [mV]")
plt.ylabel("t [ms]")
plt.savefig("verif_diff_courant.png")
#plt.savefig("verif_progr_courant_cst.png")
"""

"""
#graph with 2 different y axis
f, ax1=plt.subplots()
ax2=ax1.twinx()

#plot data
ax1.plot(t,u[:,0],'k')
ax2.plot([0],[0],'k',label='V') #to solve the legend problem 
ax2.plot(t,u[:,1],':',label='n')
ax2.plot(t,u[:,2],'--',label='m')
ax2.plot(t,u[:,3],'-.',label='h')
#grey space at beginning
ax2.fill_between([0,1],-0.2,1.2,facecolor="0.75",alpha=0.5)

#plot horizontal lines as in fig 1.15
for i in [1,2,3]:
    ax2.plot([0,25],[u[0,i],u[0,i]],':',c="0.66",linewidth=0.5)

#set plot parameters
ax1.set_xlabel("t [ms]")
ax1.set_ylabel("V [mV]")
ax2.set_ylabel("m, n, h")
ax1.axis([0,25,-20,120])
ax2.axis([0,25,-0.195,1.195])
ax2.legend(fancybox=True,shadow=True)
ax2.text(2,1.05,r"$0<t<1$ms $I_a=7\mu A/cm^2$")

f.savefig("verif_V_and_x.png")
"""
print("ELAPSED : ",time.time()-t_init)

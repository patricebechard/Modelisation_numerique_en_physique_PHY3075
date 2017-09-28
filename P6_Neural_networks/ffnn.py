# -*- coding: utf-8 -*-
# @Author: Patrice Bechard <patricebechard>
# @Date:   2017-03-02T01:52:26-05:00
# @Email:  bechardpatrice@gmail.com
# @Last modified by:   Patrice
# @Last modified time: 2017-04-13T00:15:20-04:00

#-----------------Modules and global variables---------------

import numpy as np
import sys
import matplotlib.pyplot as plt

ni,nh,no=12,6,1             #nb of input units, middle and output
wih=np.zeros([ni,nh])       #weigth of the connections between input and middle
who=np.zeros([nh,no])       #weight of the connections between middle and output
ivec=np.zeros(ni)           #signal for input
sh=np.zeros(nh)             #signal for middle neurons
so=np.zeros(no)             #signal for output neurons

err=np.zeros(no)            #error signal for output neurons
deltao=np.zeros(no)         #error gradient for output neurons
deltah=np.zeros(nh)         #error gradient for middle neurons
eta=0.1                     #learning parameter

#-----------------------Functions----------------------------

def actv(a):
    """sigmoidal activation function (Eq. 6.5)"""
    return 1./(1.+np.exp(-a))

def dactv(s):
    """derivative of the sigmoidal activation function (Eq. 6.19)"""
    return s*(1.-s)

def ffnn(ivec):
    """
    Feed-Forward Neural Network function
    calculates the output signal
    """
    for ih in range(nh):        #from input layer to middle layer
        sum=0.
        for ii in range(ni):
            sum+=wih[ii,ih]*ivec[ii]    #Eq. 6.1
        sh[ih]=actv(sum)                #Eq. 6.2
    for io in range(no):        #from middle layer to output layer
        sum=0.
        for ih in range(nh):
            sum+=who[ih,io]*sh[ih]      #Eq. 6.3 with b_1=0
        so[io]=actv(sum)                #Eq. 6.4
    return

def backprop(err):
    """Backpropagation of the error signal and weight adjusting"""
    for io in range(no):        #from output layer to middle layer
        deltao[io]=err[io]*dactv(so[io])#Eq. 6.20
        for ih in range(nh):
            who[ih,io]+=eta*deltao[io]*sh[ih]   #Eq. 6.17 for the wHO
    for ih in range(nh):        #from middle layer to input layer
        sum=0.
        for io in range(no):
            sum+=deltao[io]*who[ih,io]
        deltah[ih]=dactv(sh[ih])*sum   #Eq. 6.21
        for ii in range(ni):
            wih[ii,ih]+=eta*deltah[ih]*ivec[ii] #Eq. 6.17 for the wIH
    return

def randomize(n):
    dumvec=np.zeros(n)
    for k in range(n):
        dumvec[k]=np.random.uniform()   #random numbers array
    return np.argsort(dumvec)           #return rank table

def init_set(f,nSample):
    """Initialize training set and answers for supervized learning"""
    tset = np.zeros([nSample,ni])            #training set empty array
    oset = np.zeros([nSample,no])           #answers for training set
    for i in range(nSample):
        temp = f.readline().strip().split()
        tset[i] = [float(elem) for elem in temp[:-1]]
        oset[i] = bool(int(temp[-1]))                 #higgs boson present or not
    return tset,oset
#------------------------MAIN--------------------------------
#read or initialize training set here
f=open('bitstrings_train.txt')
i=0
nset = int(f.readline().strip())
tvec,ovec = init_set(f,nset)
niter=1000                              #number of training iterations
oset=np.zeros([nset,no])                #output for the training set
tset=np.zeros([nset,ni])                #input vector fhr the training set
rmserr=np.zeros(niter)                  #training rms error


#initializing random weights
for ii in range(ni):                    #input/middle weights
    for ih in range(nh):
        wih[ii,ih]=np.random.uniform(-0.5,0.5)
for ih in range(nh):                    #middle/output weights
    for io in range(no):
        who[ih,io]=np.random.uniform(-0.5,0.5)

for iter in range(niter):               #loop over training iterations
    sum=0.
    rvec=randomize(nset)                #randomize members
    print(rvec)
    sys.exit()
    for itrain in range(nset):          #loop over training set
        itt=rvec[itrain]                #chosen member
        ivec=tset[itt,:]                #and its associate input vector
        ffnn(ivec)                      #compute output signal
        for io in range(no):            #error signals on output neurons
            err[io]=oset[itt,io]-so[io]
            sum+=err[io]**2             #squared to calculate rms
        backprop(err)                   #backpropagation

    rmserr[iter]=np.sqrt(sum/nset/no)   #rms error for this iteration
plt.plot([x for x in range(niter)],rmserr,'k.')
plt.show()

#insert test phase here
g=open('bitstrings_test.txt')
ntest=0
for lines in g:
    if lines!='EOF':
        ntest+=1
testset=np.zeros([ntest,ni])
i=0
for lines in g:
    if lines!='EOF':
        elem=[int(x) for x in lines.strip()]
        testset[ntest,:]=elem
        i+=1

for itest in range(ntest):
    ivec=testset[itest,:]
    ffnn(ivec)
    #print(so)

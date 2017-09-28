# @Author: Patrice Bechard <Patrice>
# @Date:   2017-04-12T11:27:59-04:00
# @Email:  bechardpatrice@gmail.com
# @Last modified by:   Patrice
# @Last modified time: 2017-04-12T23:57:42-04:00

import numpy as np
from scipy.special import expit
import time

nNeurons = [12,6,1]
nLayers = len(nNeurons)
eta = 0.1
deltao=np.zeros(nNeurons[2])         #error gradient for output neurons
deltah=np.zeros(nNeurons[1])         #error gradient for middle neurons

wMatrix = [[] for i in range(nLayers)]  #list of weight matrices
for i in range(nLayers):
    if i == nLayers-1:
        wMatrix[i] = [[1]]
    else:
        wMatrix[i] = np.random.uniform(-0.5,0.5,[nNeurons[i],nNeurons[i+1]])

nodeValues = [np.zeros(nNeurons[i]) for i in range(nLayers)]

wih = wMatrix[0]
who = wMatrix[1]
sh = nodeValues[1]
so = nodeValues[2]

input = [1,0,4,6,4,8,5,7,5,10,4,63]
nodeValues[0] = input

def actv(a):
    return expit(a)

def dactv(s):
    """derivative of the sigmoidal activation function (Eq. 6.19)"""
    return s * (1.-s)

def ffnn():
    """vectorized Feed-Forward neural network function"""
    for i in range(nLayers-1):
        nodeValues[i+1] = actv(nodeValues[i] @ wMatrix[i])
    return

def ffnn2(ivec):
    """
    Feed-Forward Neural Network function
    calculates the output signal
    """
    for ih in range(nNeurons[1]):        #from input layer to middle layer
        sum=0.
        for ii in range(nNeurons[0]):
            sum+=wih[ii,ih]*ivec[ii]    #Eq. 6.1
        sh[ih]=actv(sum)                #Eq. 6.2
    for io in range(nNeurons[2]):        #from middle layer to output layer
        sum=0.
        for ih in range(nNeurons[1]):
            sum+=who[ih,io]*sh[ih]      #Eq. 6.3 with b_1=0
        so[io]=actv(sum)                #Eq. 6.4
    return

def backprop(err):
    """Backpropagation of the error signal and weight adjusting"""
    for i in reversed(range(1,nLayers)):
        err = np.dot(wMatrix[i],err) * dactv(nodeValues[i])
        wMatrix[i-1] += eta * (np.outer(nodeValues[i-1],err))
        print(wMatrix[i-1])
    return

def backprop2(err):
    """Backpropagation of the error signal and weight adjusting"""
    for io in range(nNeurons[2]):        #from output layer to middle layer
        deltao[io]=err[io]*dactv(so[io])#Eq. 6.20
        for ih in range(nNeurons[1]):
            who[ih,io]+=eta*deltao[io]*sh[ih]   #Eq. 6.17 for the wHO
    print(who)
    for ih in range(nNeurons[1]):        #from middle layer to input layer
        sum=0.
        for io in range(nNeurons[2]):
            sum+=deltao[io]*who[ih,io]
        deltah[ih]=dactv(sh[ih])*sum   #Eq. 6.21
        for ii in range(nNeurons[0]):
            wih[ii,ih]+=eta*deltah[ih]*input[ii] #Eq. 6.17 for the wIH
    print(wih)
    return

ffnn()
sv = -nodeValues[-1]
err = sv
backprop(err)

err = sv
backprop2(err)

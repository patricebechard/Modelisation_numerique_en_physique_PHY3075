# @Author: Patrice Bechard <patricebechard>
# @Date:   2017-04-12T00:31:00-04:00
# @Email:  bechardpatrice@gmail.com
# @Last modified by:   Patrice
# @Last modified time: 2017-04-13T10:47:47-04:00
#
# Neural Network (PHY3075 - Chapter 6)

#--------------------------- Modules ------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit         #sigmoid
import sys

filetraining = 'trainingset.txt'                 #input file
nTraining = 1500

filetest = filetraining
nTest = 2000-nTraining
#filetest = 'examset.txt'
#nTest = 1000

nIter = 2000                            #number of iterations
actv_fct = 0                            #which activation function to choose
nNeurons = [13,6,1]                     #number of neurons by layer
nLayers = len(nNeurons)                 #number of neuron layers
eta = 0.001                               #learning parameter

wMatrix = [[] for i in range(nLayers)]  #list of weight matrices

biasVector = np.array([np.zeros(nNeurons[i]) for i in range(nLayers)]) #list of bias vectors

nodeValues = [np.zeros(nNeurons[i]) for i in range(nLayers)]

#-------------------------- Functions -----------------------------------------
def actv(a):
    if actv_fct == 0:
        """sigmoidal activation function (eq. 6.5)"""
        return expit(a)
    elif actv_fct == 1:
        """hyperbolic tangent activation function"""
        return np.tanh(a)
    elif actv_fct == 2:
        """RELU activation function"""
        return a * (a>0)

def dactv(s):
    if actv_fct == 0:
        """derivative of the sigmoidal activation function (eq. 6.5)"""
        return s * (1.-s)
    elif actv_fct == 1:
        """derivative of the hyperbolic tangent activation function"""
        return 1 - s*s
    elif actv_fct == 2:
        """derivative of the RELU activation function"""
        return s > 0

def ffnn():
    """vectorized Feed-Forward neural network function"""
    for i in range(nLayers-1):
        if i == 0:
            nodeValues[i+1] = actv(biasVector[i+1] + (nodeValues[i] @ wMatrix[i]))
        else:
            nodeValues[i+1] = biasVector[i+1] + (nodeValues[i] @ wMatrix[i])

def backprop(err):
    """Backpropagation of the error signal and weight adjusting"""
    for i in reversed(range(1,nLayers)):
        err = np.dot(wMatrix[i],err) * dactv(nodeValues[i])
        wMatrix[i-1] += eta * (np.outer(nodeValues[i-1],err))

def init_set(f,nSample):
    """Initialize training set and answers for supervized learning"""
    tset = np.zeros([nSample,nNeurons[0]])            #training set empty array
    oset = np.zeros([nSample,nNeurons[-1]])           #answers for training set
    for i in range(nSample):
        temp = f.readline().strip().split()
        tset[i] = [float(elem) for elem in temp[:-1]]
        oset[i] = bool(int(temp[-1]))                 #higgs boson present or not
    return tset,oset

def training():
    """Training phase"""
    f = open(filetraining)
    f.readline()                                      #1st line is trash
    tset,oset = init_set(f,nTraining)
    f.close()
    for i in range(nLayers):
        if i == nLayers-1:
            wMatrix[i] = [[1]]
        else:
            wMatrix[i] = np.random.uniform(-0.5,0.5,[nNeurons[i],nNeurons[i+1]])
    rmserr = []
    for i in range(nIter):                          #loop over all iterations
        if i%100 == 0 and i != 0:
            print(i)
            test(i)
        sum = 0
        order = np.random.permutation(nTraining)
        for j in range(nTraining):                    #loop over all element of training set
            nodeValues[0] = actv(tset[order[j]])
            ffnn()
            err = oset[order[j]]-nodeValues[-1]
            sum += np.linalg.norm(err)
            backprop(err)
        rmserr.append(np.sqrt(sum/(nTraining*nNeurons[-1])))   #rms error for this iteration
        #global eta
        #eta = eta - 0.0005
    plt.plot(np.arange(nIter),rmserr,'k.')
    plt.show()

def save_network():
    """Save network after training"""
    g = open('saved_network.txt','w')
    numNodes = ''
    for i in range(nLayers):
        numNodes += str(nNeurons[i])
        if i != nLayers-1:
            numNodes += ' '
        else:
            numNodes += '\n'
    g.write(numNodes)
    for i in range(nLayers):
        for line in wMatrix[i]:
            for elem in line:
                g.write(str(elem)+' ')
        if i != nLayers-1:
            g.write('\n')
    g.close()

def load_network():
    g = open('saved_network.txt')
    nNeurons = g.readline().strip().split()
    nNeurons = [int(i) for i in nNeurons]
    nLayers = len(nNeurons)
    for i in range(nLayers):
        data = g.readline().strip().split()
        data = [float(i) for i in data]
        if i != nLayers-1:
            data = np.array([data]).reshape(nNeurons[i],nNeurons[i+1])
        wMatrix[i] = data
    g.close()

def test(i = None):
    f = open(filetest)
    f.readline()
    for i in range(nTraining):
        f.readline()
    testset,oset = init_set(f,nTest)
    f.close()
    if i == None:
        g = open('results.txt','w')
    ratio = 0
    VFNP = np.zeros([2,2])
    for i in range(nTest):
        result = False
        nodeValues[0] = actv(testset[i])
        ffnn()
        if float(nodeValues[-1]) > 0.5:
            result = True
        if result == oset[i]:
            if result == True:  #vrai positif
                VFNP[1][1] += 1
            else:               #vrai negatif
                VFNP[0][0] += 1
            ratio += 1
            if i == None:
                g.write('1\n')
        else:
            if result == True:  #faux positif
                VFNP[0][1] += 1
            else:               #faux negatif
                VFNP[1][0] += 1
            if i == None:
                g.write('0\n')
    if i == None:
        g.close()
    print("SUCCESS RATIO : %f"%(ratio/nTest))
    print(VFNP)

#----------------------------- Main -------------------------------------------
if 'train' in sys.argv:
    training()
    save_network()
if 'test' in sys.argv:
    if not 'train' in sys.argv:
        load_network()
    test()

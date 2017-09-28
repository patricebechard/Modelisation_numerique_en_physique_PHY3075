import os	
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'          #no warning messages

import tensorflow as tf                         # ML library
import numpy as np
import copy
import sys
import matplotlib.pyplot as plt

#-------------------------------Global variables--------------------------------

higgs = True                            #True : Higgs detection, False : 12 bits string
exam = True                             #True : applying on exam set
actv = 2                                #0: logistic, 1: tanh, 2: ReLU
optim_function = 0        #0: AdamOptimizer, 1: Gradient descent, 2: RMS Prop, 3: AdaGrad
nIter = 50000
batch_size = 50

RANDOM_SEED = 42                        
tf.set_random_seed(RANDOM_SEED)

if higgs:                               #we train the network on the higgs data
    filetraining = 'trainingset.txt'    #training set
    nTraining = 1500
    nTest = 2000 - nTraining
    nNeurons = [13,10,8,2]              #architecture of network
    learningrate = 0.00001
else:                                   #we train the network on strings of bits
    filetraining = 'bitstrings_train.txt' #training set
    nTraining = 1000
    filetest = 'bitstrings_test.txt'      #test set
    nTest = 1000

    nNeurons = [12,6,2]                 #architecture of network
    learningrate = 0.01

nLayers = len(nNeurons)

#--------------------------------Functions--------------------------------------

def init_weights(shape):
    """ Weight initialization """
    return tf.Variable(tf.random_normal(shape))

def model(x,wMatrix,p_keep_input,p_keep_hidden):
    """Propagation of input through the neural network"""
    x = tf.nn.dropout(x,p_keep_input)   #we keep only with some probability
    h = [[] for i in range(nLayers-1)]  
    h[0] = copy.copy(x)                 #sentinel
    for i in range(1,nLayers-1):
        if actv == 0:                   #logistic
            h[i] = tf.sigmoid(tf.matmul(h[i-1],wMatrix[i-1]))
        elif actv == 1:                 #tanh
            h[i] = tf.tanh(tf.matmul(h[i-1],wMatrix[i-1]))
        elif actv == 2:                 #relu
            h[i] = tf.nn.relu(tf.matmul(h[i-1],wMatrix[i-1]))
        h[i] = tf.nn.dropout(h[i],p_keep_hidden)   #probability of keeping change

    return tf.matmul(h[-1],wMatrix[-1]) #return output

def init_bits(file,nSample):
    """Initialization of 12 bits data from txt file"""
    f = open(file)
    f.readline()
    tset = np.zeros([nSample,nNeurons[0]])
    oset = np.zeros([nSample,nNeurons[-1]])
    for i in range(nSample):
        temp = f.readline().strip().split()
        tset[i] = np.array(list(temp[0]),dtype=int)
        if int(temp[-1]):              # consecutive bit strings
            oset[i] = [0,1]
        else:
            oset[i] = [1,0]
    return tset,oset

def init_set(f,nSample,exam=False):
    """Initialize training set and answers for supervized learning"""
    tset = np.zeros([nSample,nNeurons[0]])            #training set empty array
    oset = np.zeros([nSample,nNeurons[-1]])           #answers for training set
    for i in range(nSample):
        temp = f.readline().strip().split()
        if exam:
            tset[i] = [float(elem) for elem in temp]
        else:
            tset[i] = [float(elem) for elem in temp[:-1]]
            temp[-1]=int(temp[-1])
            if temp[-1]:
              oset[i] = [1,0]
            else:
              oset[i] = [0,1]
    #normalisation of values
    #for i in range(13):
    #    tset[:,i] /= abs(max(max(tset[:,i]), min(tset[:,i]), key=abs))
    return tset,oset

def main(optim_function,learningrate):
    if higgs:
        """Higgs detection"""
        f = open(filetraining)
        f.readline()
        train_x, train_y = init_set(f,nTraining)     #initializing data
        test_x, test_y = init_set(f,nTest)          
        if exam:
            fileexam = 'examset.txt'                 #exam set
            g = open(fileexam)
            g.readline()    
            nExam = 1000
            exam_x, exam_y = init_set(g,nExam,exam=True) #init exam data
    else:
        """5 straight in 12 bits"""
        train_x, train_y = init_bits(filetraining,nTraining)
        test_x, test_y = init_bits(filetest,nTest)
    
    costtrain, costtest = [], []                    #empty arrays to plot later

    # Symbols
    x = tf.placeholder("float", [None,nNeurons[0]])
    y = tf.placeholder("float", [None, nNeurons[-1]])

    # Weight initializations
    wMatrix = [init_weights([nNeurons[i],nNeurons[i+1]])
                                        for i in range(nLayers-1)]

    # Forward propagation
    p_keep_input = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")
    py_x = model(x,wMatrix,p_keep_input,p_keep_hidden)

    # Backward propagation
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits   
                                                (labels=y, logits=py_x))
    #cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
    #                                           (labels=y, logits=py_x))

    if optim_function == 0:     #adam optimizer
        updates = tf.train.AdamOptimizer(learningrate).minimize(cost)
    elif optim_function == 1:   #gradient descent optimizer
        updates = tf.train.GradientDescentOptimizer(learningrate).minimize(cost)
    elif optim_function == 2:   #RMSProp optimizer
        updates = tf.train.RMSPropOptimizer(learningrate).minimize(cost)
    elif optim_function == 3:   #AdaGrad optimizer
        updates = tf.train.AdagradOptimizer(learningrate).minimize(cost)
    predict = tf.argmax(py_x,1)

    with tf.Session() as sess:			#launch session
        sess.run(tf.global_variables_initializer())	#init all variables
        #saver = tf.train.Saver()

        for epoch in range(nIter+1):
            # Train with each example
            order = np.random.permutation(np.arange(nTraining)) #shuffling data
            i = 0
            while i < len(order):           #updating neural net
                start = i
                end = i + batch_size
                X = np.array(train_x[order[start:end]])
                Y = np.array(train_y[order[start:end]])
                sess.run(updates, feed_dict={x: X, y: Y,
                                p_keep_input: 1, p_keep_hidden: 1})
                i += batch_size
            
            if epoch % 100 == 0:            #showing relevant info
                results_train = sess.run(predict, feed_dict={x: train_x,
                                     p_keep_input: 1., p_keep_hidden: 1.})
                results_test = sess.run(predict, feed_dict={x: test_x,
                                     p_keep_input: 1., p_keep_hidden: 1.})
                train_accuracy = np.mean(np.argmax(train_y, axis=1) == results_train)
                test_accuracy  = np.mean(np.argmax(test_y, axis=1) == results_test)
                results_matrix = np.zeros([2,2])
                for k in range(nTraining):
                    if np.argmax(train_y[k]):
                        if results_train[k]:  #vrai positif
                            results_matrix[1][1] += 1
                        else:                 #faux negatif
                            results_matrix[1][0] += 1
                    else:
                        if results_train[k]:  #faux positif
                            results_matrix[0][1] += 1
                        else:                 #faux negatif
                            results_matrix[0][0] += 1

                print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
                     % (epoch , 100. * train_accuracy, 100. * test_accuracy))
                costing = sess.run(tf.reduce_mean(cost),feed_dict=
                    {x: train_x, y: train_y,p_keep_input: 1., p_keep_hidden: 1.})
                print("cost function value : ",costing)
                print(results_matrix)

                costtrain.append(costing)           #to plot later
                costtest.append(sess.run(tf.reduce_mean(cost),feed_dict={x: test_x,
                    y: test_y, p_keep_input: 1., p_keep_hidden: 1.}))

            if epoch == nIter and exam:         #evaluating exam set
                results_train = sess.run(predict, feed_dict={x: train_x,
                                     p_keep_input: 1., p_keep_hidden: 1.})
                results_test = sess.run(predict, feed_dict={x: exam_x,
                                             p_keep_input: 1., p_keep_hidden: 1.})
                train_accuracy = np.mean(np.argmax(train_y, axis=1) == results_train)
                print("Epoch = %d, train accuracy = %.2f%%"
                     % (1, 100. * train_accuracy))
                costing = sess.run(tf.reduce_mean(cost),feed_dict=
                    {x: train_x, y: train_y,p_keep_input: 1., p_keep_hidden: 1.})
                print("cost function value : ",costing)
                print(results_matrix)
                h = open('results.txt','w')     #saving results
                for i in range(nExam):
                    h.write(str(results_test[i])+'\n')

        #saver.save(sess,"/tmp/higgs_ffnn.ckpt")
    return costtrain,costtest

#--------------------------MAIN-------------------------------------------------

ls = ['r:','g--','b-.','k-']        # line styles
ms = ['rs','g^','bo','k*']          # marker styles
#plotlabels = ['Adam','Gradient Descent','RMSProp','AdaGrad']
plotlabels = ['sigmoid','tanh','ReLU']
for i in [2]:
    costtrain,costtest = main(i,0.001)
    plt.loglog(np.arange((nIter-1)//100 + 1)*100,costtrain,ls[i],label=plotlabels[i])
    plt.loglog(np.arange((nIter-1)//100 + 1)*100,costtest,ms[i],mfc='none') 
plt.legend(fancybox=True,shadow=True)
plt.xlabel("Itérations Entraînement")
plt.ylabel("Valeur de la fonction de coût")
plt.axis([100,nIter,0.1,10])
plt.savefig("cost_optimizer.png")
    
    #plt.show()
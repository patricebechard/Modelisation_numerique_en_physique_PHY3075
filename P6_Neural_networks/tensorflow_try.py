#
# Patrice Bechard
#
# Tensor Flow first try
#

#from the tensorflow tutorial at https://www.tensorflow.org/get_started/get_started


import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


#----------------------------test 1---------------------------------------------
hello = tf.constant('Hello, Tensorflow!')
sess = tf.Session()
print(sess.run(hello))

#--------------------getting started with Tensorflow----------------------------
node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)                        #already a float32 by default
print(node1,node2)

#output : Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)

sess = tf.Session()
print(sess.run([node1,node2]))

#output : [3.0, 4.0]

node3 = tf.add(node1,node2)
print(node3)                    #output : Tensor("Add:0", shape=(), dtype=float32)
print("sess.run(node3) : ",sess.run(node3)) #output : sess.run(node3):  7.0

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b                  #shorter than tf.add(a,b)

#on peut run la session avec les valeurs dans un dictionnaire

print(sess.run(adder_node,{a: 3., b: 4.5}))             #output : 7.5
print(sess.run(adder_node,{a: [1,3],b: [2,4]}))         #adding tensors, output : [3. 7.]

add_and_triple = adder_node * 3
print(sess.run(add_and_triple,{a: 3, b:4.5}))           #output : 22.5

W = tf.Variable([.3],tf.float32)
b = tf.Variable([-.3],tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model,{x: [1,2,3,4]}))   #output : [ 0.          0.30000001  0.60000002  0.90000004]

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))    #output : 23.66

#les parametres W=-1 et b=1 sont les parametres optimaux, on peut les assigner

fixW = tf.assign(W,[-1.])
fixb = tf.assign(b,[1.])
sess.run([fixW,fixb])                       #on l'assigne pour vrai
print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))  #output : 0.0

#le but du ML est de trouver ces parametres pour minimiser loss, pas de le deviner!

#tf.train
#on fait la descente de gradient pour minimiser la fonction loss

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)  #reset values to incorrect defaults
for i in range(1000):
    sess.run(train, {x:[1,2,3,4],y:[0,-1,-2,-3]})

print(sess.run([W,b]))
#output : [array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32)]
#we found something close!

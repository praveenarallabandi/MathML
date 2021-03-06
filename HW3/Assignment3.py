
# Code by Praveena Kondepudi 
#VID-V00933455

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from keras.datasets import mnist
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def accuracy(predictions, labels): 
    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) 
    accu = (100.0 * correctly_predicted) / predictions.shape[0] 
    return accu 

#loaded data from keras

(x_train, y_train), (x_test, y_test)= tf.keras.datasets.mnist.load_data()

# Filtered dataset with class values 4 and 5 for Vid: V00933455

train_filter_class= np.where((y_train == 5) | (y_train == 4))
test_filter_class= np.where((y_test == 5) | (y_test == 4))
x_train, y_train = x_train[train_filter_class], y_train[train_filter_class]
x_test, y_test = x_test[test_filter_class], y_test[test_filter_class]

#reshaping train and test data
x_train=x_train.astype(dtype='float32')
x_test=x_test.astype(dtype='float32')
y_train=y_train.astype(dtype='float32')
y_test=y_test.astype(dtype='float32')
y_train=y_train.reshape(y_train.shape[0],1)
y_test=y_test.reshape(y_test.shape[0],1)

x_train = x_train.reshape(x_train.shape[0],784) / 255
x_test = x_test.reshape(x_test.shape[0],784) / 255

#changing classes to -1's and 1's

for j in range(y_train.shape[0]):
        if y_train[j] == 4:
                y_train[j] = 1
        else:
           y_train[j] = -1

for j in range(y_test.shape[0]):
        if y_test[j] == 4:
                y_test[j] = 1
        else:
                y_test[j] = -1

## we use the dataset with x_train being the matrix "n by fCnt"
##   with samples as rows, and the  features as columns
## y_train is the true value of dependent variable, we have it as a matrix "n by 1"

n_train = x_train.shape[0]
fCnt = x_train.shape[1]
print(x_train.shape[0])
print(x_train.shape[1])
#### START OF LEARNING

n_epochs = 100

batch_size = 128

w_best = tf.Variable(tf.zeros([784, 10]))

b_best = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, 784])

y = tf.placeholder(dtype = tf.float32,name = 'y')

logits = tf.matmul(x,w_best) + b_best
model = tf.nn.softmax(logits)
cross_entropy = -tf.reduce_mean(y * tf.log(model))

# optimizer
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)



# TRIED WITH BELOW LOSS BUT WAS GETTING AND ERROR SO USING THE BELOW LOSS CALCULATION
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_train, logits=logits))
loss = tf.square(y - model)

#risk over all samples (a number)
risk = tf.reduce_mean(loss)

# create a tensorflow session and initialize the variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#start the iterations of training
#1 epoch == all data samples were presented
for i in range(0,n_epochs):
    #if dataset is large, we want to present it in chunks (called micro-batches)
    for j in range(0,n_train,batch_size):
        jS = j
        jE = min(n_train,j + batch_size)
        x_batch = x_train[jS:jE,:]
        y_batch = y_train[jS:jE,:]
        #do a step of gradient descent on the micro-batch
        _,best_risk, predBatchY = sess.run([train_step, risk, model],feed_dict = {x: x_batch, y: y_batch})
       
    # training done in this epoch
    # but, just so that the user can monitor progress, try best w,b on full test set
    y_pred,curr_w,curr_b = sess.run([model, w_best, b_best], feed_dict = {x: x_test, y: y_test})
    # calculate and print Mean Squared Error
    MSE = np.mean(np.mean(np.square(y_pred - y_test),axis = 1),axis = 0)
    print(MSE)
print('Training Completed')
print(np.transpose(curr_w))
print(curr_b)
accu = accuracy(y_pred, y_test)
print(accu)

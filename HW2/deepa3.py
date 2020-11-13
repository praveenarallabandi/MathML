import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import tensorflow as tf
from keras.datasets import mnist

#loaded data from keras

(x_train, y_train), (x_test, y_test)= tf.keras.datasets.mnist.load_data()

# Filtered dataset with class values 0 and 3 for Vid: V00852730

train_filter_class= np.where((y_train==0) | (y_train==3))
test_filter_class= np.where((y_test==0) | (y_test==3))
x_train, y_train = x_train[train_filter_class], y_train[train_filter_class]
x_test, y_test = x_test[test_filter_class], y_test[test_filter_class]

#reshaping train and test data
x_train=x_train.astype(dtype='float32')
x_test=x_test.astype(dtype='float32')
y_train=y_train.astype(dtype='float32')
y_train=y_train.reshape(y_train.shape[0],1)
y_test=y_test.astype(dtype='float32')
y_test=y_test.reshape(y_test.shape[0],1)

x_train = x_train.reshape(x_train.shape[0],784) / 255
x_test = x_test.reshape(x_test.shape[0],784) / 255


#changing classes to -1's and 1's

for j in range(y_train.shape[0]):
        if y_train[j]==3:
                y_train[j]=1
        else:
           y_train[j]=-1

for j in range(y_test.shape[0]):
        if y_test[j]==3:
                y_test[j]=1
        else:
                y_test[j]=-1

## we use the dataset with x_train being the matrix "n by fCnt"
##   with samples as rows, and the  features as columns
## y_train is the true value of dependent variable, we have it as a matrix "n by 1"

n_train=x_train.shape[0]
fCnt=x_train.shape[1]

#### START OF LEARNING

#number of epchos. 1 epoch is when all training data is seen

n_epochs=100

#number of samples to present as micro-batch
#could be just n_train
#or if dataset is large, could be a small chunk of if

batch_size=128


# w is the feature weights, a [fCnt x 1] vector

initialW=np.random.rand(fCnt,1).astype(dtype='float32')
w = tf.Variable(initialW,name="w")

# b is the bias, so just a single number

initialB=0.0
b = tf.Variable(initialB,name="b")

# x will be our [#samples x #features] matrix of all training samples

x = tf.placeholder(dtype=tf.float32,name='x')

# y will be our vector of dependent variable for all training samples

y = tf.placeholder(dtype=tf.float32,name='y')

## set up new variables that are functions/transformations of the above

predictions =tf.matmul(x,w) + b

#loss (square error of prediction) for each sample (a vector)

'''
Logistic loss:
wx=tf.matmul(x,w)
loss=tf.log(1+tf.exp(-(tf.multiply(y,wx))))
# This function is giving me nan issues.hence keeping it to original loss function.
'''

loss=tf.square(y-predictions)

#risk over all samples (a number)

risk=tf.reduce_mean(loss)

#define which optimizer to use

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(risk)

# create a tensorflow session and initialize the variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# calculate and print Mean Squared Error on the full test set, using initial (random) w,b
y_pred=sess.run([predictions],feed_dict={x: x_test, y: y_test})
MSE=np.mean(np.square(y_pred-y_test),axis=0)
print(MSE)

#start the iterations of training
#1 epoch == all data samples were presented
for i in range(0,n_epochs):
    #if dataset is large, we want to present it in chunks (called micro-batches)
    for j in range(0,n_train,batch_size):
        jS=j;jE=min(n_train,j+batch_size)
        x_batch=x_train[jS:jE,:]
        y_batch=y_train[jS:jE,:]
        #do a step of gradient descent on the micro-batch
        _,curr_batch_risk,predBatchY=sess.run([train,risk,predictions],feed_dict={x: x_batch, y: y_batch})
    # training done in this epoch
    # but, just so that the user can monitor progress, try current w,b on full test set
    y_pred,curr_w,curr_b=sess.run([predictions,w,b],feed_dict={x: x_test, y: y_test})
    # calculate and print Mean Squared Error
    MSE=np.mean(np.mean(np.square(y_pred-y_test),axis=1),axis=0)
    print(MSE)
print("-----Accuracy-----")
print(100- MSE*100)

import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization


# load MNIST dataset
(x_train_labels, y_train_labels), (x_test_images, y_test_images) = mnist.load_data()

# from sparse label to categorical
num_labels = len(np.unique(y_train_labels))
y_train_labels = to_categorical(y_train_labels)
y_test_images = to_categorical(y_test_images)

# reshape and normalize input images
image_size = x_train_labels.shape[1]
x_train_labels = np.reshape(x_train_labels,[-1, image_size, image_size, 1])
x_test = np.reshape(x_test_images , [-1, image_size, image_size, 1])
x_train = x_train_labels.astype('float32') / 255
x_test = x_test_images.astype('float32') / 255

# network parameters
input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
filters = 64
dropout = 0.3

# use functional API to build cnn layers
inputs = Input(shape = input_shape)
z = Conv2D(filters=filters,
           kernel_size=kernel_size,
           activation='relu')(inputs)
z = MaxPooling2D()(z)

z = Conv2D(filters = filters,
           kernel_size=kernel_size,
           activation='relu')(z)
z = MaxPooling2D()(z)

z = Conv2D(filters = filters,
           kernel_size = kernel_size,
           activation = 'relu')(z)
# image to vector before connecting to dense layer
z = Flatten()(z)
# dropout regularization
z = Dropout(dropout)(z)
outputs = Dense(num_labels, activation = 'softmax')(z)

# build the model by supplying inputs/outputs
modeltf = Model(inputs=inputs, outputs=outputs)
# network model in text
''' modeltf.add(BatchNormalization()) '''
modeltf.summary()

# classifier loss, Adam optimizer, classifier accuracy
modeltf.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# train the model with input images and labels
history = modeltf.fit(x_train_labels,
          y_train_labels,
          validation_data=(x_test, y_test_images),
          epochs = 5,
          batch_size=batch_size)
''' history = modeltf.fit(x_train,y_train, epochs = 5, batch_size=128) '''

# model accuracy on test dataset
score = modeltf.evaluate(x_test_images,
                       y_test_images,
                       batch_size =
                       batch_size,
                       verbose=0)
print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))
#plot
print(history)
x = history.history['loss']
y = np.array([1, 2, 3, 4, 5])
#plt.plot(x, y, 'o', color='blue')
plt.scatter(x, y, label='loss/epoch')
plt.show() 
import tensorflow as tf
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt

#TensorFlow - Getting and Splitting the Dataset
mnist = keras.datasets.mnist
(train_images_tf, train_labels_tf), (test_images_tf, test_labels_tf) = mnist.load_data()
#TensorFlow - Loading the Data

print(train_labels_tf[0])
#TensorFlow - Building the Model
modeltf = keras.Sequential([
    keras.layers.Conv2D(input_shape=(28,28,1), filters=6, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu),
    keras.layers.AveragePooling2D(pool_size=2, strides=2),
    keras.layers.Conv2D(16, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu),
    keras.layers.AveragePooling2D(pool_size=2, strides=2),
    keras.layers.Flatten(),
    keras.layers.Dense(120, activation=tf.nn.relu),
    keras.layers.Dense(84, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
#TensorFlow - Visualizing the Model
modeltf.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
modeltf.summary()

#TensorFlow - Training the Model
train_images = (train_images_tf / 255.0).reshape(train_images_tf.shape[0], 28, 28, 1)
test_images = (test_images_tf / 255.0).reshape(test_images_tf.shape[0], 28, 28 ,1)
train_labels_tensorflow=keras.utils.to_categorical(train_labels_tf)
test_labels_tensorflow=keras.utils.to_categorical(test_labels_tf)
history = modeltf.fit(train_images, train_labels_tensorflow, epochs=5, batch_size=32)

#TensorFlow - Comparing the Results
predictions = modeltf.predict(test_images)
correct = 0
for i, pred in enumerate(predictions):
  if np.argmax(pred) == test_labels_tf[i]:
    correct += 1
print('Test Accuracy of the model on the {} test images: {}% with TensorFlow'.format(test_images_tf.shape[0],
                                                                     100 * correct/test_images_tf.shape[0]))
print(history.history['loss'])
x = history.history['loss']
y = np.array([1, 2, 3, 4, 5])
plt.plot(x, y, 'o', color='blue')
plt.show()                                                                     

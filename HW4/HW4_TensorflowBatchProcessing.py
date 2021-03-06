import tensorflow as tf
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization

#TensorFlow - Getting and Splitting the Dataset
mnist = keras.datasets.mnist
(train_images_tf, train_labels_tf), (test_images_tf, test_labels_tf) = mnist.load_data()
#TensorFlow - Loading the Data

print(train_labels_tf[0])
#TensorFlow - Building the Model
''' modeltf = keras.Sequential([
    keras.layers.Conv2D(input_shape=(28,28,1), filters=6, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu),
    keras.layers.AveragePooling2D(pool_size=2, strides=2),
    # Batch Normalization - TF v2.3
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(16, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu),
    keras.layers.AveragePooling2D(pool_size=2, strides=2),
    # Batch Normalization - TF v2.3
    keras.layers.BatchNormalization(),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
]) '''
encoder_input = keras.Input(shape=(28, 28, 1), name="img")
x = keras.layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = keras.layers.Conv2D(32, 3, activation="relu")(x)
x = keras.layers.MaxPooling2D(3)(x),
x = keras.layers.BatchNormalization()(x),
x = keras.layers.Conv2D(32, 3, activation="relu")(x)
x = keras.layers.Conv2D(16, 3, activation="relu")(x)
x = keras.layers.Dense(64, activation=tf.nn.relu)(x),
x = keras.layers.Dense(64, activation=tf.nn.relu)(x),
x = keras.layers.Dense(10)(x)
encoder_output = keras.layers.GlobalMaxPooling2D()(x)


modeltf = keras.Model(inputs=encoder_input, outputs=encoder_output, name="mnist_model")
#TensorFlow - Visualizing the Model
modeltf.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])
modeltf.summary()

#TensorFlow - Training the Model
train_images = (train_images_tf / 255.0).reshape(train_images_tf.shape[0], 28, 28, 1)
test_images = (test_images_tf / 255.0).reshape(test_images_tf.shape[0], 28, 28 ,1)
train_labels_tensorflow=keras.utils.to_categorical(train_labels_tf)
test_labels_tensorflow=keras.utils.to_categorical(test_labels_tf)
history = modeltf.fit(train_images, train_labels_tensorflow, epochs=5, batch_size=64)

#TensorFlow - Comparing the Results
predictions = modeltf.predict(test_images)
correct = 0
for i, pred in enumerate(predictions):
  if np.argmax(pred) == test_labels_tf[i]:
    correct += 1
print('Test Accuracy of the model on the {} test images: {}% with TensorFlow'.format(test_images_tf.shape[0],
                                                                     100 * correct/test_images_tf.shape[0]))
print(history)
x = history.history['loss']
y = np.array([1, 2, 3, 4, 5])
#plt.plot(x, y, 'o', color='blue')
plt.scatter(x, y, label='loss/epoch')
plt.show()                                                                     

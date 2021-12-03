# Date: Friday December 3rd 2021
# Purpose: This code is written to look at the an image and identify what exactly it is with deep learning
# Author: josephlee94
# Editor and Viewer: Spencer Lage

# Description: After reading and understanding the code below I understand how deep learning and python takes deep
# to a different stage. This code is able to understand and know what images are put in front of it for example in this
# program it was a train and a cat. This program was able to tell the difference between a cat and a train through a
# couple lines of code to get the correct results. Its amazing how python programming is able to do such
# powerful things like being able to tell the differences between two different images and tell exactly what the images
# are! I was unfortunate in not being able to get this program how to work properly but after hours of walking
# through the program and how to execute it and what exactly each line of code is I for sure have a better under
# standing about deep learning with python programming!

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('x_train shape:', x_train.shape)

print('y_train shape:', y_train.shape)

print(x_train[0])

import matplotlib.pyplot as plt
# matplotlib inline

img = plt.imshow(x_train[0])

print('The label is:', y_train[0])

img = plt.imshow(x_train[1])

print('The label is:', y_train[1])

import keras
y_train_one_hot = keras.utils.to_categorical(y_train, 10)
y_test_one_hot = keras.utils.to_categorical(y_test, 10)

print('The one hot label is:', y_train_one_hot[1])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255

x_train[0]

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32,32,3)))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

hist = model.fit(x_train, y_train_one_hot,
           batch_size=32, epochs=20,
           validation_split=0.2)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

model.evaluate(x_test, y_test_one_hot)[1]

model.save('my_cifar10_model.h5')








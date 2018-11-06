# TensorFlow is an open-source software library
#   for dataflow programming across a range of tasks. 
# It is a symbolic math library, and is also used for 
#    machine learning applications such as neural networks. 
# It is used for both research and production at Google, 
#    often replacing its closed-source predecessor, DistBelief.
# 
# Keras is an open source neural network library written in Python. 
# It is capable of running on top of TensorFlow, 
#    Microsoft Cognitive Toolkit, or Theano. 
# Designed to enable fast experimentation with deep neural networks, 
#    it focuses on being user-friendly, modular, and extensible. 
# It was developed as part of the research effort of project 
#    ONEIROS (Open-ended Neuro-Electronic Intelligent Robot Operating System), 
#    and its primary author and maintainer is François Chollet, a Google engineer.
#
# In 2017, Google's TensorFlow team decided to support Keras in TensorFlow's core library.


import numpy as np
import keras
from keras.datasets import mnist

#________________________________________________________

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)

# Flatten
img_size = 28 * 28
x_train = np.reshape(x_train,(-1,img_size))
x_test = np.reshape(x_test,(-1,img_size))

# to float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# divide 255
x_train /= 255
x_test /= 255

print(x_train.shape)

#________________________________________________________

from keras.utils import to_categorical

print(y_train.shape)

num_classes = 10

# to_categorical
y_train = to_categorical(y_train, num_classes)

print(y_train.shape)

#________________________________________________________

import matplotlib.pyplot as plt

def showImg(imgs, labels):
  count = imgs.shape[0]
  fig = plt.figure(figsize=(10, 10))
  for i, (img, label) in enumerate(zip(imgs, labels)):
    p = plt.subplot(count, 1, i + 1)
    p.axis('off')
    p.set_title(label.argmax())
    p.imshow(np.reshape(img,(28, 28)))
  plt.show()

#________________________________________________________

showImg(x_train[:5], y_train[:5])

#________________________________________________________

from keras.models import Sequential
from keras.layers import Input, Dense

# Create model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=[img_size]))
model.add(Dense(num_classes, activation='softmax'))

# compile
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# fit
model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_split = 0.1,
          verbose=1)
#________________________________________________________

x = x_test[:10]

# predict
y = model.predict(x)

showImg(x, y)



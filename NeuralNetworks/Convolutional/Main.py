import matplotlib.pyplot as plt
import keras
import numpy
import random
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

def showInitImage():
  plt.imshow(xTrain[0])

def changeXTRAINshapes(x_train):
  return x_train.reshape(x_train.shape[0], 28, 28, 1)

def changeXTESTshapes(x_test):
  return x_test.reshape(x_test.shape[0], 28, 28, 1)

def constructModel():
  model = Sequential()
  model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape =(28, 28, 1)))
  model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
  model.add(MaxPooling2D(pool_size = (2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation = 'relu'))
  model.add(Dropout(0.5))
  model.add(Dense(10, activation = 'softmax'))
  model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
  model.fit(xTrain, yTrain, epochs = 3)
  modelEval = model.evaluate(xTest, yTest)
  print(modelEval[0],modelEval[1])
  for i in range(0, 20):
    testPredict = model.predict(xTest[i].reshape(1, 28, 28, 1))
    print(testPredict.argmax())
  model.summary()

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
showInitImage()

xTrain = changeXTRAINshapes(xTrain)
xTest = changeXTESTshapes(xTest)

xTrain = xTrain.astype('float32')/255
xTest = xTest.astype('float32')/255

yTrain = keras.utils.to_categorical(yTrain, 10)
yTest = keras.utils.to_categorical(yTest, 10)

constructModel()



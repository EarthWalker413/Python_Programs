import random
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Dense, Activation
import keras.backend as K
from keras.optimizers import SGD
import matplotlib.pyplot as plt

def load_data(filename):
  file = open(filename, 'r')
  data = file.readlines()
  return data

def split_data_elements(data):
  splittedData = list()
  for element in data:
    splittedData.append(element.split())
  return splittedData

def getX(data):
  x = list()
  for element in data:
    x.append(float(element[0]))
  return x

def getY(data):
  y = list()
  for element in data:
    y.append(float(element[1]))
  return y


data = load_data('/content/sample_data/dane11.txt')
splittedData = split_data_elements(data)
print(splittedData)
random.shuffle(splittedData)
print(splittedData)

print(len(splittedData))
trainData = splittedData[0:43]
testData = splittedData[44:61]


#print(trainData)
#print(testData)



#print(x_train)

model = Sequential()
model.add(Dense(50, input_dim=1))
model.add(Activation('relu'))
model.add(Dense(45, ))
model.add(Activation('relu'))
model.add(Dense(1,))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_train = getX(trainData)
print(x_train)
y_train = getY(trainData)
print(y_train)

model.fit(x_train, y_train, epochs=700)

x_test=getX(testData)
y_test=getY(testData)

model.evaluate(x=x_test, y=y_test)

y_pred = model.predict(x_test)

plt.scatter(x_test, y_pred, color='red', label='result')
plt.scatter(x_test, y_test, color='blue', label='actual')


# Plot model graph
plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')
from IPython.display import Image
Image(retina=True, filename='model.png')

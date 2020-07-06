
from csv import reader


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset):
	convList = list()
	for row in dataset:
		clist = list()
		for column in range(len(row)):
			if column != len(row) - 1:
				clist.append(float(row[column].strip()))
			else:
				clist.append(row[column])
		convList.append(clist)
	return(convList)


def manhattan_distance(row1, row2):
	distannce = 0.0
	for i in range(len(row1)-1):
		distannce += abs(row1[i] - row2[i])
	return distannce

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = manhattan_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

def getCorrectnesses(train, test_row, num_neighbors):
	prediction = predict_classification(train, test_row, num_neighbors)
	correctAnswers = 0
	if prediction == test_row[len(test_row) - 1]:
		correctAnswers += 1
	return correctAnswers


# Test the kNN on the Iris Flowers dataset
trainFileMame = 'heart.csv'
testFileName = 'heart_test.csv'

trainSet = load_csv(trainFileMame)
print(trainSet)
testSet = load_csv(testFileName)

formattedTrainSet = str_column_to_float(trainSet)


formattedTestSet = str_column_to_float(testSet)



try:
	print('Enter a number of neighbours:')
	number = int(input())
except:
	print('you have netered wrong statement so a number will be 10')
	number = 10

correctAnswers = 0.0

for row in formattedTestSet:
	correctAnswers += getCorrectnesses(formattedTrainSet, row, number)

print('correct: ')
print(correctAnswers/float(len(testSet)) * 100)

#logistic regression
model = LogisticRegression(solver='liblinear', random_state=0)


def getInputArray(dataset):
	convList = list()
	for row in dataset:
		clist = list()
		for column in range(len(row)):
			if column != len(row) - 1:
				clist.append(float(row[column].strip()))
		convList.append(clist)
	return (convList)

def getOutputArray(dataset):
	convList = list()
	for row in dataset:

		for column in range(len(row)):
			if column == len(row) - 1:
				convList.append(float(row[column].strip()))
	return (convList)

inputArray = getInputArray(trainSet)
print(inputArray)

outputArray = getOutputArray(trainSet)
print(outputArray)
model.fit(inputArray, outputArray)

print(model.classes_)
print(model.predict_proba(inputArray))
print(model.predict(inputArray))
print(model.score(inputArray,outputArray))


cm = confusion_matrix(outputArray, model.predict(inputArray))

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import mglearn
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()


# print(f"Keys of IRIS DATASET: {iris_dataset.keys()}")
# Keys of IRIS DATASET: dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module']

# print(f"Description: {iris_dataset['DESCR']}")
# :Number of Instances: 150 (50 in each of three classes)
# :Number of Attributes: 4 numeric, predictive attributes and the class
# :Attribute Information:
#     - sepal length in cm
#     - sepal width in cm
#     - petal length in cm
#     - petal width in cm
#     - class:
#             - Iris-Setosa			= 0
#             - Iris-Versicolour	= 1
#             - Iris-Virginica		= 2

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# print(f"X_train shape: {X_train.shape}")
# print(f"X_test shape: {X_test.shape}")
# print(f"y_train shape: {y_train.shape}")
# print(f"y_test shape: {y_test.shape}")

# X_train shape: (112, 4)
# X_test shape: (38, 4)
# y_train shape: (112,)
# y_test shape: (38,)

def find_best_n():
	# The accuracy is the same until n=23 and then it drops.
	for i in range(1, 30):
		knn = KNeighborsClassifier(n_neighbors=i)
		knn.fit(X_train, y_train)

		y_pred = knn.predict(X_test)
		score = knn.score(X_test, y_test)

		print(f"{i}: {score*100}%")

def predict(sepal_length, sepal_width, petal_length, petal_width):
	knn = KNeighborsClassifier(n_neighbors=1)
	knn.fit(X_train, y_train)

	pred = knn.predict(np.array([[sepal_length, sepal_width, petal_length, petal_width]]))
	name = iris_dataset['target_names'][pred]

	print(f"Predicted Iris Class: {name[0]}")

# predict(5, 2.9, 1, 0.2)
# Setosa

if __name__ == '__main__':
	sepal_length = input("Sepal Length (in cm): ")
	sepal_width = input("Sepal Width (in cm): ")
	petal_length = input("Petal Length (in cm): ")
	petal_width = input("Petal Width (in cm): ")
	predict(sepal_length, sepal_width, petal_length, petal_width)
import numpy as np
import matplotlib.pyplot as plt
import mglearn
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

cancer_dataset = load_breast_cancer()

# print(f"Keys: {cancer_dataset.keys()}")
# ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module']

# print(f"Shape: {cancer_dataset.data.shape}")
# Shape: (569, 30)

# print("Sample counts per class:\n{}".format({n: v for n, v in zip(cancer_dataset.target_names, np.bincount(cancer_dataset.target))}))
# Sample counts per class:
# {np.str_('malignant'): np.int64(212), np.str_('benign'): np.int64(357)}

# print(f"Feature Names: {cancer_dataset.feature_names}")
# Feature Names: ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']

# print(f"Description: {cancer_dataset.DESCR}")

X_train, X_test, y_train, y_test = train_test_split(cancer_dataset.data, cancer_dataset.target, 
													stratify=cancer_dataset.target, random_state=66)

train_acc = []
test_acc = []

for n_neighbor in range(1, 10):
	clf = KNeighborsClassifier(n_neighbors=n_neighbor).fit(X_train, y_train)
	train_accuracy = clf.score(X_train, y_train)
	test_accuracy = clf.score(X_test, y_test)

	train_acc.append(train_accuracy)
	test_acc.append(test_accuracy)



plt.plot(train_acc, label="Train Accuracy")
plt.plot(test_acc, label="Test Accuracy")

plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")

plt.legend()
plt.savefig("breast-cancer.png")

# Best result around 5

import numpy as np
import matplotlib.pyplot as plt
import mglearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

print("Waves")
X, y = mglearn.datasets.make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)

print(f"lr.coef_: {lr.coef_}")
print(f"le.intercept_: {lr.intercept_}")

# lr.coef_: [0.47954524]
# le.intercept_: -0.09847983994403892


print("Boston Dataset")
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = LinearRegression().fit(X_train, y_train)

print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

# Training set score: 0.94
# Test set score: 0.78

# This discrepancy between performance on the training set and the test set is a clear
# sign of overfitting, and therefore we should try to find a model that allows us to con‐
# trol complexity

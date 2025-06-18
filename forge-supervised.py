import numpy as np
import matplotlib.pyplot as plt
import mglearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knc

# FORGE DATASET
# Making the dataset
X, y = mglearn.datasets.make_forge()
# # Plotting the dataset
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.legend()
# plt.xlabel("First Feature")
# plt.ylabel("Second Feature")
# print(f"X.shape: {X.shape}")
# plt.savefig("forge-data.png")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = knc(n_neighbors=3)
clf.fit(X_train, y_train)

print(f"{X_test}")
print(f"{y_test}")
print(f"{clf.predict(X_test)}")

print(f"Test score accuracy: {clf.score(X_test, y_test)}")

# # WAVE DATASET
# # For regression models
# X, y = mglearn.datasets.make_wave(n_samples=40)
# plt.plot(X, y, 'o')
# plt.ylim(-3, 3)
# plt.xlabel("Feature")
# plt.ylabel("Target")
# plt.savefig("wave-data.png")

def graphs():
	fig, axes = plt.subplots(1, 3, figsize=(10, 3))
	for n_neighbors, ax in zip([1, 3, 9], axes):
		# the fit method returns the object self, so we can instantiate
		# and fit in one line
		clf = knc(n_neighbors=n_neighbors).fit(X, y)
		mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
		mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
		ax.set_title("{} neighbor(s)".format(n_neighbors))
		ax.set_xlabel("feature 0")
		ax.set_ylabel("feature 1")
	axes[0].legend(loc=3)
	fig.savefig("forge.png")

graphs()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from matplotlib.colors import ListedColormap

# Define means and covariances for two classes
mean1 = [1, 1]
cov1 = [[1, 0.5], [0.5, 1]]  # covariance matrix for class 1

mean2 = [4, 4]
cov2 = [[1, -0.5], [-0.5, 1]]  # covariance matrix for class 2

# Generate random data for each class
np.random.seed(42)
class1_data = np.random.multivariate_normal(mean1, cov1, 100)
class2_data = np.random.multivariate_normal(mean2, cov2, 100)

# Combine data from both classes
X = np.vstack((class1_data, class2_data))
y = np.hstack((np.zeros(100), np.ones(100)))  # Labels: 0 for class 1, 1 for class 2

# Train a Gaussian Naive Bayes classifier
clf = GaussianNB()
clf.fit(X, y)

# Define a mesh grid to plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Predict the class for each point in the mesh grid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Plot the decision boundary
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(('red', 'blue')))

# Plot the data points
plt.scatter(class1_data[:, 0], class1_data[:, 1], color='red', label='Class 1')
plt.scatter(class2_data[:, 0], class2_data[:, 1], color='blue', label='Class 2')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary of Gaussian Naive Bayes Classifier')
plt.legend()
plt.show()

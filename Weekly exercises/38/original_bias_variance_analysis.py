import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# PARAMETERS
n = 1000  # no. data points
maxdegree = 15  # max degrees of polynomial


def bias(ideal, actual):
    return np.mean((ideal - np.mean(actual))**2)


def variance(array):
    return np.var(array)


# Create data set
np.random.seed(2023)
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = x**4 - 16 * x**2 + 50 * x  # +  np.random.normal(0, 0.1, x.shape)

# Create arrays
degrees = np.arange(0, maxdegree, 1)  # loop from polynomial degree 1 to maxdegree
mse_arr = np.zeros_like(degrees)
bias_arr = np.zeros_like(degrees)
variance_arr = np.zeros_like(degrees)

linreg = LinearRegression(fit_intercept=False)  # include intercept

# Create design matrix first
X = np.zeros((n, maxdegree))
for degree in degrees:
    X[:, degree] = x[:, 0]**degree

# Then loop over the degrees again to do the mean error calculations
for degree in degrees:
    # Split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X[:, :degree + 1], y, test_size=0.2)
    linreg.fit(X_train, y_train)
    y_predict = linreg.predict(X_test)
    mse_arr[degree] = mean_squared_error(y_test, y_predict)
    bias_arr[degree] = bias(y_test, y_predict)
    variance_arr[degree] = variance(y_predict)

# Plotting
plt.plot(degrees, mse_arr, label="MSE")
plt.plot(degrees, bias_arr, label="Bias")
plt.plot(degrees, variance_arr, label="Variance")
plt.legend()
plt.xlabel("Polynomial degree")
plt.ylabel("Value")
plt.show()
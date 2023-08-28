"""
Created on 26.08.2023
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Data set
np.random.seed()
n = 100
maxdegree = 15  # 15th degree polynomial
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape)

"""task a) and b): 5th order polynomial"""
# Create design matrix
X = np.zeros((n, 6))
X[:, 0] = 1
X[:, 1] = x[:, 0]
X[:, 2] = x[:, 0]**2
X[:, 3] = x[:, 0]**3
X[:, 4] = x[:, 0]**4
X[:, 5] = x[:, 0]**5

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lin. regression. Not scaled, then scaled
beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
y_tilde = X_train @ beta
y_predict = X_test @ beta
"""I get an error when i try to create a beta_scaled with the scaled matrix, so the scores are wrong"""
y_tilde_scaled = X_train_scaled @ beta
y_predict_scaled = X_test_scaled @ beta

# Metrics. Not scaled, then scaled
print(f"TASK A)\n----------------------")
print(f"(Not scaled)")
print(f"Train MSE: {mean_squared_error(y_train, y_tilde):g}")
print(f"Train R2: {r2_score(y_train, y_tilde):g}")
print(f"Test MSE: {mean_squared_error(y_test, y_predict):g}")
print(f"Test R2: {r2_score(y_test, y_predict):g}")
print(f"\n(Scaled)")
print(f"Train MSE: {mean_squared_error(y_train, y_tilde_scaled):g}")
print(f"Train R2: {r2_score(y_train, y_tilde_scaled):g}")
print(f"Test MSE: {mean_squared_error(y_test, y_predict_scaled):g}")
print(f"Test R2: {r2_score(y_test, y_predict_scaled):g}")

"""Task c) 15th order polynomial"""
mse_train = np.zeros(maxdegree)
r2_train = np.zeros(maxdegree)
mse_test = np.zeros(maxdegree)
r2_test = np.zeros(maxdegree)

linreg = LinearRegression(fit_intercept=False)
polyit = PolynomialFeatures(maxdegree)
for degree in range(1, maxdegree+1):
    # Split into training and test data
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    y_tilde = X_train @ beta
    y_predict = X_test @ beta

    mse_train[degree] = mean_squared_error(y_train, y_tilde)
    r2_train[degree] = r2_score(y_train, y_tilde)
    mse_test[degree] = mean_squared_error(y_test, y_predict)
    r2_test[degree] = r2_score(y_test, y_predict)

# Plot mse and r2 to the degree of poly.
plt.plot(range(maxdegree), mse_train, label="Train MSE")
plt.plot(range(maxdegree), mse_test, label="Test MSE")
plt.plot(range(maxdegree), r2_train, label="Train R2")
plt.plot(range(maxdegree), r2_test, label="Test R2")
plt.legend()
plt.xlabel("Degree of polynomial")
plt.ylabel("Error")
plt.show()

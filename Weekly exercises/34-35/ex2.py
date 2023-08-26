"""
Created on 22.08.2023
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Data
n = 100  # no. data points
x = np.random.rand(n, 1)
y = 2.0 + 5 * x * x + 0.1 * np.random.randn(n, 1)

# Own code
X = np.zeros((n, 3))
X[:, 0] = 1
X[:, 1] = x[:, 0]
X[:, 2] = x[:, 0]**2

beta = np.linalg.inv(X.T @ X) @ X.T @ y
y_tilde = X @ beta

# Scikit-learn
"""i dont really understand if im supposed to compare my polyfit to scikit's linear regression...
 but thats what the examples show and the tasks says to do it from the examples"""
linreg = LinearRegression(fit_intercept=False)
linreg.fit(x, y)
y_skl = linreg.predict(x)

# Metrics
print("OWN CODE\n-----------------------")
print(f"MSE: {mean_squared_error(y, y_tilde):g}")
print(f"R2: {r2_score(y, y_tilde):g}")
print("\nSCIKIT-LEARN\n---------------------")
print(f"MSE: {mean_squared_error(y, y_skl):g}")
print(f"R2 score: {r2_score(y, y_skl):g}")
